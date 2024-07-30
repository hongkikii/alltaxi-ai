import json
import base64
from flask import Flask, request
from flask_socketio import SocketIO, emit, disconnect
from google.cloud import speech
from google.oauth2 import service_account
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from flask_cors import CORS
import logging
import eventlet.wsgi

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/')
def index():
    return "Hello, World!"


# Google Cloud 인증 설정
file_path = '/home/ec2-user/xxx.json'

# JSON 파일 읽기
with open(file_path, 'r') as file:
    credentials_json = json.load(file)
credentials = service_account.Credentials.from_service_account_info(credentials_json)
speech_client = speech.SpeechClient(credentials=credentials)

# OpenAI 설정
openai_api_key = '...'
llm = OpenAI(api_key=openai_api_key, temperature=0.7)

# 프롬프트 템플릿
main_prompt = PromptTemplate.from_template(
    """
    당신은 노년층을 위한 택시 앱의 대화 에이전트입니다. 다음과 같은 대화를 진행하세요:
    1. 사용자에게 목적지를 물어보세요.
    2. 사용자가 말한 목적지를 확인하세요.
    3. 사용자의 응답이 긍정적이면 목적지를 확정하고, 부정적이면 다시 물어보세요.
    이전 대화:
    {chat_history}
    인간: {human_input}
    AI: 
    """
)

# 메인 대화 체인
main_chain = (
    {"human_input": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
    | main_prompt
    | llm
    | StrOutputParser()
)

# 텍스트 보정 및 목적지 추출을 위한 프롬프트 템플릿
correction_prompt = PromptTemplate.from_template(
    """
    다음 텍스트는 택시 앱에서 사용자가 말한 목적지입니다.
    이 텍스트에 다음 작업을 수행해 목적지 출력을 완료해주세요.

    1. 문맥 확인: 목적지와 관련이 없거나 의미가 불분명한 부분, 맥락이 어색한 부분이 있다면 수정해 주세요.
    2. 발음 교정: 발음이 어눌하거나, 맥락에 맞지 않게 센 발음이거나, 잘못된 부분이 있다면 올바르게 수정해 주세요.
    3. 목적지 단어 추출: 목적지에 해당하는 단어만 추출해 주세요. 예를 들어, "서울역에 가고 싶어"에서는 "서울역"만 추출합니다.
    4. 체인점 지점 확인: 만약 목적지가 체인점(예: 스타벅스, 맥도날드 등)이고 지점을 언급했다면, 체인점 이름과 지점 정보를 모두 포함해 주세요. 예를 들어, "강남역 근처 스타벅스"는 "스타벅스 강남역점"으로 응답해 주세요.
    5. 단어 중복 방지: 목적지 단어를 중복하지 말고, 정확하게 추출된 단어만 응답해 주세요. 중복된 단어는 제외해 주세요.
    6. 응답 형식: 위의 작업을 완료한 후, 최종 목적지 단어만 명확하게 제공해 주세요. 예를 들어, "추출된 목적지: 봉대박 파스타"는 "봉대박 파스타"로 응답해 주세요.

    원본 텍스트: {original_text}

    보정된 목적지:
    """
)

# 텍스트 보정 체인
correction_chain = correction_prompt | llm | StrOutputParser()

class TaxiChatBot:
    def __init__(self):
        self.main_chain = main_chain
        self.correction_chain = correction_chain
        self.chat_history = ""
        self.destination = None
        self.confirmed = False
        self.chain_stores = ['스타벅스', '맥도날드', '버거킹', '이디야', '올리브영', 'CU', 'GS25', '더현대', '현대백화점', '롯데백화점', '롯데월드', '서브웨이', '투썸플레이스', '투썸', '올드페리도넛', '스타필드']
        self.asking_branch = False
        self.branch_info = ""

    def correct_destination(self, text):
        corrected = correction_chain.invoke({"original_text": text}).strip()
        return corrected

    def process_message(self, message):
        if not self.destination:
            corrected_destination = self.correct_destination(message)
            self.destination = corrected_destination.strip()

            if self.is_chain_store(self.destination):
                self.asking_branch = True
                response = f"{self.destination}(이)가 맞나요? 지점을 말해주실 수 있나요?"
            else:
                response = f"{self.destination}(이)가 맞을까요?"
        elif self.asking_branch:
            self.branch_info = message.strip()
            response = f"{self.destination} {self.branch_info}(이)가 맞을까요?"
            self.asking_branch = False
        elif not self.confirmed:
            positive_responses = ["네", "맞아", "응", "어", "마자", "마저", "그래"]
            negative_responses = ["아니", "아녀", "아뇨", "아니야", "노"]

            if any(pos_response in message for pos_response in positive_responses):
                self.confirmed = True
                response = f"목적지 {self.destination} {self.branch_info}(으)로 정확한 위치 검색을 시작하겠습니다."
                self.final_destination = f"{self.destination} {self.branch_info}".strip()
                self.asking_branch = False
                return response, self.final_destination, True
            elif any(neg_response in message for neg_response in negative_responses):
                self.destination = None
                self.branch_info = ""
                response = "죄송합니다. 다시 목적지를 말씀해 주세요."
            else:
                response = "응답을 이해하지 못했습니다. '네' 또는 '아니오'로 대답해 주세요."
        else:
            response = f"이미 목적지가 {self.destination} {self.branch_info}(으)로 확정되었습니다."

        return response, None, False

    def is_chain_store(self, destination):
        return destination in self.chain_stores

# 세션 데이터 저장을 위한 딕셔너리
session_data_storage = {}

@socketio.on('connect')
def handle_connect():
    print("WebSocket connected")
    emit('response', {'message': '안녕하세요, 어디로 가고 싶으세요?'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    session_id = request.sid
    if session_id in session_data_storage:
        del session_data_storage[session_id]

@socketio.on('message')
def handle_message(data):
    session_id = request.sid
    if session_id not in session_data_storage:
        session_data_storage[session_id] = TaxiChatBot()

    bot = session_data_storage[session_id]

    if 'audio_data' in data:
        audio_data = base64.b64decode(data['audio_data'])
        message = convert_audio_to_text(audio_data)
        print(message)
    else:
        message = data.get('message', '')

    reply, final_destination, is_ended = bot.process_message(message)

    if is_ended:
        emit('response', {'message': reply})
        emit('response', {'destination': final_destination})
        print(final_destination)
        handle_disconnect()
    else:
        emit('response', {'message': reply})

def convert_audio_to_text(audio_data):
    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR"
    )
    response = speech_client.recognize(config=config, audio=audio)
    return response.results[0].alternatives[0].transcript if response.results else ""

if __name__ == '__main__':
    logging.info("Starting the server...")
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 5000)), app)
