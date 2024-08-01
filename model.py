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
from openai import OpenAI as ai

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

# 메인 프롬프트
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

# 목적지 보정 프롬프트
correction_prompt = PromptTemplate.from_template(
    """
    다음 텍스트는 택시 앱에서 사용자가 말한 목적지입니다.
    이 텍스트에 다음 작업을 수행해 목적지 출력을 완료해주세요.

    1. 발음 교정: 발음이 어눌하거나, 맥락에 맞지 않게 센 발음이거나, 어색하거나 잘못된 부분이 있다면 올바르게 수정해 주세요.
    2. 한국의 장소 중 한 곳이라는 점을 고려해서 단어를 교정해주세요.
    3. 목적지 단어 추출: 목적지에 해당하는 단어만 추출해 주세요. 예를 들어, "서울역에 가고 싶어"에서는 "서울역"만 추출합니다.
    4. 지하철역 추출: 한국의 지하철역에 해당한다고 하면 지하철역 정보 전체를 추출해주세요. 예를 들어, "홍대입구역 9번 출구"는 "홍대입구역 9번 출구", "홍대입구역에 가고 싶어"는 "홍대입구역"으로 추출해주세요.
    5. 영어에 해당하는 단어라면 영어로 변경해주세요. 예를 들어, "스타벅스 신월디티점"은 "스타벅스 DT점"으로, "지에스"는 "GS"로, "디디피"는 "DDP"로, "씨지비"는 "CGV"로 수정합니다.
    6. 체인점 지점 확인: 만약 목적지가 체인점(예: 스타벅스, 맥도날드 등)이고 지점을 언급했다면, 체인점 이름과 지점 정보를 모두 포함해 주세요. 예를 들어, "맥도날드 명지대점"은 "맥도날드 명지대점", "강남역 근처 스타벅스"는 "스타벅스 강남역점", "하남에 있는 스타필드"는 "스타필드 하남점"으로 응답해 주세요.
    7. 단어 중복 방지: 목적지 단어를 중복하지 말고, 정확하게 추출된 단어만 응답해 주세요. 중복된 단어는 제외해 주세요.
    8. 만약 목적지를 찾을 수 없거나 추출할 수 없는 경우 XXXXX를 최종 목적지로 해주세요.
    9. 응답 형식: 최종 목적지 단어만 명확하게 제공해 주세요.

    원본 텍스트: {original_text}
    
    최종 목적지:
    """
)

# 목적지 체인
correction_chain = correction_prompt | llm | StrOutputParser()

# 지점 보정
branch_prompt = PromptTemplate.from_template(
    """
    다음 텍스트 내부에는 목적지가 체인점인 경우 지점을 물어본 경우라, 지점에 관한 정보가 들어 있습니다.
    이 텍스트에 다음 작업을 수행해 지점 출력을 완료해주세요.

    1. 발음 교정: 발음이 어눌하거나, 맥락에 맞지 않게 센 발음이거나, 잘못된 부분이 있다면 올바르게 수정해 주세요.
    2. 지점 추출: 지점에 해당하는 단어만 추출해 주세요. 
       예를 들어, "하남에 있는 스타필드"는 "하남점"으로, "스타필드 하남점"은 "하남점"으로, "맥도날드 명지대점"은 "명지대점"으로 응답해 주세요.
    3. 응답 형식: 최종 지점 정보만 명확하게 제공해 주세요. 예를 들어, "추출된 지점: 명지대점"은 "명지대점"으로 응답해 주세요.

    원본 텍스트: {original_text}

    보정된 지점:
    """
)

# 지점 체인
branch_chain = branch_prompt | llm | StrOutputParser()

# 지하철역 출구 보정
exit_prompt = PromptTemplate.from_template(
    """
    다음 텍스트 내부에는 지하철역의 출구 번호에 대한 정보가 들어있습니다.
    이 텍스트에 다음 작업을 수행해 몇번 출구인지를 응답해주세요.

    1. 텍스트 내부에 지하철역과 출구 번호가 모두 포함된 경우, 출구 번호만 추출해주세요. 예를 들어, "김포공항역 9번 출구"는 "9번 출구"로, "홍대입구역 1번 출구"는 "1번 출구"로 응답해주세요.
    2. 응답은 오직 "3번 출구"의 형태로만 해주세요. 만약 "1번"이라는 텍스트가 인식되어도 반드시 "1번 출구"로 변경해 응답해주세요.
    3. 만약 이미 텍스트가 "9번 출구"의 형태인 경우 그대로 응답해주세요.
    
    원본 텍스트: {original_text}
    
    보정된 출구:
    """
)

# 지하철역 체인
exit_chain = exit_prompt | llm | StrOutputParser()

class TaxiChatBot:
    def __init__(self):
        self.main_chain = main_chain
        self.correction_chain = correction_chain
        self.chat_history = ""
        self.destination = None
        self.confirmed = False
        self.chain_stores = ['스타벅스', '맥도날드', '버거킹', '이디야', '올리브영', 'CU', '씨유', '지에스', '지에스이십오', '지에스25', 'GS25', 'GS', '더현대', '현대백화점', '롯데백화점', '롯데월드', '서브웨이', '투썸플레이스', '투썸', '올드페리도넛', '스타필드', '다이소', '비비큐', '교촌치킨', '던킨', '던킨도넛', '던킨도너츠', 'CGV', '씨지비', ]
        self.asking_branch = False
        self.asking_exit = False
        self.branch_info = ""
        self.exit_info = ""

     def correct_destination(self, text):
        corrected = correction_chain.invoke({"original_text": text}).strip()
        return corrected

    def correct_branch(self, text):
        corrected = branch_chain.invoke({"original_text": text}).strip()
        return corrected

    def correct_exit(self, text):
        corrected = exit_chain.invoke({"original_text": text}).strip()
        return corrected

    def process_message(self, message):
        if not self.destination or self.destination == "XXXXX" or "없음" in self.destination:
            corrected_destination = self.correct_destination(message)
            cleaned_destination = corrected_destination.strip()
            last_colon_index = cleaned_destination.rfind(':')
            if last_colon_index != -1:
                self.destination = cleaned_destination[last_colon_index + 1:].strip()
            else:
                self.destination = cleaned_destination
            self.destination = self.destination.replace('"', '')
            print(self.destination)

            if self.destination.strip() == "XXXXX" or "없음" in self.destination:
                response = "목적지를 인식할 수 없어요. 다시 한 번 목적지를 말씀해 주세요."
            elif self.is_subway(self.destination):
                if not '출구' in self.destination:
                    self.asking_exit = True
                    response = f"{self.destination} 몇 번 출구인지 말씀해주세요."
                else:
                    response = f"{self.destination}(이)가 맞을까요?"
            elif self.is_chain_store(self.destination):
                self.asking_branch = True
                response = f"{self.destination}의 지점을 말씀해주세요."
            else:
                response = f"{self.destination}(이)가 맞을까요?"

        elif self.asking_branch:
            corrected_branch = self.correct_branch(message)
            cleaned_branch = corrected_branch.strip()
            last_colon_index = cleaned_branch.rfind(':')
            if last_colon_index != -1:
                self.branch = cleaned_branch[last_colon_index + 1:].strip()
            else:
                self.branch = cleaned_branch
            self.branch_info = ' ' + self.branch.strip()
            response = f"{self.destination}{self.branch_info}(이)가 맞을까요?"
            self.asking_branch = False
        elif self.asking_exit:
            corrected_exit = self.correct_exit(message)
            cleaned_exit = corrected_exit.strip()
            last_colon_index = cleaned_exit.rfind(':')
            if last_colon_index != -1:
                self.exit = cleaned_exit[last_colon_index + 1:].strip()
            else:
                self.exit = cleaned_exit
            self.exit_info = ' ' + self.exit.strip()
            response = f"{self.destination}{self.exit_info}(이)가 맞을까요?"
            self.asking_exit = False
        elif not self.confirmed:
            positive_responses = ["네", "맞아", "응", "어", "마자", "마저", "그래", "웅", "예", "맞아요"]
            negative_responses = ["아니", "아녀", "아뇨", "아니야", "노", "아니요", "아니오", "아닌데", "아니라", "아니여", "아닌디", "그게 아니라"]

            if any(pos_response in message for pos_response in positive_responses):
                self.confirmed = True
                # 기본적으로 destination으로 설정
                self.final_destination = self.destination.strip()

                # branch_info가 빈 문자열이 아닌 경우 destination에 추가
                if self.branch_info.strip():
                    self.final_destination += f" {self.branch_info.strip()}"

                # exit_info가 빈 문자열이 아닌 경우 destination에 추가
                if self.exit_info.strip():
                    self.final_destination += f" {self.exit_info.strip()}"

                # 최종 결과를 다시 한번 trim
                self.final_destination = self.final_destination.strip()

                response = f"목적지 {self.final_destination}(으)로 정확한 위치 검색을 시작하겠습니다."
                self.asking_branch = False
                return response, self.final_destination, True
            elif any(neg_response in message for neg_response in negative_responses):
                self.destination = None
                self.branch_info = ""
                self.exit_info = ""
                response = "죄송합니다. 다시 목적지를 말씀해 주세요."
            else:
                response = "응답을 이해하지 못했어요. '네' 또는 '아니오'로 대답해 주세요."
        else:
            # 기본적으로 destination으로 설정
            self.final_destination = self.destination.strip()

            # branch_info가 빈 문자열이 아닌 경우 destination에 추가
            if self.branch_info.strip():
                self.final_destination += f" {self.branch_info.strip()}"

            # exit_info가 빈 문자열이 아닌 경우 destination에 추가
            if self.exit_info.strip():
                self.final_destination += f" {self.exit_info.strip()}"

            # 최종 결과를 다시 한번 trim
            self.final_destination = self.final_destination.strip()
            response = f"이미 목적지가 {self.final_destination}(으)로 확정되었습니다."

        return response, None, False

    def is_chain_store(self, destination):
        return destination in self.chain_stores

    def is_subway(self, destination):
        if not '역' in destination:
            print('역이라는 단어가 안들어있음')
            return False

        api_key = '...'

        station_name = destination.split('역')[0].strip()

        URL = f'http://openAPI.seoul.go.kr:8088/{api_key}/xml/SearchInfoBySubwayNameService/1/5/{station_name}/'

        try:
            response = requests.get(URL)
            response.raise_for_status()  # 요청이 실패하면 예외 발생

            # XML 응답 파싱
            root = ET.fromstring(response.content)

            # 응답 코드 확인
            code = root.find('.//CODE').text

            if code == 'INFO-000':
                return True
            else:
                return False
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return False
        except ET.ParseError as e:
            print(f"XML parse error: {e}")
            return False
        finally:
            print('finally')

    def extract_after_last_colon(self, reply):
        cleaned_text = reply.strip()

        last_colon_index = cleaned_text.rfind(':')

        if last_colon_index != -1:
            return cleaned_text[last_colon_index + 1:].strip()
        else:
            return cleaned_text

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
        print(reply)

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
    # eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 5001)), app)
