import json
import os
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "..."

# LLM 설정
llm = OpenAI(temperature=0.7)

# 텍스트 보정 및 목적지 추출을 위한 프롬프트 템플릿
correction_prompt = PromptTemplate.from_template(
    """
    다음 텍스트는 택시 앱에서 사용자가 말한 목적지입니다. 
    이 텍스트를 분석하여 다음 작업을 수행해주세요:

    1. 맥락상 맞지 않는 부분이 있다면 보정해주세요.
    2. 발음이 어눌하거나 망가진 부분이 있다면 올바르게 수정해주세요.
    3. 목적지에 해당하는 단어만 추출해주세요. (예: "서울역에 가고 싶어"에서 "서울역"만 추출)

    원본 텍스트: {original_text}

    보정된 목적지:
    """
)

# 텍스트 보정 체인
correction_chain = correction_prompt | llm | StrOutputParser()

# 메인 대화 프롬프트 템플릿
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

class TaxiChatBot:
    def __init__(self):
        self.main_chain = main_chain
        self.correction_chain = correction_chain
        self.chat_history = ""
        self.destination = None
        self.confirmed = False

    def correct_destination(self, text):
        corrected = self.correction_chain.invoke({"original_text": text}).strip()
        
        if not corrected or corrected == text.strip():
            return text
        
        return corrected

    def process_message(self, message):
        if not self.destination:
            corrected_destination = self.correct_destination(message)
            self.destination = corrected_destination.strip()
            response = self.main_chain.invoke({"human_input": message, "chat_history": self.chat_history})
            self.chat_history += f"Human: {message}\nAI: {response}\n"
            return f"{response}\n{self.destination}(이)가 맞을까요?"

        if not self.confirmed:
            positive_responses = ["네", "맞아", "응", "어", "마자", "마저", "그래"]
            negative_responses = ["아니", "아녀", "아뇨", "아니야", "노"]

            if any(pos_response in message for pos_response in positive_responses):
                self.confirmed = True
                return f"목적지 {self.destination}(으)로 정확한 위치 검색을 시작하겠습니다."
            elif any(neg_response in message for neg_response in negative_responses):
                self.destination = None
                return "죄송합니다. 다시 목적지를 말씀해 주세요."
            else:
                return "응답을 이해하지 못했습니다. '네' 또는 '아니오'로 대답해 주세요."

bot = TaxiChatBot()

# Lambda 핸들러 함수
def lambda_handler(event, context):
    message = json.loads(event['body'])['message']
    response = bot.process_message(message)

    return {
        'statusCode': 200,
        'body': json.dumps({'response': response})
    }

# 로컬 테스트용 main 함수
def main():
    bot = TaxiChatBot()
    print("AI: 안녕하세요, 어디로 가고 싶으세요?")

    while True:
        user_input = input("사용자: ")
        response = bot.process_message(user_input)
        print(f"AI: {response}")

        if bot.confirmed:
            break

if __name__ == "__main__":
    main()
