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

# 프롬프트 템플릿 정의
prompt = PromptTemplate.from_template(
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

# 체인 구성
chain = (
        {"human_input": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)


class TaxiChatBot:
    def __init__(self):
        self.chain = chain
        self.chat_history = ""
        self.destination = None
        self.confirmed = False

    def process_message(self, message):
        response = self.chain.invoke({"human_input": message, "chat_history": self.chat_history})
        self.chat_history += f"Human: {message}\nAI: {response}\n"

        if not self.destination:
            self.destination = message
            return f"{response}\n{self.destination}(이)가 맞을까요?"

        if not self.confirmed:
            if "네" in message or "맞아" in message or "응" in message:
                self.confirmed = True
                return f"목적지 {self.destination}(으)로 정확한 위치 검색을 시작하겠습니다."
            elif "아니" in message:
                self.destination = None
                return "죄송합니다. 다시 목적지를 말씀해 주세요."
            else:
                return "응답을 이해하지 못했습니다. '네' 또는 '아니오'로 대답해 주세요."

        return response

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
# def main():
#     bot = TaxiChatBot()
#     print("AI: 안녕하세요, 어디로 가고 싶으세요?")
#
#     while True:
#         user_input = input("사용자: ")
#         response = bot.process_message(user_input)
#         print(f"AI: {response}")
#
#         if bot.confirmed:
#             break
#
#
# if __name__ == "__main__":
#     main()