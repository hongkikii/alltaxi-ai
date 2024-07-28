!pip install langchain openai

from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
import os

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "..."

# 프롬프트 템플릿 정의
template = """
당신은 노년층을 위한 택시 앱의 대화 에이전트입니다. 다음과 같은 대화를 진행하세요:

1. 사용자에게 목적지를 물어보세요.
2. 사용자가 말한 목적지를 확인하세요.
3. 사용자의 응답이 긍정적이면 목적지를 확정하고, 부정적이면 다시 물어보세요.

이전 대화:
{chat_history}

인간: {human_input}
AI: """

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template
)

# 메모리 설정
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# LLM 체인 설정
llm = OpenAI(temperature=0.7)
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

class TaxiChatBot:
    def __init__(self):
        self.conversation = conversation
        self.destination = None
        self.confirmed = False

    def process_message(self, message):
        response = self.conversation.predict(human_input=message)

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

# 대화에 사용
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
