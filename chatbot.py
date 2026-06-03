from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro", task="text-generation")

model = ChatHuggingFace(llm=llm)

chat_histroy = [SystemMessage(content="You are a helpful AI assistant")]

while True:
    user_input = input("You : ")
    chat_histroy.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    result = model.invoke(chat_histroy)
    chat_histroy.append(AIMessage(content=result.content))
    print("AI : ", result.content)

print(chat_histroy)
