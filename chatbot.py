from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro", task="text-generation")

model = ChatHuggingFace(llm=llm)

chat_histroy = []

while True:
    user_input = input("You : ")
    chat_histroy.append(user_input)
    if user_input == "exit":
        break
    result = model.invoke(chat_histroy)
    chat_histroy.append(result.content)
    print("AI : ", result.content)

print(chat_histroy)