from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Say: setup successfull!")
print(response.content)