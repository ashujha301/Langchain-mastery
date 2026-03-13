from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,  # 0 = deterministic( AI bot replies ), 1 = creative
    max_tokens=500
)

#invoke() , single call - asynchronous
resp = llm.invoke("What are the top 3 reasons SaaS startup fails?")

print(type(resp))               #AI message
print(resp.content)             #the actual content
print(resp.usage_metadata)      #Token usage