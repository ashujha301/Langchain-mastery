from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Config ---
COMPANY = "Acme CRM"
PRODUCT = "sales automation SaaS"

# --- Chain ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior support agent for {company}, a {product} platform.
Rules:
- Be concise (max 150 words)  
- Always end with one clear next step
- If you don't know something, say so honestly"""),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

# --- Run ---
def support_bot(question: str):
    print(f"\n{COMPANY} Support: ", end="")
    for chunk in chain.stream({
        "company": COMPANY,
        "product": PRODUCT,
        "question": question
    }):
        print(chunk, end="", flush=True)
    print("\n")

# --- Test it ---
support_bot("How do I integrate with Salesforce?")
support_bot("My dashboard is showing wrong revenue numbers")
support_bot("Can I have multiple users on one account?")