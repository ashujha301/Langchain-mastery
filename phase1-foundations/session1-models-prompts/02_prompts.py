from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Define a reusable prompt template with variables
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior customer support agent for {company_name}, 
    a {product_type} SaaS platform. 
    Be concise, empathetic, and always offer a next step.
    Tone: {tone}"""),
    ("human", "{user_message}")
])

# See what the prompt looks like before sending to LLM
formatted = prompt.format_messages(
    company_name="Acme CRM",
    product_type="sales automation",
    tone="professional but friendly",
    user_message="I can't export my contacts to CSV, this is urgent!"
)

print("--- Formatted Prompt ---")
for msg in formatted:
    print(f"{msg.type}: {msg.content}\n")

# Now actually call the LLM
response = llm.invoke(formatted)
print("--- Response ---")
print(response.content)