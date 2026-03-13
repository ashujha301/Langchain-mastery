from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Define each component separately
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior customer support agent for {company_name}.
    Be concise and always offer a next step."""),
    ("human", "{user_message}")
])

parser = StrOutputParser()  # Converts AIMessage → plain string

# THIS is LCEL — chain components with the pipe operator
# prompt → llm → parser
chain = prompt | llm | parser

# invoke the whole chain in one line
response = chain.invoke({
    "company_name": "Acme CRM",
    "user_message": "How do I set up automated follow-up emails?"
})

print(response)  # Plain string now, not AIMessage
print(type(response))  # <class 'str'>