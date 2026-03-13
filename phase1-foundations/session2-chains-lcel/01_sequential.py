from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
parser = StrOutputParser()

# Chain 1 — Summarize the document
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert business analyst. Summarize the document in max 100 words."),
    ("human", "{document}")
])

# Chain 2 — Extract action items FROM the summary
action_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract all action items as a numbered list. If none exist, write 'No action items found.'"),
    ("human", "{summary}")
])

# Build individual chains
summarize_chain = summarize_prompt | llm | parser
action_chain = action_prompt | llm | parser

# Connect them — output of summarize becomes input of action_chain
full_chain = summarize_chain | (lambda summary: {"summary": summary}) | action_chain

# Test it
document = """
Q3 Board Meeting Notes - Acme Corp
Revenue is down 12% vs Q2. The sales team must review pricing strategy by Oct 15.
Engineering needs to ship the new API integration before the enterprise client demo on Nov 1.
HR to complete all performance reviews by end of October.
CFO to present revised budget projections at next board meeting.
Customer churn has increased to 8% - customer success team to investigate root causes.
"""

result = full_chain.invoke({"document": document})
print("=== Action Items ===")
print(result)