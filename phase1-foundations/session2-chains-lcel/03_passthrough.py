from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
parser = StrOutputParser()

# Step 1 — Summarize
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize this document in 60 words max."),
    ("human", "{document}")
])

# Step 2 — Generate executive brief using BOTH original doc AND summary
brief_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a C-suite executive assistant.
    Write a 3-sentence executive brief using the summary provided.
    Original document length context: {original_length} characters."""),
    ("human", "Summary: {summary}")
])

# RunnablePassthrough carries "document" forward alongside the summary
chain = (
    RunnableParallel({
        "summary": summarize_prompt | llm | parser,
        "original_length": lambda x: str(len(x["document"])),
        "document": RunnablePassthrough() | (lambda x: x["document"])
    })
    | brief_prompt
    | llm
    | parser
)

document = """
Q3 Board Meeting Notes - Acme Corp
Revenue is down 12% vs Q2. The sales team must review pricing strategy by Oct 15.
Engineering needs to ship the new API integration before the enterprise client demo on Nov 1.
HR to complete all performance reviews by end of October.
CFO to present revised budget projections at next board meeting.
Customer churn has increased to 8% - customer success team to investigate root causes.
"""

result = chain.invoke({"document": document})
print("=== EXECUTIVE BRIEF ===")
print(result)