from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
parser = StrOutputParser()

# Three different analysis chains
summary_chain = ChatPromptTemplate.from_messages([
    ("system", "Summarize this business document in max 80 words."),
    ("human", "{document}")
]) | llm | parser

action_chain = ChatPromptTemplate.from_messages([
    ("system", "Extract action items as a numbered list from this document."),
    ("human", "{document}")
]) | llm | parser

risk_chain = ChatPromptTemplate.from_messages([
    ("system", """Assess the risk level of this document.
    Respond with:
    Risk Level: [LOW/MEDIUM/HIGH]
    Reason: [one sentence]"""),
    ("human", "{document}")
]) | llm | parser

# Run ALL THREE in parallel — ~same time as running one!
parallel_chain = RunnableParallel({
    "summary": summary_chain,
    "action_items": action_chain,
    "risk_assessment": risk_chain
})

document = """
Q3 Board Meeting Notes - Acme Corp
Revenue is down 12% vs Q2. The sales team must review pricing strategy by Oct 15.
Engineering needs to ship the new API integration before the enterprise client demo on Nov 1.
HR to complete all performance reviews by end of October.
CFO to present revised budget projections at next board meeting.
Customer churn has increased to 8% - customer success team to investigate root causes.
"""

import time
start = time.time()
results = parallel_chain.invoke({"document": document})
elapsed = time.time() - start

print(f"⚡ All 3 analyses completed in {elapsed:.2f}s\n")
print("=== SUMMARY ===")
print(results["summary"])
print("\n=== ACTION ITEMS ===")
print(results["action_items"])
print("\n=== RISK ASSESSMENT ===")
print(results["risk_assessment"])