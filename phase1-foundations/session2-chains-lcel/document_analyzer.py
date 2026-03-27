from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
parser = StrOutputParser()

# --- Individual Analysis Chains ---
summary_chain = ChatPromptTemplate.from_messages([
    ("system", "Summarize this business document in max 80 words."),
    ("human", "{document}")
]) | llm | parser

action_chain = ChatPromptTemplate.from_messages([
    ("system", "Extract action items as a numbered list. Include owner and deadline if mentioned."),
    ("human", "{document}")
]) | llm | parser

risk_chain = ChatPromptTemplate.from_messages([
    ("system", """Assess risk level.
Respond EXACTLY in this format:
Risk Level: [LOW/MEDIUM/HIGH]
Reason: [one sentence]
Key Concerns: [bullet points if HIGH/MEDIUM, else 'None']"""),
    ("human", "{document}")
]) | llm | parser

# --- Executive Brief Chain (uses summary as input) ---
brief_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a 2-sentence executive brief suitable for C-suite. Be direct and outcome-focused."),
    ("human", "{summary}")
])

# --- Urgency Detector ---
def detect_urgency(doc: str) -> str:
    keywords = ["urgent", "asap", "immediately", "critical", "overdue", "deadline", "today"]
    found = [w for w in keywords if w in doc.lower()]
    return f"Urgent keywords detected: {', '.join(found)}" if found else "No urgent keywords"

# --- Master Chain ---
analyze_chain = (
    RunnableParallel({
        "summary": summary_chain,
        "action_items": action_chain,
        "risk": risk_chain,
        "urgency": lambda x: detect_urgency(x["document"]),
        "doc_length": lambda x: f"{len(x['document'].split())} words"
    })
)

def run_analyzer(document: str):
    print("\n" + "="*60)
    print("DOCUMENT ANALYSIS REPORT")
    print("="*60)
    
    results = analyze_chain.invoke({"document": document})
    
    # Generate executive brief from summary
    brief = (brief_prompt | llm | parser).invoke({"summary": results["summary"]})
    
    print(f"\n📌 Document Size: {results['doc_length']}")
    print(f"Urgency Check: {results['urgency']}")
    print(f"\ SUMMARY\n{results['summary']}")
    print(f"\ ACTION ITEMS\n{results['action_items']}")
    print(f"\n RISK ASSESSMENT\n{results['risk']}")
    print(f"\ EXECUTIVE BRIEF\n{brief}")
    print("\n" + "="*60)

# --- Test Documents ---
contract = """
Software License Agreement - TechCorp & Acme Inc
This agreement expires on December 31st 2024. 
Renewal requires 30-day notice. Failure to renew will result in immediate service termination.
Payment of $50,000 annual fee is overdue by 15 days - penalties apply.
Legal team must review indemnification clause added in Section 4.2 immediately.
CEO must countersign before end of week or contract is void.
"""

board_update = """
Q3 Product Update for Board Review
We shipped 3 major features this quarter ahead of schedule.
NPS score improved from 32 to 47.
Mobile app downloads up 34% month over month.
Engineering team requesting budget approval for infrastructure upgrade by November.
Next milestone: enterprise tier launch planned for Q1 next year.
"""

run_analyzer(contract)
run_analyzer(board_update)