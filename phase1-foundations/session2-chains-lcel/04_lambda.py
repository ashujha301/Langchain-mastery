from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
parser = StrOutputParser()

# Custom Python function that runs inside the chain
def assess_urgency(inputs: dict) -> dict:
    """Add urgency flag based on keywords in document"""
    doc = inputs["document"].lower()
    urgent_keywords = ["urgent", "asap", "immediately", "critical", "overdue"]
    is_urgent = any(word in doc for word in urgent_keywords)
    
    return {
        "document": inputs["document"],
        "urgency_prefix": "🚨 URGENT DOCUMENT\n" if is_urgent else "📄 STANDARD DOCUMENT\n"
    }

def format_output(inputs: dict) -> str:
    """Format final output with urgency prefix"""
    return f"{inputs['urgency_prefix']}\n{inputs['analysis']}"

# Prompt
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze this business document. List key points in 3 bullets."),
    ("human", "{document}")
])

# Chain with custom logic injected via RunnableLambda
chain = (
    RunnableLambda(assess_urgency)
    | {
        "analysis": analysis_prompt | llm | parser,
        "urgency_prefix": lambda x: x["urgency_prefix"]
    }
    | RunnableLambda(format_output)
)

# Test with urgent document
urgent_doc = """
URGENT: Client contract renewal due TODAY.
Legal must sign off immediately or we lose $2M ARR.
CEO approval needed ASAP.
"""

normal_doc = """
Monthly newsletter draft for review.
Please provide feedback by end of next week.
Marketing team to finalize design.
"""

print(chain.invoke({"document": urgent_doc}))
print("\n" + "="*50 + "\n")
print(chain.invoke({"document": normal_doc}))