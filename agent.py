
from dotenv import load_dotenv
load_dotenv()

from typing import List, Optional

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import traceable

from rag_pipeline import FinancialRAG


# ==========================================
# 1. PYDANTIC RESPONSE MODEL
# ==========================================

class FinancialResponse(BaseModel):

    answer: str = Field(
        description="Final financial answer"
    )

    financial_metrics: List[str] = Field(
        description="Financial figures used in the analysis"
    )

    calculations: Optional[str] = Field(
        default=None,
        description="Calculation details"
    )

    source_pages: List[int] = Field(
        description="Annual report page numbers"
    )

    limitations: Optional[str] = Field(
        default=None,
        description="Missing information or limitations"
    )


# ==========================================
# 2. INITIALIZE RAG
# ==========================================

rag = FinancialRAG()
# ==========================================
# 3. TOOL 1: ANNUAL REPORT SEARCH
# ==========================================

@tool
def search_annual_report(question: str) -> str:
    """Search annual reports for financial information,
    company performance, risks, revenue and profits."""

    answer, ranked = rag.ask(question)

    sources = []

    for doc, score in ranked:

        page = doc.metadata.get("page", 0) + 1

        sources.append(
            f"Page {page}: {doc.page_content}"
        )

    return (
        f"Retrieved answer:\n{answer}\n\n"
        f"Sources:\n" + "\n\n".join(sources)
    )


# ==========================================
# 4. TOOL 2: PROFIT MARGIN CALCULATOR
# ==========================================

@tool
def calculate_profit_margin(
    net_profit: float,
    revenue: float
) -> str:
    """Calculate net profit margin as a percentage."""

    if revenue <= 0:
        return "Revenue must be greater than zero."

    margin = (net_profit / revenue) * 100

    return f"Net profit margin: {margin:.2f}%"


# ==========================================
# 5. TOOL 3: FINANCIAL COMPARISON
# ==========================================

@tool
def compare_financial_values(
    previous_value: float,
    current_value: float,
    metric_name: str
) -> str:
    """Compare financial values across two periods."""

    if previous_value == 0:
        return "Cannot calculate percentage change from zero."

    change = current_value - previous_value

    percentage_change = (
        change / abs(previous_value)
    ) * 100

    if change > 0:
        trend = "increased"
    elif change < 0:
        trend = "decreased"
    else:
        trend = "remained unchanged"

    return (
        f"{metric_name} {trend} by {abs(change):.2f}, "
        f"representing a {abs(percentage_change):.2f}% change."
    )


# ==========================================
# 6. INITIALIZE GROQ
# ==========================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# ==========================================
# 7. CREATE FINANCIAL AGENT
# ==========================================

agent = create_agent(
    model=llm,

    tools=[
        search_annual_report,
        calculate_profit_margin,
        compare_financial_values
    ],

    system_prompt="""
    You are a Financial Research Assistant.

    Available tools:

    1. search_annual_report:
       Retrieve information from annual reports.

    2. calculate_profit_margin:
       Calculate net profit margin.

    3. compare_financial_values:
       Compare financial values across two periods.

    Rules:

    - Retrieve financial figures before calculations.
    - Never invent financial numbers.
    - Verify financial years and units.
    - Use the calculator for mathematical calculations.
    - Use the comparison tool for growth calculations.
    - Distinguish reported values from calculated values.
    - Include source page numbers.
    - If information is unavailable, explain it.
    - Use conversation history for follow-up questions.
    """,

    checkpointer=InMemorySaver()
)


# ==========================================
# 8. STRUCTURED OUTPUT
# ==========================================

structured_llm = llm.with_structured_output(
    FinancialResponse
)


# ==========================================
# 9. AGENT EXECUTION WITH LANGSMITH
# ==========================================

@traceable(name="Financial Agent")
def run_agent(
    question: str,
    thread_id: str = "financial-user-1"
):

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },

        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return response


# ==========================================
# 10. STRUCTURED RESPONSE FUNCTION
# ==========================================

def get_financial_response(
    question: str,
    thread_id: str = "financial-user-1"
):

    response = run_agent(
        question,
        thread_id
    )

    agent_answer = response["messages"][-1].content

    final_answer = structured_llm.invoke(
        f"""
        Convert the following financial answer into
        the required FinancialResponse structure.

        Do not invent financial values or page numbers.
        Preserve only information present in the answer.

        Financial answer:
        {agent_answer}
        """
    )

    return final_answer


# ==========================================
# 11. MAIN LOOP
# ==========================================

if __name__ == "__main__":

    print("\nFinancial Research Agent Started!")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nYou: ")

        if question.lower() == "exit":
            break

        try:

            final_answer = get_financial_response(
                question,
                thread_id="terminal-user"
            )

            print("\nFINANCIAL AGENT:")

            print(
                final_answer.model_dump_json(indent=2)
            )

        except Exception as e:

            print(f"\nError: {e}")

        print("\n" + "-" * 60)
