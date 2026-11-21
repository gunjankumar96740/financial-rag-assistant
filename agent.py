
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


class FinancialResponse(BaseModel):
    answer: str = Field(description="Final financial answer")
    financial_metrics: List[str] = Field(default_factory=list)
    calculations: Optional[str] = None
    source_pages: List[int] = Field(default_factory=list)
    limitations: Optional[str] = None


class UserAgent:

    def __init__(self):

        self.rag = FinancialRAG()
        self.checkpointer = InMemorySaver()

        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0
        )

        @tool
        def search_annual_report(question: str) -> str:
            """Search the uploaded annual report for financial information."""

            answer, ranked = self.rag.ask(question)

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

        @tool
        def compare_financial_values(
            previous_value: float,
            current_value: float,
            metric_name: str
        ) -> str:
            """Compare financial values between two periods."""

            if previous_value == 0:
                return "Cannot calculate percentage change from zero."

            change = current_value - previous_value

            percentage = (
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
                f"representing a {abs(percentage):.2f}% change."
            )

        self.agent = create_agent(
            model=self.llm,

            tools=[
                search_annual_report,
                calculate_profit_margin,
                compare_financial_values
            ],

            system_prompt="""
You are a Financial Research Assistant.

Rules:

- Retrieve financial figures before calculations.
- Never invent financial numbers.
- Verify financial years and units.
- Use the calculator for mathematical calculations.
- Use the comparison tool for growth calculations.
- Distinguish reported values from calculated values.
- Include source page numbers.
- Explain when information is unavailable.
- Use conversation history for follow-up questions.
""",

            checkpointer=self.checkpointer
        )

    def ingest(self, pdf_path):
        return self.rag.ingest_pdfs([pdf_path])

    @traceable(name="Financial Agent")
    def run(self, question, thread_id):

        return self.agent.invoke(
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
