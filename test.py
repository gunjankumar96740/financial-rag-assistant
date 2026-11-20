
from rag_pipeline import FinancialRAG
import traceback


def main():

    try:

        # Initialize RAG
        rag = FinancialRAG()

        # Ingest PDF
        print("\n========== PDF INGESTION ==========", flush=True)

        chunks = rag.ingest_pdfs([
            "data/FINAL Annual Report.pdf"
        ])

        print(f"\nTotal chunks: {chunks}", flush=True)

        # Ask question
        print("\n========== ASK QUESTION ==========", flush=True)

        while True:

            query = input("\nAsk a question (type exit to stop): ")

            if query.lower().strip() == "exit":
                print("Exiting Financial RAG...")
                break

            if not query.strip():
                print("Please enter a valid question.")
                continue

            # Generate answer
            answer, sources = rag.ask(query)

            print("\n========== ANSWER ==========")
            print(answer)

            # Display sources
            print("\n========== SOURCES ==========")

            for i, (doc, score) in enumerate(sources, 1):

                print(f"\nSource {i}")
                print("Relevance Score:", round(float(score), 4))
                print("Metadata:", doc.metadata)
                print("Content:", doc.page_content[:300])

    except Exception:

        print("\nAPPLICATION FAILED:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
