
import os
from dotenv import load_dotenv
from langsmith import Client
from rag_pipeline import FinancialRAG

load_dotenv(override=True)

client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

# Initialize existing RAG
rag = FinancialRAG()

rag.ingest_pdfs(["data/FINAL Annual Report.pdf"])


# 1. Target: Retrieval + reranking only
def target(inputs: dict):

    question = inputs["question"]

    # Hybrid retrieval
    documents = rag.hybrid_retriever.invoke(question)

    if not documents:
        return {"retrieved_pages": []}

    # CrossEncoder reranking
    if rag.reranker is None:
        from sentence_transformers import CrossEncoder
        rag.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    pairs = [
        [question, doc.page_content]
        for doc in documents
    ]

    scores = rag.reranker.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    # Extract unique page numbers
    retrieved_pages = []

    for doc, score in ranked:
        page = doc.metadata.get("page")

        if page is not None and page not in retrieved_pages:
            retrieved_pages.append(page)

    return {
        "retrieved_pages": retrieved_pages
    }


# 2. Precision@5
def precision_at_5(inputs, outputs):

    relevant = set(inputs["relevant_pages"])
    retrieved = outputs["retrieved_pages"][:5]

    score = len(set(retrieved) & relevant) / 5

    return {
        "key": "precision_at_5",
        "score": score
    }


# 3. Recall@5
def recall_at_5(inputs, outputs):

    relevant = set(inputs["relevant_pages"])
    retrieved = set(outputs["retrieved_pages"][:5])

    score = len(relevant & retrieved) / len(relevant)

    return {
        "key": "recall_at_5",
        "score": score
    }


# 4. Mean Reciprocal Rank
def mrr(inputs, outputs):

    relevant = set(inputs["relevant_pages"])
    retrieved = outputs["retrieved_pages"][:5]

    score = 0

    for rank, page in enumerate(retrieved, start=1):
        if page in relevant:
            score = 1 / rank
            break

    return {
        "key": "mrr",
        "score": score
    }


# 5. Run LangSmith experiment
results = client.evaluate(
    target,
    data="Financial-RAG-Annual-Report",
    evaluators=[
        precision_at_5,
        recall_at_5,
        mrr
    ],
    experiment_prefix="financial-rag-retrieval",
    description="Hybrid retrieval and CrossEncoder evaluation",
    max_concurrency=1
)

print("\nRetrieval evaluation completed!")
print(results)
