
import os
import threading

from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# --------------------------------------------------
# SHARED MODELS: ONE COPY PER BACKEND PROCESS
# --------------------------------------------------

_embeddings = None
_reranker = None

_embeddings_lock = threading.Lock()
_reranker_lock = threading.Lock()
_reranker_predict_lock = threading.Lock()


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                print("Loading shared embedding model...", flush=True)

                _embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={
                        "batch_size": 4,
                        "normalize_embeddings": True
                    }
                )

                print("Embedding model loaded.", flush=True)

    return _embeddings


def get_reranker():
    global _reranker

    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                print("Loading shared CrossEncoder...", flush=True)

                _reranker = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    device="cpu"
                )

                print("CrossEncoder loaded.", flush=True)

    return _reranker


class FinancialRAG:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        # Shared embedding model
        self.embeddings = get_embeddings()

        # User-specific retrieval components
        self.vectorstore = None
        self.bm25_retriever = None
        self.hybrid_retriever = None

        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=api_key
        )

        self.prompt = ChatPromptTemplate.from_template("""
You are a financial research assistant.

Answer using only the provided context.
Never invent financial figures.
If the answer is unavailable, say so.
Cite sources using the provided page numbers.

Context:
{context}

Question:
{question}

Answer:
""")

        self.chain = self.prompt | self.llm | StrOutputParser()

    # --------------------------------------------------
    # PDF INGESTION
    # --------------------------------------------------

    def ingest_pdfs(self, pdf_paths):

        documents = []

        for path in pdf_paths:

            if not os.path.isfile(path):
                raise FileNotFoundError(f"PDF not found: {path}")

            loader = PyMuPDFLoader(path)
            documents.extend(loader.load())

        if not documents:
            raise ValueError("No PDF pages loaded.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No text chunks created.")

        print("Total chunks:", len(chunks), flush=True)

        # User-specific FAISS index
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        # User-specific BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = 5

        vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[
                self.bm25_retriever,
                vector_retriever
            ],
            weights=[0.5, 0.5]
        )

        print("Hybrid retrieval initialized.", flush=True)

        return len(chunks)

    # --------------------------------------------------
    # RETRIEVAL + RERANKING + GENERATION
    # --------------------------------------------------

    def ask(self, query):

        if not query or not query.strip():
            raise ValueError("Please enter a question.")

        if self.hybrid_retriever is None:
            raise ValueError("Upload a PDF before asking questions.")

        # Hybrid retrieval
        documents = self.hybrid_retriever.invoke(query)

        if not documents:
            return "No relevant information found.", []

        # Shared CrossEncoder
        reranker = get_reranker()

        pairs = [
            [query, doc.page_content]
            for doc in documents
        ]

        # Protect shared model during prediction
        with _reranker_predict_lock:
            scores = reranker.predict(
                pairs,
                batch_size=4,
                show_progress_bar=False
            )

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Prepare context with source pages
        context = "\n\n".join(
            f"[{i}] Source: {doc.metadata.get('source', 'Unknown')}, "
            f"Page: {doc.metadata.get('page', 0) + 1}\n"
            f"{doc.page_content}"
            for i, (doc, score) in enumerate(ranked, 1)
        )

        # LLM generation
        answer = self.chain.invoke({
            "context": context,
            "question": query
        })

        return answer, ranked
