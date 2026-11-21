
import os
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


class FinancialRAG:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"batch_size": 8}
        )

        self.reranker = None

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

        self.vectorstore = None
        self.bm25_retriever = None
        self.hybrid_retriever = None

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

        # Each FinancialRAG instance gets its own FAISS index.
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        # Do not save to a shared FAISS directory.

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

    def ask(self, query):

        if not query or not query.strip():
            raise ValueError("Please enter a question.")

        if self.hybrid_retriever is None:
            raise ValueError("Upload a PDF before asking questions.")

        documents = self.hybrid_retriever.invoke(query)

        if not documents:
            return "No relevant information found.", []

        if self.reranker is None:
            self.reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )

        pairs = [
            [query, doc.page_content]
            for doc in documents
        ]

        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        context = "\n\n".join(
            f"[{i}] Source: {doc.metadata.get('source', 'Unknown')}, "
            f"Page: {doc.metadata.get('page', 0) + 1}\n"
            f"{doc.page_content}"
            for i, (doc, score) in enumerate(ranked, 1)
        )

        answer = self.chain.invoke({
            "context": context,
            "question": query
        })

        return answer, ranked
