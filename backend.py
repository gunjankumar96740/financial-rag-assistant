
import os
import uuid
import secrets
import logging
import threading

from pathlib import Path
from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Header,
    HTTPException
)
from pydantic import BaseModel, Field

# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial RAG API",
    description="AI-powered Financial Research Assistant",
    version="1.0.0"
)

API_KEY = os.getenv("APP_API_KEY")

MAX_FILE_SIZE = 20 * 1024 * 1024

DATA_DIR = Path("uploaded_reports")
DATA_DIR.mkdir(exist_ok=True)

UPLOAD_PATH = DATA_DIR / "active_report.pdf"

# Agent is loaded only when required
agent_module = None

# Protect shared index and agent operations
rag_lock = threading.RLock()

document_ready = False


# ==========================================
# API KEY AUTHENTICATION
# ==========================================

def verify_api_key(x_api_key: str):

    if not API_KEY:
        logger.error("APP_API_KEY is not configured")

        raise HTTPException(
            status_code=500,
            detail="Server API key is not configured"
        )

    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


# ==========================================
# LAZY LOAD AGENT
# ==========================================

def get_agent_module():

    global agent_module

    with rag_lock:

        if agent_module is None:
            logger.info("Loading Financial RAG agent...")

            import agent

            agent_module = agent

            logger.info("Financial RAG agent loaded")

        return agent_module


# ==========================================
# REQUEST MODELS
# ==========================================

class ChatRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=2000
    )

    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        min_length=1,
        max_length=100
    )


class ChatResponse(BaseModel):

    answer: str


# ==========================================
# HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Financial RAG API is online"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "document_uploaded": document_ready,
        "agent_loaded": agent_module is not None
    }


# ==========================================
# PDF UPLOAD AND PROCESSING
# ==========================================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    x_api_key: str = Header(default="", alias="X-API-Key")
):

    global document_ready

    verify_api_key(x_api_key)

    if not file.filename or not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    try:

        content = await file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded PDF is empty"
            )

        if len(content) > MAX_FILE_SIZE:

            raise HTTPException(
                status_code=413,
                detail="PDF must be 20 MB or smaller"
            )

        if not content.startswith(b"%PDF-"):

            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file"
            )

        module = get_agent_module()

        with rag_lock:

            # Save uploaded PDF
            UPLOAD_PATH.write_bytes(content)

            logger.info("PDF saved: %s", file.filename)

            # Process and index PDF
            module.rag.ingest_pdfs([
                str(UPLOAD_PATH)
            ])

            document_ready = True

        logger.info("PDF indexed successfully")

        return {
            "status": "success",
            "message": "PDF uploaded and indexed successfully",
            "filename": file.filename,
            "size_mb": round(
                len(content) / (1024 * 1024), 2
            )
        }

    except HTTPException:
        raise

    except Exception:

        document_ready = False

        logger.exception("PDF processing failed")

        raise HTTPException(
            status_code=500,
            detail="PDF processing failed. Check backend logs."
        )

    finally:

        await file.close()


# ==========================================
# CHAT ENDPOINT
# ==========================================

@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    x_api_key: str = Header(default="", alias="X-API-Key")
):

    verify_api_key(x_api_key)

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    if not document_ready:

        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF before asking questions"
        )

    try:

        module = get_agent_module()

        with rag_lock:

            result = module.run_agent(
                question,
                request.thread_id
            )

        # Extract final agent response
        answer = result["messages"][-1].content

        if not answer:
            raise ValueError("Agent returned an empty response")

        return ChatResponse(
            answer=answer
        )

    except HTTPException:
        raise

    except Exception:

        logger.exception("Chat processing failed")

        raise HTTPException(
            status_code=500,
            detail="Unable to process question. Check backend logs."
        )


# ==========================================
# STARTUP
# ==========================================

@app.on_event("startup")
def startup():

    logger.info("Financial RAG API started successfully")
    logger.info("Agent will load when the first PDF is uploaded.")
