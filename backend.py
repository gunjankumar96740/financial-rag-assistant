
import os
import uuid
import secrets
import threading

from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(
    title="Financial RAG API",
    description="AI-powered Financial Research Assistant",
    version="1.0.0"
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

API_KEY = os.getenv("APP_API_KEY")

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

DATA_DIR = Path("uploaded_reports")
DATA_DIR.mkdir(exist_ok=True)

UPLOAD_PATH = DATA_DIR / "active_report.pdf"

# Lazy-loaded agent module
agent_module = None

# Protect shared RAG index and agent initialization
rag_lock = threading.RLock()

# Tracks whether a PDF has been indexed
document_ready = False


# ==========================================
# 2. API KEY AUTHENTICATION
# ==========================================

def verify_api_key(x_api_key: str):

    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="APP_API_KEY is not configured"
        )

    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


# ==========================================
# 3. LAZY LOAD FINANCIAL AGENT
# ==========================================

def get_agent_module():

    global agent_module

    with rag_lock:

        if agent_module is None:
            import agent

            agent_module = agent

        return agent_module


# ==========================================
# 4. REQUEST MODELS
# ==========================================

class ChatRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=2000
    )

    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )


class ChatResponse(BaseModel):
    answer: str


# ==========================================
# 5. HOME / HEALTH CHECK
# ==========================================

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Financial RAG API is online"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "document_uploaded": document_ready
    }


# ==========================================
# 6. PDF UPLOAD ENDPOINT
# ==========================================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    x_api_key: str = Header(default="")
):

    global document_ready

    verify_api_key(x_api_key)

    # Validate filename
    if not file.filename or not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    try:

        # Read maximum allowed size + 1 byte
        content = await file.read(MAX_FILE_SIZE + 1)

        if len(content) > MAX_FILE_SIZE:

            raise HTTPException(
                status_code=413,
                detail="PDF must be 20 MB or smaller"
            )

        # Basic PDF signature validation
        if not content.startswith(b"%PDF-"):

            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file"
            )

        # Initialize agent only when needed
        module = get_agent_module()

        with rag_lock:

            # Save uploaded document
            UPLOAD_PATH.write_bytes(content)

            # Index uploaded PDF
            module.rag.ingest_pdfs([
                str(UPLOAD_PATH)
            ])

            document_ready = True

        return {
            "status": "success",
            "message": "PDF uploaded and indexed successfully",
            "filename": file.filename,
            "size_mb": round(len(content) / (1024 * 1024), 2)
        }

    except HTTPException:
        raise

    except Exception as e:

        document_ready = False

        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )

    finally:
        await file.close()


# ==========================================
# 7. CHAT ENDPOINT
# ==========================================

@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    x_api_key: str = Header(default="")
):

    verify_api_key(x_api_key)

    if not request.question.strip():

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
                request.question,
                request.thread_id
            )

        answer = result["messages"][-1].content

        return ChatResponse(
            answer=answer
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to process question: {str(e)}"
        )


# ==========================================
# 8. STARTUP MESSAGE
# ==========================================

@app.on_event("startup")
def startup():

    print("Financial RAG API started successfully")
    print("Agent will load when the first PDF is uploaded.")
