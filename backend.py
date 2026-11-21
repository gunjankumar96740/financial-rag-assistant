
import os
import uuid
import secrets
import logging
import threading
import time

from pathlib import Path
from collections import OrderedDict
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from pydantic import BaseModel, Field

from agent import UserAgent

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial RAG API",
    description="AI-powered Financial Research Assistant",
    version="2.0.0"
)

API_KEY = os.getenv("APP_API_KEY")

MAX_FILE_SIZE = 20 * 1024 * 1024
SESSION_TTL = 3600
MAX_SESSIONS = 30

DATA_DIR = Path("uploaded_reports")
DATA_DIR.mkdir(exist_ok=True)

sessions = OrderedDict()
sessions_lock = threading.RLock()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=1, max_length=100)


class ChatResponse(BaseModel):
    answer: str


def verify_api_key(x_api_key: str):

    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server API key is not configured"
        )

    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


def cleanup_sessions():

    now = time.time()

    expired = [
        sid for sid, data in sessions.items()
        if now - data["last_used"] > SESSION_TTL
    ]

    for sid in expired:
        data = sessions.pop(sid)
        data["pdf_path"].unlink(missing_ok=True)

    while len(sessions) > MAX_SESSIONS:
        sid, data = sessions.popitem(last=False)
        data["pdf_path"].unlink(missing_ok=True)


def get_session(session_id):

    with sessions_lock:
        cleanup_sessions()

        data = sessions.get(session_id)

        if data is None:
            raise HTTPException(
                status_code=404,
                detail="Session expired or not found. Upload your PDF again."
            )

        data["last_used"] = time.time()
        sessions.move_to_end(session_id)

        return data


@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Financial RAG API is online"
    }


@app.get("/health")
def health():

    with sessions_lock:
        cleanup_sessions()
        count = len(sessions)

    return {
        "status": "healthy",
        "active_sessions": count
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    x_api_key: str = Header(default="", alias="X-API-Key")
):

    verify_api_key(x_api_key)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    try:
        content = await file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(400, "Uploaded PDF is empty")

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(413, "PDF must be 20 MB or smaller")

        if not content.startswith(b"%PDF-"):
            raise HTTPException(400, "Invalid PDF file")

        session_id = str(uuid.uuid4())
        pdf_path = DATA_DIR / f"{session_id}.pdf"

        pdf_path.write_bytes(content)

        user_agent = UserAgent()
        user_agent.ingest(str(pdf_path))

        with sessions_lock:

            cleanup_sessions()

            if len(sessions) >= MAX_SESSIONS:
                pdf_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=503,
                    detail="Demo is currently busy. Please try again later."
                )

            sessions[session_id] = {
                "agent": user_agent,
                "pdf_path": pdf_path,
                "last_used": time.time(),
                "lock": threading.RLock()
            }

        logger.info("PDF indexed for session %s", session_id)

        return {
            "status": "success",
            "message": "PDF uploaded and indexed successfully",
            "session_id": session_id,
            "filename": file.filename,
            "size_mb": round(len(content) / (1024 * 1024), 2)
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("PDF processing failed")
        raise HTTPException(
            status_code=500,
            detail="PDF processing failed. Check backend logs."
        )

    finally:
        await file.close()


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    x_api_key: str = Header(default="", alias="X-API-Key")
):

    verify_api_key(x_api_key)

    question = request.question.strip()

    if not question:
        raise HTTPException(400, "Question cannot be empty")

    data = get_session(request.session_id)

    try:
        with data["lock"]:
            result = data["agent"].run(
                question,
                request.session_id
            )

        answer = result["messages"][-1].content

        if isinstance(answer, list):
            answer = "\n".join(
                item.get("text", str(item))
                if isinstance(item, dict) else str(item)
                for item in answer
            )

        if not answer:
            raise ValueError("Agent returned an empty response")

        return ChatResponse(answer=answer)

    except Exception:
        logger.exception("Chat processing failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to process question. Check backend logs."
        )
