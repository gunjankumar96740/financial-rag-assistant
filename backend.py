
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil
import uuid

from agent import run_agent, rag

app = FastAPI(title="Financial RAG API")


# ==========================================
# 1. CHAT REQUEST MODEL
# ==========================================

class ChatRequest(BaseModel):
    question: str
    thread_id: str = "default-user"


class ChatResponse(BaseModel):
    answer: str


# ==========================================
# 2. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Financial RAG API is running"
    }


# ==========================================
# 3. PDF UPLOAD ENDPOINT
# ==========================================

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    filename = f"{uuid.uuid4().hex}.pdf"
    file_path = data_dir / filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Index the uploaded PDF
        rag.ingest_pdfs([str(file_path)])

        return {
            "message": "PDF uploaded and indexed successfully",
            "filename": file.filename
        }

    except Exception:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="PDF processing failed"
        )

    finally:
        await file.close()


# ==========================================
# 4. CHAT ENDPOINT
# ==========================================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:
        result = run_agent(
            request.question,
            request.thread_id
        )

        answer = result["messages"][-1].content

        return ChatResponse(answer=answer)

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to process your question"
        )
