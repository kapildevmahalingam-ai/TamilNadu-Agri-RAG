"""FastAPI REST API for the agriculture RAG assistant."""
from __future__ import annotations
import json
import logging
import os
from typing import Any
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from rag import answer, answer_stream

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Tamil Nadu Agriculture Schemes RAG", version="1.0.0")

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    history: list[dict[str, Any]] = Field(default_factory=list)

@app.get("/")
def root():
    return {"message": "Tamil Nadu Agriculture Schemes Assistant API is running", "docs": "/docs"}

@app.get("/health")
def health():
    from pathlib import Path
    index_dir = Path(__file__).resolve().parent / "faiss_index"
    return {
        "status": "ok",
        "index_ready": (index_dir / "index.faiss").exists(),
        "nemo_guardrails": False,
        "guardrails_note": "Rule-based guardrails are active; NeMo integration is optional."
    }

@app.post("/chat")
def chat(body: ChatRequest):
    try:
        return answer(body.question, body.history)
    except Exception:
        logging.exception("Chat request failed")
        return {"answer": "Sorry, I couldn't process that request. Check backend logs.", "sources": []}

@app.post("/chat/stream")
def chat_stream(body: ChatRequest):
    def events():
        try:
            for item in answer_stream(body.question, body.history):
                yield json.dumps(item, ensure_ascii=False) + "\n"
        except Exception:
            logging.exception("Streaming request failed")
            yield json.dumps({"event": "error", "text": "Sorry, something went wrong. Check backend logs."}) + "\n"
    return StreamingResponse(events(), media_type="application/x-ndjson")