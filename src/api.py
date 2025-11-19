# src/api.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import QueryRequest, QueryResponse
from .retriever import EmbeddingRetriever
from .rag import generate_rag_answer
from .utils import configure_logging

app = FastAPI(
    title="AI Support API",
    description="Embedding + RAG-based support assistant API",
    version="0.1.0",
)

# CORS (så du kan teste fra f.eks. en webklient senere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

configure_logging()
retriever = EmbeddingRetriever(top_k=5)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    # 1. hent relevante dokumenter
    docs = retriever.search(req.question, top_k=req.top_k)

    if not docs:
        raise HTTPException(status_code=404, detail="No documents found in knowledge base.")

    # 2. generer RAG-svar
    resp = generate_rag_answer(req.question, docs)
    return resp
