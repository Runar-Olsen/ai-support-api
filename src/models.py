# src/models.py
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., description="User question in natural language")
    top_k: int = Field(3, description="Number of documents to retrieve")
    include_context: bool = Field(
        True, description="Whether to include retrieved docs in response"
    )


class RetrievedDocModel(BaseModel):
    id: str
    section: str
    title: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    docs: List[RetrievedDocModel]
    used_rag: bool
    model: Optional[str] = None
