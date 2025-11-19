# src/rag.py
from __future__ import annotations
from typing import List

from openai import OpenAI

from .models import QueryResponse, RetrievedDocModel
from .utils import is_mock_mode, load_env, get_env_var
from .retriever import RetrievedDoc


def build_prompt(question, docs):
    context = "\n\n".join(
        f"[{d.section}] {d.title}\n{d.content}"
        for d in docs
    )
    return f"""
You are a helpful support AI assistant.
CONTEXT:
{context}

QUESTION:
{question}
"""


def generate_rag_answer(question: str, retrieved: List[RetrievedDoc]) -> QueryResponse:
    # MOCK MODE
    if is_mock_mode():
        fallback_doc = retrieved[0]
        mock_text = (
            f"[MOCK RAG RESPONSE]\n"
            f"Based on the knowledge base, the question is most similar to '{fallback_doc.title}'.\n"
            f"This is a simulated RAG-style response. In production, an LLM would generate a precise answer "
            f"using the retrieved context."
        )

        return QueryResponse(
            answer=mock_text,
            used_rag=False,
            model="mock-mode",
            docs=[
                RetrievedDocModel(
                    id=d.id, section=d.section, title=d.title, score=d.score
                ) for d in retrieved
            ],
        )

    # REAL MODE
    load_env()
    client = OpenAI(api_key=get_env_var("OPENAI_API_KEY"))
    prompt = build_prompt(question, retrieved)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful support assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content.strip()

    return QueryResponse(
        answer=answer,
        used_rag=True,
        model="gpt-4o-mini",
        docs=[
            RetrievedDocModel(
                id=d.id, section=d.section, title=d.title, score=d.score
            ) for d in retrieved
        ],
    )
