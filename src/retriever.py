# src/retriever.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from openai import OpenAI

from .utils import is_mock_mode, load_env, get_env_var, get_data_path
from .embeddings import EMB_FILE, EMB_MODEL_NAME, KB_INDEX_FILE


@dataclass
class RetrievedDoc:
    id: str
    section: str
    title: str
    content: str
    score: float


class EmbeddingRetriever:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        data_path = get_data_path()

        self.emb = np.load(data_path / EMB_FILE)["embeddings"]
        self.index = pd.read_csv(data_path / KB_INDEX_FILE)
        self.kb = pd.read_csv(data_path / "knowledge_base.csv")

        load_env()
        self.mock_mode = is_mock_mode()

        if not self.mock_mode:
            self.client = OpenAI(api_key=get_env_var("OPENAI_API_KEY"))

    def embed_query_mock(self, text):
        seed = sum(ord(c) for c in text) % 97
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, self.emb.shape[1])

    def embed_query_openai(self, text):
        resp = self.client.embeddings.create(
            model=EMB_MODEL_NAME,
            input=[text]
        )
        return np.array(resp.data[0].embedding, dtype="float32")

    def search(self, query: str, top_k: int | None = None) -> List[RetrievedDoc]:
        if top_k is None:
            top_k = self.top_k

        # Embedded query
        if self.mock_mode:
            q = self.embed_query_mock(query)
        else:
            q = self.embed_query_openai(query)

        # Normalize
        doc_norm = self.emb / np.linalg.norm(self.emb, axis=1, keepdims=True)
        q_norm = q / np.linalg.norm(q)

        sims = doc_norm @ q_norm
        top_idx = np.argsort(-sims)[:top_k]

        results = []
        for i in top_idx:
            meta = self.index.iloc[i]
            kb_row = self.kb[self.kb["id"] == meta["id"]].iloc[0]

            results.append(RetrievedDoc(
                id=str(kb_row["id"]),
                section=str(kb_row["section"]),
                title=str(kb_row["title"]),
                content=str(kb_row["content"]),
                score=float(sims[i])
            ))

        return results
