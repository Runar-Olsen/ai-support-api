# src/embeddings.py
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI

from .utils import configure_logging, get_data_path, load_env, get_env_var, is_mock_mode

logger = logging.getLogger(__name__)

EMB_MODEL_NAME = "text-embedding-3-small"
EMB_FILE = "kb_embeddings_openai.npz"
KB_INDEX_FILE = "kb_index.csv"


def load_knowledge_base() -> pd.DataFrame:
    data_path = get_data_path() / "knowledge_base.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find knowledge base at {data_path}")
    df = pd.read_csv(data_path)
    return df


def embed_texts_openai(client, texts):
    response = client.embeddings.create(
        model=EMB_MODEL_NAME,
        input=texts
    )
    vectors = [d.embedding for d in response.data]
    return np.array(vectors, dtype="float32")


def embed_texts_mock(texts):
    """Deterministic pseudo-embeddings for mock mode."""
    emb = []
    for t in texts:
        seed = sum(ord(c) for c in t) % 97
        rng = np.random.default_rng(seed)
        emb.append(rng.normal(0, 1, 256))  # fake 256-dim embedding
    return np.array(emb, dtype="float32")


def build_embeddings():
    configure_logging()
    load_env()

    kb = load_knowledge_base()
    texts = kb["content"].astype(str).tolist()

    logger.info("Embedding %d documents...", len(texts))

    if is_mock_mode():
        logger.warning("MOCK MODE ACTIVE: Creating fake embeddings without API calls.")
        emb = embed_texts_mock(texts)
    else:
        api_key = get_env_var("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        emb = embed_texts_openai(client, texts)

    data_path = get_data_path()
    np.savez_compressed(data_path / EMB_FILE, embeddings=emb)
    kb[["id", "section", "title"]].to_csv(data_path / KB_INDEX_FILE, index=False)

    logger.info("Saved embeddings + index to disk.")
    
if __name__ == "__main__":
    build_embeddings()