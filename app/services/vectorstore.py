# app/services/vectorstore.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import httpx
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

from app.core.config import settings


DEFAULT_PERSIST_DIR = "data/chroma"
DEFAULT_COLLECTION = "kb1"  # 3+ chars


class CerebrasEmbeddings(Embeddings):
    """
    Uses Cerebras OpenAI-compatible embeddings endpoint.
    Avoids HuggingFace downloads on deploy.
    """

    def __init__(self, model: Optional[str] = None):
        self.base_url = settings.CEREBRAS_BASE_URL.rstrip("/")
        self.api_key = settings.CEREBRAS_API_KEY
        # If you have a dedicated embedding model name, set it via env; fallback to chat model.
        self.model = model or getattr(settings, "CEREBRAS_EMBED_MODEL", "") or settings.CEREBRAS_MODEL

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "company-llm-service/1.0",
        }
        body = {"model": self.model, "input": inputs}

        timeout = httpx.Timeout(settings.HTTP_TIMEOUT_SECS)
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            # OpenAI-compatible shape: data["data"][i]["embedding"]
            return [item["embedding"] for item in data["data"]]


_embeddings: Optional[CerebrasEmbeddings] = None


def get_embeddings() -> CerebrasEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = CerebrasEmbeddings()
    return _embeddings


def get_vectorstore(persist_dir: str = DEFAULT_PERSIST_DIR, collection: str = DEFAULT_COLLECTION) -> Chroma:
    return Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=get_embeddings(),
    )


def load_documents_from_folder(folder: str) -> List:
    docs = []
    base = Path(folder)
    if not base.exists():
        return docs

    for path in base.rglob("*.txt"):
        docs.extend(TextLoader(str(path), encoding="utf-8").load())

    return docs


def build_or_update_index(
    docs_folder: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection: str = DEFAULT_COLLECTION,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> int:
    docs = load_documents_from_folder(docs_folder)
    if not docs:
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    vs = get_vectorstore(persist_dir=persist_dir, collection=collection)
    vs.add_documents(chunks)
    vs.persist()
    return len(chunks)



# # app/services/vectorstore.py
# from __future__ import annotations

# from pathlib import Path
# from typing import List

# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain_chroma import Chroma


# DEFAULT_PERSIST_DIR = "data/chroma"
# DEFAULT_COLLECTION = "kb1"  # must be 3+ chars

# _embeddings = None

# def get_embeddings():
#     global _embeddings
#     if _embeddings is None:
#         _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return _embeddings

# # def get_embeddings():
# #     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# def get_vectorstore(persist_dir: str = DEFAULT_PERSIST_DIR, collection: str = DEFAULT_COLLECTION) -> Chroma:
#     return Chroma(
#         collection_name=collection,
#         persist_directory=persist_dir,
#         embedding_function=get_embeddings(),
#     )


# def load_documents_from_folder(folder: str) -> List:
#     docs = []
#     base = Path(folder)
#     if not base.exists():
#         return docs

#     for path in base.rglob("*.txt"):
#         docs.extend(TextLoader(str(path), encoding="utf-8").load())

#     return docs


# def build_or_update_index(
#     docs_folder: str,
#     persist_dir: str = DEFAULT_PERSIST_DIR,
#     collection: str = DEFAULT_COLLECTION,
#     chunk_size: int = 800,
#     chunk_overlap: int = 120,
# ) -> int:
#     docs = load_documents_from_folder(docs_folder)
#     if not docs:
#         return 0

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#     )
#     chunks = splitter.split_documents(docs)

#     vs = get_vectorstore(persist_dir=persist_dir, collection=collection)
#     vs.add_documents(chunks)
#     vs.persist()
#     return len(chunks)



