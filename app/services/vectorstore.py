# app/services/vectorstore.py
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma


DEFAULT_PERSIST_DIR = "data/chroma"
DEFAULT_COLLECTION = "kb1"  # must be 3+ chars

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings

# def get_embeddings():
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    vs = get_vectorstore(persist_dir=persist_dir, collection=collection)
    vs.add_documents(chunks)
    vs.persist()
    return len(chunks)



# # app/services/vectorstore.py
# from __future__ import annotations

# import os
# from pathlib import Path
# from typing import List

# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma


# DEFAULT_PERSIST_DIR = "data/chroma"
# DEFAULT_COLLECTION = "kb1"


# def get_embeddings():
#     # Good default; you can switch later
#     return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


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

#     # add more loaders later: md, pdf, docx etc.
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
