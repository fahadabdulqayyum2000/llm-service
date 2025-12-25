
from __future__ import annotations

from pathlib import Path
from typing import List
import logging

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = "data/chroma"
DEFAULT_COLLECTION = "kb1"

_embeddings = None


def get_embeddings():
    """
    Get embeddings with multiple fallback options:
    1. Cohere (if COHERE_API_KEY is set) - Recommended for production
    2. OpenAI (if OPENAI_API_KEY is set) - Alternative
    3. HuggingFace (local fallback) - Development only
    """
    global _embeddings
    if _embeddings is None:
        
        # Try Cohere first (best for production)
        if settings.COHERE_API_KEY:
            logger.info("Initializing Cohere embeddings...")
            try:
                from langchain_cohere import CohereEmbeddings
                
                _embeddings = CohereEmbeddings(
                    model="embed-english-light-v3.0",
                    cohere_api_key=settings.COHERE_API_KEY
                )
                logger.info("✅ Cohere embeddings initialized successfully")
                return _embeddings
                
            except Exception as e:
                logger.error(f"❌ Cohere initialization failed: {e}")
        
        # Try OpenAI second
        if settings.OPENAI_API_KEY:
            logger.info("Initializing OpenAI embeddings...")
            try:
                from langchain_openai import OpenAIEmbeddings
                
                _embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=settings.OPENAI_API_KEY
                )
                logger.info("✅ OpenAI embeddings initialized successfully")
                return _embeddings
                
            except Exception as e:
                logger.error(f"❌ OpenAI initialization failed: {e}")
        
        # Fallback to HuggingFace (local only - will fail on Koyeb)
        logger.warning("⚠️  No API keys found. Using HuggingFace embeddings (local only)...")
        logger.warning("⚠️  This may fail on Koyeb. Set COHERE_API_KEY or OPENAI_API_KEY for production.")
        
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            
            _embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✅ HuggingFace embeddings initialized successfully")
            return _embeddings
            
        except Exception as e:
            logger.error(f"❌ All embedding methods failed: {e}")
            raise RuntimeError(
                "Failed to initialize embeddings. Please set one of:\n"
                "  - COHERE_API_KEY (recommended for production)\n"
                "  - OPENAI_API_KEY (alternative)\n"
                "  - Ensure HuggingFace models can download (local development only)"
            )
    
    return _embeddings


def get_vectorstore(
    persist_dir: str = DEFAULT_PERSIST_DIR, 
    collection: str = DEFAULT_COLLECTION
) -> Chroma:
    """Get or create a Chroma vectorstore."""
    return Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=get_embeddings(),
    )


def load_documents_from_folder(folder: str) -> List:
    """Load all .txt documents from a folder."""
    docs = []
    base = Path(folder)
    
    if not base.exists():
        logger.warning(f"📁 Folder {folder} does not exist")
        return docs

    txt_files = list(base.rglob("*.txt"))
    logger.info(f"📁 Found {len(txt_files)} .txt files in {folder}")

    for path in txt_files:
        try:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
            logger.info(f"✅ Loaded: {path.name}")
        except Exception as e:
            logger.error(f"❌ Failed to load {path.name}: {e}")

    logger.info(f"📄 Total documents loaded: {len(docs)}")
    return docs


def build_or_update_index(
    docs_folder: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection: str = DEFAULT_COLLECTION,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> int:
    """Build or update the vector index from documents."""
    logger.info(f"🔨 Building index from folder: {docs_folder}")
    
    docs = load_documents_from_folder(docs_folder)
    if not docs:
        logger.warning("⚠️  No documents found to index")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"✂️  Split into {len(chunks)} chunks")

    vs = get_vectorstore(persist_dir=persist_dir, collection=collection)
    vs.add_documents(chunks)
    vs.persist()
    
    logger.info(f"✅ Successfully indexed {len(chunks)} chunks")
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



