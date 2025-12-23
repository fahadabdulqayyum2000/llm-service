from app.core.config import settings
from app.services.vectorstore import build_or_update_index, DEFAULT_COLLECTION

n = build_or_update_index(
    docs_folder=settings.RAG_DOCS_DIR,
    persist_dir=settings.RAG_PERSIST_DIR,
    collection=DEFAULT_COLLECTION,
)
print("Indexed chunks:", n)

