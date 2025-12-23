import logging
from fastapi import APIRouter, HTTPException
import httpx

from app.api.deps import ServiceAuth
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.cerebras_client import cerebras_client
from app.core.config import settings
from app.services.rag import rag_answer

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, _auth=ServiceAuth):
    if not settings.CEREBRAS_API_KEY:
        raise HTTPException(status_code=500, detail="CEREBRAS_API_KEY is not set")

    try:
        if settings.RAG_ENABLED:
            reply, sources = await rag_answer(
        question=payload.message,
        cerebras_chat_fn=cerebras_client.chat,
        persist_dir=settings.RAG_PERSIST_DIR,
        top_k=settings.RAG_TOP_K,
    )
            
            return ChatResponse(
                reply=reply,
                model=settings.CEREBRAS_MODEL,
                sources=sources,
            )

        reply = await cerebras_client.chat(payload.message)
        return ChatResponse(
            reply=reply,
            model=settings.CEREBRAS_MODEL,
        )

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=e.response.text,
        )
    except Exception as e:
        logger.exception("Chat endpoint error: %s", repr(e))
        raise HTTPException(status_code=500, detail="Internal error")



# import logging
# from fastapi import APIRouter, HTTPException
# import httpx

# from app.api.deps import ServiceAuth
# from app.schemas.chat import ChatRequest, ChatResponse
# from app.services.cerebras_client import cerebras_client
# from app.core.config import settings
# from app.services.rag import rag_answer


# logger = logging.getLogger(__name__)
# router = APIRouter()

# @router.post("/chat", response_model=ChatResponse)
# async def chat(payload: ChatRequest, _auth=ServiceAuth):
#     if not settings.CEREBRAS_API_KEY:
#         raise HTTPException(status_code=500, detail="CEREBRAS_API_KEY is not set")

#     try:
#         if settings.RAG_ENABLED:
#             reply, _ = await rag_answer(
#                 question=payload.message,
#                 cerebras_chat_fn=cerebras_client.chat,
#                 persist_dir=settings.RAG_PERSIST_DIR,
#                 top_k=settings.RAG_TOP_K,
#             )
#             return ChatResponse(reply=reply, model=settings.CEREBRAS_MODEL,sources=sources)

#         reply = await cerebras_client.chat(payload.message)
#         return ChatResponse(reply=reply, model=settings.CEREBRAS_MODEL,)

#     except httpx.HTTPStatusError as e:
#         raise HTTPException(
#             status_code=e.response.status_code,
#             detail=e.response.text,
#         )
#     except Exception as e:
#         logger.exception("Chat endpoint error: %s", repr(e))
#         raise HTTPException(status_code=500, detail="Internal error")



# import logging
# from fastapi import APIRouter, HTTPException, Depends
# import httpx

# from app.api.deps import ServiceAuth
# from app.schemas.chat import ChatRequest, ChatResponse
# from app.services.cerebras_client import cerebras_client
# from app.core.config import settings

# logger = logging.getLogger(__name__)
# router = APIRouter()

# @router.post("/chat", response_model=ChatResponse)
# async def chat(payload: ChatRequest, _auth=ServiceAuth):
#     if not settings.CEREBRAS_API_KEY:
#         raise HTTPException(status_code=500, detail="CEREBRAS_API_KEY is not set")

#     try:
#         reply = await cerebras_client.chat(payload.message)
#         return ChatResponse(reply=reply, model=settings.CEREBRAS_MODEL)
#     except httpx.HTTPStatusError as e:
#         # Pass through upstream status
#         status = e.response.status_code
#         detail = e.response.text
#         raise HTTPException(status_code=status, detail=detail)
#     except Exception as e:
#         logger.exception("Chat endpoint error: %s", repr(e))
#         raise HTTPException(status_code=500, detail="Internal error")
