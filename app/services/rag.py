
# app/services/rag.py
from __future__ import annotations

from typing import List, Tuple, Dict

from app.services.vectorstore import get_vectorstore, DEFAULT_COLLECTION


def _format_context(docs) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        parts.append(f"[{i}] source={src}\n{d.page_content}")
    return "\n\n".join(parts)


async def rag_answer(
    question: str,
    cerebras_chat_fn,
    top_k: int = 4,
    persist_dir: str = "data/chroma",
    collection: str = DEFAULT_COLLECTION,
) -> Tuple[str, List[Dict[str, str]]]:
    vs = get_vectorstore(persist_dir=persist_dir, collection=collection)
    retriever = vs.as_retriever(search_kwargs={"k": top_k})

    # LangChain new API: use ainvoke/invoke instead of get_relevant_documents
    try:
        docs = await retriever.ainvoke(question)
    except AttributeError:
        docs = retriever.invoke(question)

    sources = [{"source": d.metadata.get("source", "unknown")} for d in docs]

    if not docs:
        return "I don't know based on the provided context.", sources

    context = _format_context(docs)

    system_prompt = (
        "You are a retrieval QA assistant.\n"
        "Use ONLY the provided context.\n"
        "If the context does not contain the answer, reply exactly: "
        "\"I don't know based on the provided context.\"\n"
        "Do NOT add any extra details, steps, timelines, conditions, or suggestions.\n"
        "Keep the answer to 1 sentence.\n"
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    reply = await cerebras_chat_fn(
        user_prompt,
        system_prompt=system_prompt,
        temperature=0.0,
        max_tokens=200,
    )

    return reply.strip(), sources


# # app/services/rag.py
# from __future__ import annotations

# from typing import List, Tuple, Dict

# from app.services.vectorstore import get_vectorstore, DEFAULT_COLLECTION


# def _format_context(docs) -> str:
#     parts = []
#     for i, d in enumerate(docs, start=1):
#         src = d.metadata.get("source", "unknown")
#         parts.append(f"[{i}] source={src}\n{d.page_content}")
#     return "\n\n".join(parts)


# async def rag_answer(
#     question: str,
#     cerebras_chat_fn,
#     top_k: int = 4,
#     persist_dir: str = "data/chroma",
#     collection: str = DEFAULT_COLLECTION,  # <-- IMPORTANT: kb1 (3+ chars)
# ) -> Tuple[str, List[Dict[str, str]]]:
#     vs = get_vectorstore(persist_dir=persist_dir, collection=collection)
#     retriever = vs.as_retriever(search_kwargs={"k": top_k})
#     docs = retriever.get_relevant_documents(question)

#     sources = [{"source": d.metadata.get("source", "unknown")} for d in docs]

#     if not docs:
#         return "I don't know based on the provided context.", sources

#     context = _format_context(docs)

#     system_prompt = (
#         "You are a retrieval QA assistant.\n"
#         "Use ONLY the provided context.\n"
#         "If the context does not contain the answer, reply exactly: "
#         "\"I don't know based on the provided context.\"\n"
#         "Do NOT add any extra details, steps, timelines, conditions, or suggestions.\n"
#         "Keep the answer to 1 sentence.\n"
#     )

#     user_prompt = (
#         f"Context:\n{context}\n\n"
#         f"Question: {question}\n"
#         "Answer:"
#     )

#     reply = await cerebras_chat_fn(
#         user_prompt,
#         system_prompt=system_prompt,
#         temperature=0.0,
#         max_tokens=200,
#     )

#     return reply.strip(), sources




# # app/services/rag.py
# from __future__ import annotations

# from typing import List, Tuple

# from app.services.vectorstore import get_vectorstore


# def _format_context(docs) -> str:
#     parts = []
#     for i, d in enumerate(docs, start=1):
#         src = d.metadata.get("source", "unknown")
#         parts.append(f"[{i}] (source: {src})\n{d.page_content}")
#     return "\n\n".join(parts)


# async def rag_answer(
#     question: str,
#     cerebras_chat_fn,
#     top_k: int = 4,
#     persist_dir: str = "data/chroma",
#     collection: str = "kb",
# ) -> Tuple[str, List[dict]]:
#     vs = get_vectorstore(persist_dir=persist_dir, collection=collection)
#     retriever = vs.as_retriever(search_kwargs={"k": top_k})
#     docs = retriever.get_relevant_documents(question)

#     context = _format_context(docs)

#     prompt = (
#     "You are a retrieval QA assistant.\n"
#     "Rules:\n"
#     "1) Use ONLY the provided Context.\n"
#     "2) If the Context does not contain the answer, reply exactly: \"I don't know based on the provided context.\" \n"
#     "3) Do NOT add policies, timelines, conditions, or suggestions not explicitly stated in the Context.\n"
#     "4) Keep the answer short.\n\n"
#     f"Context:\n{context}\n\n"
#     f"Question: {question}\n"
#     "Answer:"
# )


#     # prompt = (
#     #     "You are a helpful assistant. Answer using ONLY the context below. "
#     #     "If the answer is not in the context, say you don't know.\n\n"
#     #     f"Context:\n{context}\n\n"
#     #     f"Question: {question}\n"
#     #     "Answer:"
#     # )

#     reply = await cerebras_chat_fn(
#     prompt,  # same prompt you already build
#     system_prompt=system_prompt,
#     temperature=0.0,
# )

#     # reply = await cerebras_chat_fn(prompt)

#     sources = []
#     for d in docs:
#         sources.append({
#             "source": d.metadata.get("source", "unknown"),
#         })

#     return reply, sources
