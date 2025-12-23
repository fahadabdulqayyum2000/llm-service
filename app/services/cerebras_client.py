import asyncio
import logging
import httpx
from typing import Optional, List, Dict, Union

from app.core.config import settings

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are a company assistant for internal employees. "
    "Be concise. If you do not know something, say you don't know. "
    "Do not invent company-specific facts."
)


class CerebrasClient:
    def __init__(self):
        if not settings.CEREBRAS_API_KEY:
            logger.warning("CEREBRAS_API_KEY is empty. Calls will fail until set.")
        self.base_url = settings.CEREBRAS_BASE_URL.rstrip("/")
        self.api_key = settings.CEREBRAS_API_KEY

    async def chat(
        self,
        user_message: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Chat with a single user message + optional system prompt override.
        Backward compatible with existing callers.
        """
        sys_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_message},
        ]
        return await self.chat_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def chat_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Chat with full messages array (system/user/assistant).
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "company-llm-service/1.0",
        }

        body = {
            "model": settings.CEREBRAS_MODEL,
            "temperature": settings.TEMPERATURE if temperature is None else temperature,
            "max_tokens": settings.MAX_TOKENS if max_tokens is None else max_tokens,
            "messages": messages,
        }

        timeout = httpx.Timeout(settings.HTTP_TIMEOUT_SECS)
        retries = max(0, int(settings.HTTP_MAX_RETRIES))

        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, headers=headers, json=body)
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_err = e
                wait = 0.4 * (attempt + 1)
                logger.warning(
                    "Network/timeout error calling Cerebras (attempt %s/%s): %s",
                    attempt + 1, retries + 1, repr(e)
                )
                await asyncio.sleep(wait)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                body_text = e.response.text
                logger.error("Cerebras HTTP error %s: %s", status, body_text[:2000])
                if 500 <= status <= 599 and attempt < retries:
                    await asyncio.sleep(0.4 * (attempt + 1))
                    continue
                raise
            except Exception as e:
                logger.exception("Unexpected error calling Cerebras: %s", repr(e))
                raise

        raise RuntimeError(f"Failed to call Cerebras after retries: {repr(last_err)}")


cerebras_client = CerebrasClient()


# import asyncio
# import logging
# import httpx

# from app.core.config import settings

# logger = logging.getLogger(__name__)


# class CerebrasClient:
#     def __init__(self):
#         if not settings.CEREBRAS_API_KEY:
#             logger.warning("CEREBRAS_API_KEY is empty. Calls will fail until set.")
#         self.base_url = settings.CEREBRAS_BASE_URL.rstrip("/")
#         self.api_key = settings.CEREBRAS_API_KEY

#     async def chat(self, user_message: str) -> str:
#         """
#         Calls Cerebras OpenAI-compatible chat completions endpoint.
#         """
#         url = f"{self.base_url}/chat/completions"
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#             # Helps avoid some CDN/edge blocks in certain environments
#             "User-Agent": "company-llm-service/1.0",
#         }

#         body = {
#             "model": settings.CEREBRAS_MODEL,
#             "temperature": settings.TEMPERATURE,
#             "max_tokens": settings.MAX_TOKENS,
#             "messages": [
#                 {
#                     "role": "system",
#                     "content": (
#                         "You are a company assistant for internal employees. "
#                         "Be concise. If you do not know something, say you don't know. "
#                         "Do not invent company-specific facts."
#                     ),
#                 },
#                 {"role": "user", "content": user_message},
#             ],
#         }

#         timeout = httpx.Timeout(settings.HTTP_TIMEOUT_SECS)
#         retries = max(0, int(settings.HTTP_MAX_RETRIES))

#         last_err: Exception | None = None
#         for attempt in range(retries + 1):
#             try:
#                 async with httpx.AsyncClient(timeout=timeout) as client:
#                     resp = await client.post(url, headers=headers, json=body)
#                     resp.raise_for_status()
#                     data = resp.json()
#                     return data["choices"][0]["message"]["content"]
#             except (httpx.TimeoutException, httpx.NetworkError) as e:
#                 last_err = e
#                 wait = 0.4 * (attempt + 1)
#                 logger.warning("Network/timeout error calling Cerebras (attempt %s/%s): %s",
#                                attempt + 1, retries + 1, repr(e))
#                 await asyncio.sleep(wait)
#             except httpx.HTTPStatusError as e:
#                 # No retry by default for 4xx; retry for some 5xx
#                 status = e.response.status_code
#                 body_text = e.response.text
#                 logger.error("Cerebras HTTP error %s: %s", status, body_text[:2000])
#                 if 500 <= status <= 599 and attempt < retries:
#                     await asyncio.sleep(0.4 * (attempt + 1))
#                     continue
#                 raise
#             except Exception as e:
#                 logger.exception("Unexpected error calling Cerebras: %s", repr(e))
#                 raise

#         raise RuntimeError(f"Failed to call Cerebras after retries: {repr(last_err)}")


# cerebras_client = CerebrasClient()
