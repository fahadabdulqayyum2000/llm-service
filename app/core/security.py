from fastapi import Header, HTTPException
from app.core.config import settings


def require_service_token(x_service_token: str = Header(default="")) -> bool:
    """
    Simple internal auth for service-to-service calls.
    Set LLM_SERVICE_TOKEN in env; MERN backend must send X-Service-Token header.
    """
    # If no token configured, allow (not recommended for production)
    if not settings.LLM_SERVICE_TOKEN:
        return True

    if x_service_token != settings.LLM_SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True
