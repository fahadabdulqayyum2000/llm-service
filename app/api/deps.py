from fastapi import Depends
from app.core.security import require_service_token


ServiceAuth = Depends(require_service_token)
