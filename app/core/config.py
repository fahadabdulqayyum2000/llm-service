from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra env vars
    )

    APP_NAME: str = "Company LLM Service"
    APP_VERSION: str = "1.0.0"
    ENABLE_DOCS: bool = True

    # Security: token between MERN backend -> FastAPI
    LLM_SERVICE_TOKEN: str = Field(default="", description="Internal service token")

    # Cerebras OpenAI-compatible
    CEREBRAS_API_KEY: str = ""
    CEREBRAS_BASE_URL: str = "https://api.cerebras.ai/v1"
    CEREBRAS_MODEL: str = "llama3.1-8b"

    # Generation defaults
    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 700

    # Networking
    HTTP_TIMEOUT_SECS: float = 60.0
    HTTP_MAX_RETRIES: int = 2

    # CORS
    CORS_ALLOW_ORIGINS: List[str] = ["http://localhost:3000"]

    # Observability
    LOG_LEVEL: str = "INFO"
    
    # RAG Configuration
    RAG_ENABLED: bool = False
    RAG_DOCS_DIR: str = "data/docs"
    RAG_PERSIST_DIR: str = "data/chroma"
    RAG_TOP_K: int = 4
    
    # Embeddings API Keys (loaded from .env, NOT hardcoded!)
    COHERE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None


settings = Settings()

# from pydantic_settings import BaseSettings, SettingsConfigDict
# from pydantic import Field
# from typing import List


# class Settings(BaseSettings):
#     model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

#     APP_NAME: str = "Company LLM Service"
#     APP_VERSION: str = "1.0.0"
#     ENABLE_DOCS: bool = True

#     # Security: token between MERN backend -> FastAPI
#     LLM_SERVICE_TOKEN: str = Field(default="", description="Internal service token")

#     # Cerebras OpenAI-compatible
#     CEREBRAS_API_KEY: str = ""
#     CEREBRAS_BASE_URL: str = "https://api.cerebras.ai/v1"
#     CEREBRAS_MODEL: str = "llama3.1-8b"

#     # Generation defaults
#     TEMPERATURE: float = 0.2
#     MAX_TOKENS: int = 700

#     # Networking
#     HTTP_TIMEOUT_SECS: float = 60.0
#     HTTP_MAX_RETRIES: int = 2

#     # CORS
#     CORS_ALLOW_ORIGINS: List[str] = ["http://localhost:3000"]

#     # Observability
#     LOG_LEVEL: str = "INFO"
#       # RAG
#     RAG_ENABLED: bool = False
#     RAG_DOCS_DIR: str = "data/docs"
#     RAG_PERSIST_DIR: str = "data/chroma"
#     RAG_TOP_K: int = 4
#     COHERE_API_KEY: str = "vCL2ntD0Qf95VjqtmdXFBJ8jrF4AmFYmPnV7ce8l"

# settings = Settings()
