# Company LLM Service (FastAPI + Cerebras)

## Local Run (without Docker)
1) Create venv
2) pip install -e .
3) copy .env.example to .env and set values
4) uvicorn app.main:app --reload --port 8000

## Docker
docker compose up --build

## Endpoints
GET  /v1/health
POST /v1/chat

## Auth
Send header:
X-Service-Token: <LLM_SERVICE_TOKEN>
