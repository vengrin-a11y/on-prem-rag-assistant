"""Public API with a deliberately small response contract."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from on_prem_rag.config import Settings
from on_prem_rag.pipeline import Pipeline

app = FastAPI(title="On-Prem RAG Assistant")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="question must not be blank")
    try:
        return Pipeline(Settings.from_env()).ask(request.question.strip())
    except Exception as exc:
        raise HTTPException(status_code=503, detail="RAG dependencies unavailable") from exc
