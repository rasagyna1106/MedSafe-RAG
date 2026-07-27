"""MedSafe RAG — professional web application (FastAPI)."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent / ".env")

from web_service import (
    EXAMPLE_QUESTIONS,
    check_medication_pair,
    prepare_medication_check,
    process_question,
    resolve_medication_entry,
)

PROJECT_ROOT = Path(__file__).resolve().parent

app = FastAPI(title="MedSafe", version="1.0.0")
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")
templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")


class AskRequest(BaseModel):
    question: str


class MedicationsRequest(BaseModel):
    medications: str


class PairRequest(BaseModel):
    drug_a: str
    drug_b: str


class ResolveMedicationRequest(BaseModel):
    medication: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/examples")
async def examples():
    return EXAMPLE_QUESTIONS


@app.post("/api/ask")
async def ask(body: AskRequest):
    return process_question(body.question)


@app.post("/api/medications/resolve")
async def medications_resolve(body: ResolveMedicationRequest):
    return resolve_medication_entry(body.medication)


@app.post("/api/medications/prepare")
async def medications_prepare(body: MedicationsRequest):
    return prepare_medication_check(body.medications)


@app.post("/api/medications/check-pair")
async def medications_check_pair(body: PairRequest):
    return check_medication_pair(body.drug_a, body.drug_b)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
