"""
EchoCheck API.   uvicorn app:app --reload    ->  http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from gen4.api import common_router
from service import EchoCheckService

service = EchoCheckService()
app = FastAPI(title="EchoCheck", version="2.0.0",
              description="Continuous LLM evaluation: accuracy and consistency, routed grading, "
                          "regression gate against a promoted baseline, and human adjudication "
                          "that improves the answer key.")
app.include_router(common_router(service, "echocheck"))


class Case(BaseModel):
    id: str | None = None
    question: str
    expected_answer: str


class RunIn(BaseModel):
    suite: str = "sample"
    model_label: str
    provider: Literal["offline", "openai", "anthropic"] = "offline"
    runs: int = Field(3, ge=1, le=10)
    responses: dict[str, list[str]] | None = Field(
        None, description="Pre-recorded answers per question id (evaluate outputs produced elsewhere)")


class AdjudicateIn(BaseModel):
    suite: str
    question_id: str
    answer: str
    correct: bool


@app.get("/", tags=["ops"])
def root():
    return {"service": "EchoCheck", "docs": "/docs",
            "flow": "PUT /suites/{name} -> POST /runs -> POST /suites/{name}/baseline/{run_id} -> "
                    "POST /adjudicate -> POST /runs/{id}/regrade"}


@app.get("/suites", tags=["suites"])
def list_suites():
    return service.suites()


@app.put("/suites/{name}", tags=["suites"])
def put_suite(name: str, cases: list[Case]):
    return service.save_suite(name, [c.model_dump() for c in cases])


@app.get("/suites/{name}", tags=["suites"])
def get_suite(name: str):
    s = service.suite(name)
    if s is None:
        raise HTTPException(404, "unknown suite")
    return s


@app.post("/runs", tags=["runs"])
def run(body: RunIn):
    try:
        return service.run(body.suite, body.model_label, body.provider, body.runs, body.responses)
    except KeyError:
        raise HTTPException(404, f"unknown suite {body.suite}")


@app.get("/runs/{run_id}", tags=["runs"])
def get_run(run_id: str):
    d = service.memory.get_decision(run_id)
    if not d:
        raise HTTPException(404, "unknown run")
    return {"run_id": run_id, **d["output"]}


@app.post("/runs/{run_id}/regrade", tags=["feedback"])
def regrade(run_id: str):
    try:
        return service.regrade(run_id)
    except KeyError:
        raise HTTPException(404, "unknown run")


@app.post("/suites/{name}/baseline/{run_id}", tags=["runs"])
def promote(name: str, run_id: str):
    try:
        return service.promote(name, run_id)
    except KeyError:
        raise HTTPException(404, "run not found for this suite")


@app.post("/adjudicate", tags=["feedback"])
def adjudicate(body: AdjudicateIn):
    try:
        return service.adjudicate(body.suite, body.question_id, body.answer, body.correct)
    except KeyError:
        raise HTTPException(404, "unknown question")


@app.get("/suites/{name}/flaky", tags=["feedback"])
def flaky(name: str):
    return service.flaky(name)
