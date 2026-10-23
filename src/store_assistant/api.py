from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .graph import ask, build_app, pending_draft, review
from .render import render
from .schemas import AskRequest

api = FastAPI(title="Store Associate Product Knowledge Assistant")
_app = None


def graph():
    global _app
    if _app is None:
        _app = build_app()
    return _app


def _with_display(result: dict) -> dict:
    if result["status"] == "done":
        result["display"] = render(result["response"])
    return result


@api.post("/ask")
def post_ask(req: AskRequest):
    return _with_display(ask(graph(), req.model_dump()))


@api.get("/pending/{thread_id}")
def get_pending(thread_id: str):
    try:
        return pending_draft(graph(), thread_id)
    except KeyError:
        raise HTTPException(404, "unknown thread") from None


class Decision(BaseModel):
    approved: bool
    note: str | None = None


@api.post("/review/{thread_id}")
def post_review(thread_id: str, d: Decision):
    return _with_display(review(graph(), thread_id, d.approved, d.note))
