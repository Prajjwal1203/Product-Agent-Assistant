"""Tests run offline by default (fake model + fake embeddings). Set LIVE=1 to hit real Ollama."""
import math
import os
import re
import zlib

import pytest
from langchain_core.embeddings import Embeddings

from store_assistant.graph import ask, build_app
from store_assistant.retrieval import Retriever
from store_assistant.schemas import Fact, LLMOut, Recommendation

SESSION_DATE = "2026-10-12"  # fixed so the promo-expiry check is deterministic
LIVE = os.getenv("LIVE") == "1"


class BagOfWords(Embeddings):
    def _vec(self, text: str) -> list[float]:
        v = [0.0] * 256
        for w in re.findall(r"\w+", text.lower()):
            v[zlib.crc32(w.encode()) % 256] += 1
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, task, question, chunks, preferences, candidate_ids):
        self.calls += 1
        q = question.lower()
        if task == "facts":
            facts = []
            for c in chunks:
                if c["doc_type"] == "catalog" and "battery" in q:
                    facts.append(Fact(product_id=c["product_id"], attribute="battery_life",
                                      value=c["attributes"]["battery_life"], citation=c["citation"]))
                if c["doc_type"] == "policy" and ("refund" in q or "return" in q) and "15-day" in c["text"]:
                    facts.append(Fact(product_id="ALL", attribute="return_policy",
                                      value="15-day returns, unopened", citation=c["citation"]))
            return LLMOut(facts=facts)
        products = [c for c in chunks if c["doc_type"] == "catalog"]
        ok = []
        for c in products:
            if preferences.get("budget_max_inr") and int(c["attributes"]["price_inr"]) > preferences["budget_max_inr"]:
                continue
            if any("noise" in m for m in preferences["must_have"]) and c["attributes"]["noise_cancellation"] == "none":
                continue
            ok.append(c)
        if not ok:
            return LLMOut()
        best = ok[0]
        based = [k for k, present in (("must_have", preferences["must_have"]),
                                      ("budget_max_inr", preferences.get("budget_max_inr"))) if present]
        return LLMOut(recommendation=Recommendation(
            product_id=best["product_id"], reasoning=f"{best['product_id']} meets the stated needs.",
            based_on_preferences=based or ["use_case"]))


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def app(llm):
    if LIVE:
        return build_app()
    return build_app(retriever=Retriever(embeddings=BagOfWords()), llm_fn=llm)


@pytest.fixture
def run(app):
    def _run(question, ids=(), prefs=None, on=SESSION_DATE):
        req = {"question": question, "product_ids": list(ids), "preferences": prefs or {}}
        return ask(app, req, on=on)
    return _run
