import json

from .config import OLLAMA_MODEL
from .schemas import LLMOut

SYSTEM = """You are a product-knowledge assistant for store associates.
Use ONLY the DOCUMENTS provided. Never use outside knowledge.
Never mention stock, availability, discounts or promotions.
Copy values verbatim from the documents. Cite with the exact document id shown in the brackets, WITHOUT the brackets, e.g. CAT-HP-200@v2.
Reply with ONLY a JSON object matching this schema exactly (use these exact key names, no extras):
{schema}"""

TASKS = {
    "facts": (
        "Return `facts`: only the facts that directly answer the question. Each fact has product_id, "
        "attribute, value (copied from the document) and citation. Use product_id 'ALL' for policy documents "
        "that cover all products. If the documents do not answer the question return an empty list. "
        "If the question is about discounts, promotions, offers, stock or availability, return an empty list "
        "(never state a discount, percentage off or availability, even if a document mentions one). "
        "`recommendation` must be null."
    ),
    "recommendation": (
        "Return `recommendation`: choose ONE product from CANDIDATES that satisfies the hard constraints "
        "(budget_max_inr, must_have) and best fits use_case. `reasoning` is 1-2 sentences using only document "
        "facts. `based_on_preferences` lists which of the stated preference keys you used. "
        "If no candidate satisfies the constraints return null. `facts` must be an empty list."
    ),
}


def _documents(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['citation']}] product={c['product_id']} type={c['doc_type']}\n"
        f"attributes: {json.dumps(c['attributes'])}\n{c['text']}"
        for c in chunks
    )


def _clean(out: LLMOut) -> LLMOut:
    """Models sometimes echo the brackets around a citation id; the id itself is what must match."""
    for f in out.facts:
        f.citation = f.citation.strip().strip("[]").strip()
    return out


def generate_llm(task: str, question: str, chunks: list[dict], preferences: dict, candidate_ids: list[str]) -> LLMOut:
    from langchain_ollama import ChatOllama

    model = ChatOllama(model=OLLAMA_MODEL, temperature=0).with_structured_output(LLMOut, method="json_schema")
    human = (
        f"TASK: {TASKS[task]}\n\nQUESTION: {question}\n\n"
        f"PREFERENCES: {json.dumps(preferences)}\n\nCANDIDATES: {candidate_ids}\n\n"
        f"DOCUMENTS:\n{_documents(chunks)}"
    )
    last_error = None
    for _ in range(2):  # one retry on malformed structured output
        try:
            return _clean(model.invoke([("system", SYSTEM.replace("{schema}", json.dumps(LLMOut.model_json_schema()))), ("human", human)]))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"model output invalid: {last_error}")
