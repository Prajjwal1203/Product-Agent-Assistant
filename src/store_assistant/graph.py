"""LangGraph flow: intake -> check_input -> classify -> retrieve -> generate -> guard_checks
-> (escalate? supervisor_review : respond). Escalation is decided by guard_checks, not the model."""
import logging
import re
import uuid
from datetime import date
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import guards
from .config import session_date
from .llm import generate_llm
from .retrieval import Retriever
from .schemas import ATTRIBUTES, AskRequest, Comparison, Conflict, Fact, Preferences, Response, Row

log = logging.getLogger("store_assistant")  # never log question text: it may hold personal data

ID_RE = re.compile(r"\b(?:HP|CH)-\d{2,3}W?\b")
COMPARE_RE = re.compile(r"\b(compare|versus|vs|difference)\b", re.I)
RECOMMEND_RE = re.compile(r"\b(should i (buy|get|choose|pick)|recommend\w*|suggest\w*|best|which (headphones?|chargers?))\b", re.I)
CORE_PREFS = {"budget_max_inr": "budget", "use_case": "use case"}


class State(TypedDict, total=False):
    request: dict
    session_date: str
    question: str
    preferences: dict
    requested_ids: list[str]
    valid_ids: list[str]
    unsupported: list[str]
    reasons: list[str]
    skip: bool
    reject: bool
    intent: str
    chunks: list[dict]
    conflicts: list[dict]
    unresolved: bool
    candidate_ids: list[str]
    draft: dict
    decision: dict
    response: dict


def build_app(retriever: Retriever | None = None, llm_fn=None, checkpointer=None):
    retriever = retriever or Retriever()
    llm_fn = llm_fn or generate_llm

    def on(state: State) -> date:
        return date.fromisoformat(state["session_date"])

    def core_missing(state: State) -> list[str]:
        prefs = Preferences(**state["preferences"])
        return [label for key, label in CORE_PREFS.items() if key not in prefs.stated()]

    # 1. Request ---------------------------------------------------------
    def intake(state: State):
        req = AskRequest.model_validate(state["request"])
        found = ID_RE.findall(req.question.upper())
        ids = list(dict.fromkeys([i.strip().upper() for i in req.product_ids] + found))
        return {
            "question": req.question,
            "preferences": req.preferences.model_dump(),
            "requested_ids": ids,
            "unsupported": [],
            "reasons": [],
            "skip": False,
            "reject": False,
        }

    # 2. Check input: known product IDs, no personal data --------------------
    def check_input(state: State):
        prefs = state["preferences"]
        if guards.has_personal_data(state["question"], prefs.get("use_case") or "", *prefs["must_have"]):
            return {
                "skip": True,
                "reject": True,
                "valid_ids": [],
                "unsupported": ["Request contains personal data. Remove it and ask again."],
            }
        valid, unsupported = [], []
        for pid in state["requested_ids"]:
            status = retriever.product_status(pid, on(state))
            if status == "ok":
                valid.append(pid)
            elif status == "not_approved":
                unsupported.append(f"{pid} is not in approved content, so nothing can be said about it.")
            else:
                unsupported.append(f"{pid} was not found in the catalog.")
        return {"valid_ids": valid, "unsupported": unsupported}

    # 3. Classify intent (and decide whether the model is needed at all) -------
    def classify(state: State):
        if state["skip"]:
            return {"intent": "answer"}
        q = state["question"]
        if len(state["requested_ids"]) >= 2 or COMPARE_RE.search(q):
            intent = "compare"
        elif RECOMMEND_RE.search(q):
            intent = "recommend"
        else:
            intent = "answer"
        unsupported = list(state["unsupported"])
        skip = False
        availability = bool(guards.AVAILABILITY_RE.search(q))
        if availability:
            unsupported.append("Stock availability is not in approved content; check the inventory system or ask a supervisor.")
        discount_blocked = False
        if guards.DISCOUNT_RE.search(q):
            if not retriever.promos(state["valid_ids"], on(state), expired=False):
                discount_blocked = True
                expired = retriever.promos(state["valid_ids"], on(state), expired=True)
                if expired:
                    unsupported.append(
                        f"Promotion {expired[0].citation} expired on {expired[0].effective_to}; no discount can be quoted."
                    )
                else:
                    unsupported.append("There is no approved promotion content; no discount can be quoted.")
        if state["requested_ids"] and not state["valid_ids"]:
            skip = True
        elif intent == "answer" and (availability or discount_blocked):
            skip = True
        elif intent == "compare" and not state["requested_ids"]:
            skip = True
            unsupported.append("Give at least one product ID to compare.")
        elif intent == "recommend" and not state["valid_ids"] and core_missing(state):
            skip = True
        return {"intent": intent, "unsupported": unsupported, "skip": skip}

    # 4. Retrieve ---------------------------------------------------------
    def retrieve(state: State):
        ids = list(state["valid_ids"])
        if not ids and state["intent"] == "recommend":
            category = "chargers" if "charger" in state["question"].lower() else "headphones"
            ids = retriever.products_in_category(category, on(state))
        chunks, conflicts, unresolved = [], [], False
        for pid in ids:
            winner, conf, unres = retriever.resolve(pid, on(state))
            if winner:
                chunks.append(winner.as_chunk())
                conflicts += conf
                unresolved = unresolved or unres
        chunks += [d.as_chunk() for d in retriever.policy_docs(state["question"], on(state))]
        return {"chunks": chunks, "conflicts": conflicts, "unresolved": unresolved, "candidate_ids": ids}

    # 5. Generate: code builds the table and attribute facts, the model adds facts/recommendation --
    def generate(state: State):
        intent = state["intent"]
        chunks = state.get("chunks", [])
        products = [c for c in chunks if c["doc_type"] == "catalog"]
        resp = Response(
            intent=intent,
            unsupported=list(state["unsupported"]),
            data_conflicts=[Conflict(**c) for c in state.get("conflicts", [])],
        )
        reasons = list(state["reasons"])
        missing = core_missing(state) if intent != "answer" and not state["reject"] else []
        resp.missing_preferences = list(missing)
        if not state["skip"]:
            categories = {c["category"] for c in products}
            if intent in ("compare", "recommend") and len(categories) == 1:
                category = categories.pop()
                attrs = ATTRIBUTES[category]
                resp.comparison = Comparison(
                    attributes=attrs,
                    rows=[
                        Row(
                            product_id=c["product_id"],
                            values=[str(c["attributes"][a]) for a in attrs],
                            citations=[c["citation"]],
                        )
                        for c in products
                    ],
                )
                resp.facts = [
                    Fact(product_id=c["product_id"], attribute=a, value=str(c["attributes"][a]), citation=c["citation"])
                    for c in products
                    for a in attrs
                ]
            elif intent in ("compare", "recommend") and len(categories) > 1:
                resp.unsupported.append("Products from different categories cannot be compared.")
            task = "facts" if intent == "answer" else ("recommendation" if products and not missing else None)
            if task:
                try:
                    out = llm_fn(task, state["question"], chunks, state["preferences"], state.get("candidate_ids", []))
                    if task == "facts":
                        resp.facts = out.facts
                        if not out.facts:
                            resp.unsupported.append("The approved content does not answer this question.")
                    else:
                        resp.recommendation = out.recommendation
                        if out.recommendation and products[0]["category"] == "headphones":
                            resp.missing_preferences.append("preferred weight or comfort")
                except Exception as exc:  # noqa: BLE001
                    log.warning("model call failed: %s: %s", type(exc).__name__, exc)
                    reasons.append("model output invalid or unavailable")
        return {"draft": resp.model_dump(), "reasons": reasons}

    # 6. Guard checks: deterministic, decide escalation ------------------------
    def guard_checks(state: State):
        resp = Response.model_validate(state["draft"])
        by_citation = {c["citation"]: c for c in state.get("chunks", [])}
        reasons = list(state["reasons"])
        reasons += guards.check_facts(resp.facts, by_citation)
        if resp.recommendation:
            candidates = {c["product_id"]: c for c in by_citation.values() if c["doc_type"] == "catalog"}
            reasons += guards.check_recommendation(
                resp.recommendation, Preferences(**state["preferences"]), candidates
            )
        reasons += guards.check_claims(resp.facts, resp.recommendation)
        category = next((c["category"] for c in by_citation.values() if c["doc_type"] == "catalog"), None)
        reasons += guards.check_comparison(resp.comparison, category)
        if state.get("unresolved"):
            reasons.append("conflicting versions cannot be resolved (same effective date)")
        if not state["reject"] and guards.is_policy_exception(state["question"]):
            reasons.append("policy exception: opened item, not defective, return requested")
        if reasons:
            resp.escalate = True
            resp.escalation_reason = "; ".join(dict.fromkeys(reasons))
        log.info("intent=%s ids=%s escalate=%s", resp.intent, state.get("valid_ids"), resp.escalate)
        return {"draft": resp.model_dump()}

    def route_after_classify(state: State):
        return "generate" if state["skip"] else "retrieve"

    def route_after_guards(state: State):
        return "supervisor_review" if state["draft"]["escalate"] else "respond"

    # 7. Supervisor review: pauses the graph until a human decides ------------
    def supervisor_review(state: State):
        decision = interrupt({"reason": state["draft"]["escalation_reason"], "draft": state["draft"]})
        return {"decision": decision}

    # 8. Response ----------------------------------------------------------
    def respond(state: State):
        resp = Response.model_validate(state["draft"])
        decision = state.get("decision")
        if decision is not None:
            note = f": {decision['note']}" if decision.get("note") else ""
            if decision.get("approved"):
                resp.escalation_reason += f" | approved by supervisor{note}"
            else:
                resp.facts, resp.comparison, resp.recommendation = [], None, None
                resp.unsupported.append("Supervisor rejected this answer. Do not quote it to the customer.")
                resp.escalation_reason += f" | rejected by supervisor{note}"
        return {"response": resp.model_dump()}

    g = StateGraph(State)
    for fn in (intake, check_input, classify, retrieve, generate, guard_checks, supervisor_review, respond):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "intake")
    g.add_edge("intake", "check_input")
    g.add_edge("check_input", "classify")
    g.add_conditional_edges("classify", route_after_classify, ["generate", "retrieve"])
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "guard_checks")
    g.add_conditional_edges("guard_checks", route_after_guards, ["supervisor_review", "respond"])
    g.add_edge("supervisor_review", "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def _result(app, thread_id: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    snap = app.get_state(config)
    if snap.next:  # paused at supervisor_review
        return {
            "status": "pending_supervisor",
            "thread_id": thread_id,
            "reason": snap.values["draft"]["escalation_reason"],
        }
    return {"status": "done", "thread_id": thread_id, "response": snap.values["response"]}


def ask(app, request: dict, on: str | None = None) -> dict[str, Any]:
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    app.invoke({"request": request, "session_date": on or session_date().isoformat()}, config)
    return _result(app, thread_id)


def pending_draft(app, thread_id: str) -> dict:
    return app.get_state({"configurable": {"thread_id": thread_id}}).values["draft"]


def review(app, thread_id: str, approved: bool, note: str | None = None) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    app.invoke(Command(resume={"approved": approved, "note": note}), config)
    return _result(app, thread_id)
