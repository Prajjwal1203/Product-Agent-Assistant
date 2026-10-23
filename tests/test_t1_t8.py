import json

from store_assistant.graph import review
from store_assistant.schemas import ATTRIBUTES

TRAVEL = {"budget_max_inr": 10000, "must_have": ["noise cancellation"], "use_case": "travel"}


def done(result):
    assert result["status"] == "done", result
    return result["response"]


def test_t1_happy_path_compare_and_recommend(run):
    r = done(run("Which of these is better for long flights?", ["HP-100", "HP-200"], TRAVEL))
    assert r["intent"] == "compare"
    assert r["comparison"]["attributes"] == ATTRIBUTES["headphones"]
    assert [row["product_id"] for row in r["comparison"]["rows"]] == ["HP-100", "HP-200"]
    hp200 = r["comparison"]["rows"][1]
    assert hp200["values"][1] == "28 hours" and hp200["citations"] == ["CAT-HP-200@v2"]
    assert r["recommendation"]["product_id"] == "HP-200"
    assert "must_have" in r["recommendation"]["based_on_preferences"]
    assert "preferred weight or comfort" in r["missing_preferences"]
    assert r["escalate"] is False and r["escalation_reason"] is None


def test_t2_missing_product(run):
    r = done(run("Compare HP-200 and HP-999", ["HP-200", "HP-999"]))
    assert any("HP-999" in u for u in r["unsupported"])
    assert all(f["product_id"] != "HP-999" for f in r["facts"])
    assert [row["product_id"] for row in r["comparison"]["rows"]] == ["HP-200"]
    assert any(f["product_id"] == "HP-200" for f in r["facts"])


def test_t3_unapproved_product_never_cited(run, llm):
    r = done(run("Tell me about HP-400", ["HP-400"]))
    assert r["facts"] == [] and r["comparison"] is None
    assert any("HP-400" in u and "approved" in u for u in r["unsupported"])
    assert llm.calls == 0
    assert "CAT-HP-400" not in json.dumps(r)


def test_t4_conflicting_data(run):
    r = done(run("Battery life of HP-200?", ["HP-200"]))
    assert [(f["value"], f["citation"]) for f in r["facts"]] == [("28 hours", "CAT-HP-200@v2")]
    conflict = r["data_conflicts"][0]
    assert conflict["attribute"] == "battery_life"
    assert "CAT-HP-200@v1" in conflict["sources"] and "CAT-HP-200@v2" in conflict["sources"]
    assert r["escalate"] is False


def test_t5_stock_is_unsupported(run):
    r = done(run("Is HP-200 in stock in Pune?", ["HP-200"]))
    assert any("availability" in u.lower() for u in r["unsupported"])
    assert r["facts"] == [] and r["recommendation"] is None


def test_t6_expired_promotion(run):
    r = done(run("Is there a discount on HP-200?", ["HP-200"]))
    assert any("expired" in u for u in r["unsupported"])
    assert "10%" not in json.dumps(r)


def test_t6_promo_is_live_inside_its_window(run):
    r = run("Is there a discount on HP-200?", ["HP-200"], on="2026-10-05")
    assert not any("expired" in u for u in done(r)["unsupported"])


def test_t7_policy_exception_pauses_graph(app, run):
    result = run("Customer opened HP-100 and wants a refund, not defective", ["HP-100"])
    assert result["status"] == "pending_supervisor"
    assert "policy exception" in result["reason"]
    config = {"configurable": {"thread_id": result["thread_id"]}}
    assert app.get_state(config).next == ("supervisor_review",)
    approved = done(review(app, result["thread_id"], True, "ok, goodwill"))
    assert approved["escalate"] is True and "approved by supervisor" in approved["escalation_reason"]


def test_t7_supervisor_rejection_clears_answer(app, run):
    result = run("Customer opened HP-100 and wants a refund, not defective", ["HP-100"])
    r = done(review(app, result["thread_id"], False))
    assert r["facts"] == [] and "rejected by supervisor" in r["escalation_reason"]


def test_t8_missing_preferences(run, llm):
    r = done(run("Which headphones should I buy?"))
    assert r["recommendation"] is None
    assert {"budget", "use case"} <= set(r["missing_preferences"])
    assert llm.calls == 0


def test_personal_data_rejected_before_model(run, llm):
    r = done(run("Customer ravi@example.com wants to know HP-100 battery", ["HP-100"]))
    assert any("personal data" in u for u in r["unsupported"]) and llm.calls == 0


def test_bad_citation_escalates(llm):
    from conftest import BagOfWords
    from store_assistant.graph import ask, build_app
    from store_assistant.retrieval import Retriever
    from store_assistant.schemas import Fact, LLMOut

    def liar(task, *a, **k):  # cites a source that was not retrieved, with an invented value
        return LLMOut(facts=[Fact(product_id="HP-200", attribute="battery_life", value="99 hours", citation="CAT-HP-200@v1")])

    app = build_app(retriever=Retriever(embeddings=BagOfWords()), llm_fn=liar)
    result = ask(app, {"question": "Battery life of HP-200?", "product_ids": ["HP-200"]}, on="2026-10-12")
    assert result["status"] == "pending_supervisor" and "not retrieved" in result["reason"]
