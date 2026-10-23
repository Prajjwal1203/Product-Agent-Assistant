"""Optional Streamlit UI: uv sync --extra ui && uv run streamlit run src/store_assistant/ui.py"""
import html

import streamlit as st

from store_assistant.config import session_date
from store_assistant.graph import ask, build_app, pending_draft, review
from store_assistant.render import LABELS
from store_assistant.retrieval import load_docs

st.set_page_config(page_title="Store Product Assistant", layout="wide")

st.markdown(
    """
<style>
.card{border-radius:8px;padding:12px 16px;margin:6px 0 12px;border:1px solid;color:#1f2937}
.card b{display:block;font-size:1.05rem}
.rec{background:#e6f4d7;border-color:#7cb342}
.warn{background:#fff6d6;border-color:#e0b43a}
.na{background:#e9eaec;border-color:#b5b8be}
.bad{background:#fde2e1;border-color:#e57373}
.lbl{font-size:.85rem;font-weight:600;opacity:.75;margin-top:4px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px}
table.cmp{width:100%;border-collapse:collapse;margin-bottom:4px}
table.cmp th,table.cmp td{border:1px solid #c9ccd1;padding:6px 10px;text-align:left}
table.cmp th{background:#e9eaec;color:#1f2937}
table.cmp td{background:#fff;color:#1f2937}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_app():
    return build_app()


@st.cache_data
def approved_product_ids() -> list[str]:
    today = session_date()
    return sorted(
        {
            d.product_id
            for d in load_docs()
            if d.doc_type == "catalog"
            and d.approved
            and d.effective_from <= today
            and (d.effective_to is None or today <= d.effective_to)
        }
    )


def esc(x) -> str:
    return html.escape(str(x))


def card(kind: str, body: str):
    st.markdown(f'<div class="card {kind}">{body}</div>', unsafe_allow_html=True)


def show_response(r: dict):
    rec = r.get("recommendation")
    if rec:
        st.markdown('<div class="lbl">Recommendation</div>', unsafe_allow_html=True)
        card("rec", f"<b>{esc(rec['product_id'])}</b>{esc(rec['reasoning'])}")

    c = r.get("comparison")
    if c and c["rows"]:
        st.markdown('<div class="lbl">Comparison</div>', unsafe_allow_html=True)
        head = "".join(f"<th>{esc(row['product_id'])}</th>" for row in c["rows"])
        body = "".join(
            f"<tr><td>{esc(LABELS.get(attr, attr))}</td>"
            + "".join(f"<td>{esc(row['values'][i])}</td>" for row in c["rows"])
            + "</tr>"
            for i, attr in enumerate(c["attributes"])
        )
        st.markdown(
            f'<table class="cmp"><tr><th>Attribute</th>{head}</tr>{body}</table>', unsafe_allow_html=True
        )
        sources = sorted({cit for row in c["rows"] for cit in row["citations"]})
        st.caption("Sources: " + ", ".join(sources))
    elif r.get("facts"):
        st.markdown('<div class="lbl">Answer</div>', unsafe_allow_html=True)
        for f in r["facts"]:
            st.markdown(f"- **{f['product_id']}** {f['attribute']}: {f['value']}  `{f['citation']}`")

    if r.get("missing_preferences"):
        card("warn", "Missing: " + esc(", ".join(r["missing_preferences"])))
    for u in r.get("unsupported", []):
        card("na", "Not supported: " + esc(u))
    for d in r.get("data_conflicts", []):
        card(
            "warn",
            f"Data conflict: {esc(d['product_id'])} {esc(d['attribute'])} differs across "
            f"{esc(' vs '.join(d['sources']))}. {esc(d['resolution'])}.",
        )

    if r.get("escalate"):
        color, text = "#c62828", f"Supervisor review: required &ndash; {esc(r['escalation_reason'])}"
    else:
        color, text = "#2e9e4f", "Supervisor review: not needed"
    st.markdown(f'<span class="dot" style="background:{color}"></span>{text}', unsafe_allow_html=True)


app = get_app()
st.subheader("Store Product Assistant")
left, right = st.columns(2, gap="large")

with left:
    question = st.text_area("Customer question", placeholder="Which of these is better for long flights?")
    ids = st.multiselect("Product IDs", approved_product_ids(), default=["HP-100", "HP-200"])
    budget = st.number_input("Budget (max, Rs)", min_value=0, value=0, step=500)
    must = st.text_input("Must have (comma separated)", placeholder="noise cancellation")
    use_case = st.text_input("Use case", placeholder="travel")
    asked = st.button("Ask assistant", type="primary")

if asked and question:
    st.session_state["result"] = ask(
        app,
        {
            "question": question,
            "product_ids": ids,
            "preferences": {
                "budget_max_inr": budget or None,
                "must_have": [m.strip() for m in must.split(",") if m.strip()],
                "use_case": use_case or None,
            },
        },
    )

with right:
    result = st.session_state.get("result")
    if asked and not question:
        st.warning("Enter a customer question.")
    elif result and result["status"] == "done":
        show_response(result["response"])
    elif result:
        card("bad", f"Waiting for supervisor: {esc(result['reason'])}")
        with st.expander("Supervisor panel", expanded=True):
            st.json(pending_draft(app, result["thread_id"]))
            note = st.text_input("Note")
            a, b = st.columns(2)
            if a.button("Approve"):
                st.session_state["result"] = review(app, result["thread_id"], True, note)
                st.rerun()
            if b.button("Reject"):
                st.session_state["result"] = review(app, result["thread_id"], False, note)
                st.rerun()
    else:
        st.caption("The cited answer will appear here.")
