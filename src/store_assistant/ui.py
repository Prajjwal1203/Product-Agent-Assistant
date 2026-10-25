"""Optional Streamlit UI: uv sync --extra ui && uv run streamlit run src/store_assistant/ui.py"""
import html

import streamlit as st

from store_assistant.config import session_date
from store_assistant.graph import ask, build_app, pending_draft, review
from store_assistant.render import LABELS
from store_assistant.retrieval import load_docs

st.set_page_config(page_title="Store Product Assistant", page_icon="🛍️", layout="wide")

st.markdown(
    """
<style>
    :root {
        --bg: #f3f7fb;
        --panel: rgba(255, 255, 255, 0.78);
        --panel-alt: #ffffff;
        --primary: #1e5eff;
        --primary-soft: #edf3ff;
        --success: #1b8f5a;
        --warning: #c27912;
        --danger: #c93b3b;
        --neutral: #4a5568;
        --line: #dfe7f1;
        --shadow: 0 14px 30px rgba(17, 24, 39, 0.08);
    }

    .stApp {
        background: linear-gradient(180deg, #eef4ff 0%, var(--bg) 100%);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eff5ff 0%, #f9fbff 100%);
    }

    .hero {
        padding: 1.25rem 1.5rem 1rem;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(30,94,255,0.08), rgba(79,195,247,0.05));
        margin-bottom: 1.25rem;
        box-shadow: var(--shadow);
    }

    .eyebrow {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: var(--primary-soft);
        color: var(--primary);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .hero h1 {
        margin: 0.6rem 0 0.25rem;
        font-size: clamp(2rem, 3vw, 2.8rem);
        line-height: 1.1;
        color: #0f172a;
    }

    .hero p {
        margin: 0;
        color: var(--neutral);
        font-size: 1rem;
    }

    .status-pills {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }

    .status-pill {
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(255,255,255,0.7);
        padding: 0.45rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 600;
        color: #334155;
    }

    .panel {
        background: rgba(255,255,255,0.7);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.1rem 1.15rem;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .section-title {
        margin: 0 0 0.7rem;
        font-size: 0.84rem;
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #475569;
    }

    .card{border-radius:14px;padding:14px 16px;margin:6px 0 14px;border:1px solid;color:#1f2937;background:#fff;box-shadow:0 10px 20px rgba(15,23,42,0.04)}
    .card b{display:block;font-size:1.05rem;color:#0f172a}
    .rec{background:linear-gradient(135deg,#ebfbee,#f5fff8);border-color:#8ad3a2}
    .warn{background:linear-gradient(135deg,#fff5dd,#fffaf0);border-color:#f0c768}
    .na{background:linear-gradient(135deg,#eef2f7,#f8fafc);border-color:#cbd5e1}
    .bad{background:linear-gradient(135deg,#fdecec,#fff4f4);border-color:#f0a2a2}
    .lbl{font-size:.8rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#475569;margin-top:4px}
    .dot{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:8px;vertical-align:middle}
    table.cmp{width:100%;border-collapse:collapse;margin-bottom:6px;border-radius:12px;overflow:hidden}
    table.cmp th,table.cmp td{border:1px solid #dfe7f1;padding:8px 10px;text-align:left;vertical-align:top}
    table.cmp th{background:#edf3ff;color:#0f172a;font-weight:700}
    table.cmp td{background:#fff;color:#1f2937}
    .stAlert, .stSuccess, .stWarning, .stInfo {border-radius: 12px;}
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

st.markdown(
    """
<div class="hero">
    <div class="eyebrow">Store associate copilot</div>
    <h1>Store Product Assistant</h1>
    <p>Compare products with approved catalog and policy guidance, while keeping supervisor review as a safety net.</p>
    <div class="status-pills">
        <span class="status-pill">Approved-only answers</span>
        <span class="status-pill">Cited comparisons</span>
        <span class="status-pill">Safety guards enabled</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
<div class="panel">
    <div class="section-title">Quick guide</div>
    <p>Ask about product fit, budget, must-haves, and use cases. The assistant only answers from approved documents.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.sidebar.caption("Use approved catalog IDs and policy references for the most reliable answers.")

left, right = st.columns([1.05, 1.35], gap="large")

with left:
    st.markdown('<div class="section-title">Customer request</div>', unsafe_allow_html=True)
    with st.form("product_question_form"):
        question = st.text_area("Customer question", placeholder="Which of these is better for long flights?")
        ids = st.multiselect("Product IDs", approved_product_ids(), default=["HP-100", "HP-200"])
        budget = st.number_input("Budget (max, Rs)", min_value=0, value=0, step=500)
        must = st.text_input("Must have (comma separated)", placeholder="noise cancellation")
        use_case = st.text_input("Use case", placeholder="travel")
        asked = st.form_submit_button("Ask assistant", type="primary", use_container_width=True)

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
    st.markdown('<div class="section-title">Answer panel</div>', unsafe_allow_html=True)
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
        st.info("The cited answer will appear here after you submit a question.")
