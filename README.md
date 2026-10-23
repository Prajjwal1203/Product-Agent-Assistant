# Store Associate Product Knowledge Assistant

Answers store associates' product questions **only** from approved catalog/policy documents, compares products in
a fixed layout, and hands anything uncertain to a supervisor. LangGraph + Ollama (`gpt-oss:120b-cloud`) + RAG.

## Run it
```powershell
uv sync --extra ui
ollama signin
ollama pull gpt-oss:120b-cloud
ollama pull nomic-embed-text
$env:SESSION_DATE = "2026-10-12"        # optional: pin the date (needed for the promo-expiry checks)
uv run uvicorn store_assistant.api:api --reload      # POST http://127.0.0.1:8000/ask  (docs at /docs)
uv run streamlit run src/store_assistant/ui.py        # optional UI
uv run python -m store_assistant.cli "Which is better for long flights?" --ids HP-100 HP-200 --budget 10000 --must-have "noise cancellation" --use-case travel
uv run pytest            # offline: fake model + fake embeddings
$env:LIVE = "1"; uv run pytest   # against real Ollama
```

## How it works
`intake -> check_input -> classify -> retrieve -> generate -> guard_checks -> (supervisor_review | respond)`

* **Retrieval (RAG):** metadata filters first (approved, effective date reached, not expired, latest version);
  then vector search (Ollama `nomic-embed-text`) over policy/compatibility docs.
* **Code builds** the comparison table (fixed attribute list/order) and attribute facts. **The model only**
  selects facts for free-form questions and writes the recommendation. Everything it says is verified in code.
* **Guard checks decide escalation**, never the model.

## README questions
**Which version wins on conflict, and why?** The version with the latest `effective_from` (ties broken by higher
`version`). It is the most recent approved statement of the product. The conflict is still reported in
`data_conflicts` listing every version. If two versions share the same effective date it cannot be resolved,
so the answer is escalated.

**What counts as "approved"?** `approved: true` in the document metadata, `effective_from <= session date`, and
no `effective_to` before the session date. Drafts (HP-300) and expired promotions never reach retrieval, so they
can't be cited. The model only ever sees the winning version of each document.

**Which questions are always escalated?** (1) policy exceptions: opened item returned and not defective;
(2) unresolved conflicts; (3) any failed citation check (unknown source, wrong product, value not in the source);
(4) a recommendation that breaks a stated budget/must-have or isn't based on a stated preference; (5) generated
availability/discount claims; (6) invalid or unavailable model output. The graph pauses at `supervisor_review`
with `interrupt()` until `/review/{thread_id}` is called.

**What can't the assistant know?** Stock/availability, live prices beyond the catalog, discounts outside an
active approved promo, anything not in the approved documents, and whether a customer's item is really defective.
Availability and discount questions are answered with "cannot say" only.

## Data assumptions and limitations
* Data pack is fictional. `COMP-CHG` text and `CH-30W` (my own product) were added to the pack from the case study.
* Comparison attributes live in each catalog doc's front matter (`attributes:`), so the table is exact.
* Without product IDs, a recommendation considers every approved product in a category (headphones, or chargers if
  "charger" is in the question) and requires budget and use case; otherwise it asks for them (T8).
* Intent classification and policy-exception detection are keyword/regex rules (deterministic, testable); they
  will miss unusual phrasing, and escalation fails safe only for the patterns listed above.
* Personal data detection covers emails, 10-digit phone numbers and "my name is X". Question text is never logged.
* In-memory checkpointer: paused supervisor reviews are lost on restart. Use a SQLite/Postgres checkpointer for production.
