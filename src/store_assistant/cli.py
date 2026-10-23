import argparse
import json

from .graph import build_app, ask, pending_draft, review
from .render import render


def main():
    p = argparse.ArgumentParser(description="Ask the store assistant a question")
    p.add_argument("question")
    p.add_argument("--ids", nargs="*", default=[])
    p.add_argument("--budget", type=int)
    p.add_argument("--must-have", nargs="*", default=[])
    p.add_argument("--use-case")
    p.add_argument("--json", action="store_true", help="print raw JSON instead of the readable layout")
    a = p.parse_args()
    request = {
        "question": a.question,
        "product_ids": a.ids,
        "preferences": {"budget_max_inr": a.budget, "must_have": a.must_have, "use_case": a.use_case},
    }
    app = build_app()
    result = ask(app, request)
    if result["status"] == "pending_supervisor":
        print(f"\nPAUSED for supervisor: {result['reason']}")
        print(json.dumps(pending_draft(app, result["thread_id"]), indent=2))
        approved = input("Supervisor approve? [y/N] ").strip().lower() == "y"
        result = review(app, result["thread_id"], approved)
    print(json.dumps(result["response"], indent=2) if a.json else "\n" + render(result["response"]))


if __name__ == "__main__":
    main()
