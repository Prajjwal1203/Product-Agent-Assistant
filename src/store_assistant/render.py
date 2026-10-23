"""Turn the structured response into something an associate can read in under 10 seconds."""

LABELS = {
    "noise_cancellation": "Noise cancellation",
    "battery_life": "Battery",
    "weight": "Weight",
    "price_inr": "Price (Rs)",
    "power_output": "Power",
    "ports": "Ports",
    "connector": "Connector",
}


def render(r: dict) -> str:
    lines = []
    if r.get("escalate"):
        lines.append(f"ESCALATED TO SUPERVISOR: {r['escalation_reason']}")
        lines.append("")
    c = r.get("comparison")
    if c and c["rows"]:
        ids = [row["product_id"] for row in c["rows"]]
        lines.append("| | " + " | ".join(ids) + " |")
        lines.append("|---|" + "---|" * len(ids))
        for i, attr in enumerate(c["attributes"]):
            lines.append(f"| {LABELS.get(attr, attr)} | " + " | ".join(row["values"][i] for row in c["rows"]) + " |")
        sources = sorted({cit for row in c["rows"] for cit in row["citations"]})
        lines.append("")
        lines.append(f"Sources: {', '.join(sources)}")
        lines.append("")
    elif r["facts"]:
        for f in r["facts"]:
            lines.append(f"- {f['product_id']} {f['attribute']}: {f['value']}  [{f['citation']}]")
        lines.append("")
    rec = r.get("recommendation")
    if rec:
        lines.append(f"RECOMMENDATION: {rec['product_id']}. {rec['reasoning']}")
        lines.append("")
        lines.append(f"(Based on: {', '.join(rec['based_on_preferences'])})")
        lines.append("")
    if r["missing_preferences"]:
        lines.append("Ask the customer: " + ", ".join(r["missing_preferences"]))
    for u in r["unsupported"]:
        lines.append(f"CANNOT SAY: {u}")
    for d in r["data_conflicts"]:
        lines.append(
            f"DATA CONFLICT: {d['product_id']} {d['attribute']} differs across {' vs '.join(d['sources'])}. {d['resolution']}."
        )
    return "\n".join(lines).strip()
