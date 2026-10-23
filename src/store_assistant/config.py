import os
from datetime import date
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def session_date() -> date:
    """Today, unless SESSION_DATE=YYYY-MM-DD pins it (needed for deterministic expiry tests)."""
    raw = os.getenv("SESSION_DATE")
    return date.fromisoformat(raw) if raw else date.today()
