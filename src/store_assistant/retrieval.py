"""Retrieval = metadata filtering (approved, in-date, latest version) + vector search for policy docs."""
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import frontmatter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from .config import DATA_DIR, EMBED_MODEL

POLICY_TYPES = ("compat", "policy", "promo")


def _d(value) -> date | None:
    return None if value is None else date.fromisoformat(str(value))


@dataclass(frozen=True)
class Doc:
    doc_id: str
    product_id: str
    doc_type: str
    category: str | None
    version: int
    effective_from: date
    effective_to: date | None
    approved: bool
    attributes: dict
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc_id}@v{self.version}"

    def as_chunk(self) -> dict:
        return {
            "citation": self.citation,
            "product_id": self.product_id,
            "doc_type": self.doc_type,
            "category": self.category,
            "attributes": dict(self.attributes),
            "text": self.text,
        }


def load_docs(directory: Path = DATA_DIR) -> list[Doc]:
    docs = []
    for path in sorted(Path(directory).glob("*.md")):
        post = frontmatter.load(path)
        m = post.metadata
        docs.append(
            Doc(
                doc_id=m["doc_id"],
                product_id=m["product_id"],
                doc_type=m["doc_type"],
                category=m.get("category"),
                version=int(m.get("version", 1)),
                effective_from=_d(m["effective_from"]),
                effective_to=_d(m.get("effective_to")),
                approved=bool(m.get("approved", False)),
                attributes=dict(m.get("attributes") or {}),
                text=post.content.strip(),
            )
        )
    return docs


class Retriever:
    def __init__(self, docs: list[Doc] | None = None, embeddings=None):
        self.docs = docs if docs is not None else load_docs()
        self._embeddings = embeddings
        self._stores: dict[date, InMemoryVectorStore] = {}

    @property
    def embeddings(self):
        if self._embeddings is None:
            from langchain_ollama import OllamaEmbeddings

            self._embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        return self._embeddings

    # ---- filters -------------------------------------------------------
    def active(self, on: date) -> list[Doc]:
        """Approved and in effect on the session date. Drafts and expired docs never pass."""
        return [
            d
            for d in self.docs
            if d.approved and d.effective_from <= on and (d.effective_to is None or on <= d.effective_to)
        ]

    def product_status(self, product_id: str, on: date) -> str:
        mine = [d for d in self.docs if d.product_id == product_id and d.doc_type == "catalog"]
        if not mine:
            return "unknown"
        return "ok" if any(d in self.active(on) for d in mine) else "not_approved"

    def products_in_category(self, category: str, on: date) -> list[str]:
        ids = [d.product_id for d in self.active(on) if d.doc_type == "catalog" and d.category == category]
        return list(dict.fromkeys(ids))

    def promos(self, product_ids: list[str], on: date, *, expired: bool) -> list[Doc]:
        out = []
        for d in self.docs:
            if d.doc_type != "promo" or not d.approved:
                continue
            if product_ids and d.product_id not in product_ids and d.product_id != "ALL":
                continue
            is_expired = d.effective_to is not None and d.effective_to < on
            if expired == is_expired and (is_expired or d.effective_from <= on):
                out.append(d)
        return out

    # ---- version resolution ---------------------------------------------
    def resolve(self, product_id: str, on: date) -> tuple[Doc | None, list[dict], bool]:
        """Return (winning doc, conflicts, unresolved). Latest effective_from wins; ties are unresolved."""
        docs = sorted(
            (d for d in self.active(on) if d.product_id == product_id and d.doc_type == "catalog"),
            key=lambda d: (d.effective_from, d.version),
        )
        if not docs:
            return None, [], False
        winner = docs[-1]
        if len(docs) == 1:
            return winner, [], False
        unresolved = docs[-1].effective_from == docs[-2].effective_from
        conflicts = []
        for attr in winner.attributes:
            if len({d.attributes.get(attr) for d in docs}) > 1:
                conflicts.append(
                    {
                        "product_id": product_id,
                        "attribute": attr,
                        "sources": [d.citation for d in docs],
                        "resolution": (
                            "Unresolved: versions share the same effective date"
                            if unresolved
                            else f"Used latest effective version {winner.citation} (effective {winner.effective_from})"
                        ),
                    }
                )
        return winner, conflicts, unresolved

    # ---- RAG over policy / compatibility docs ----------------------------
    def policy_docs(self, question: str, on: date, k: int = 2) -> list[Doc]:
        docs = [d for d in self.active(on) if d.doc_type in POLICY_TYPES]
        if not docs:
            return []
        if on not in self._stores:
            store = InMemoryVectorStore(self.embeddings)
            store.add_documents(
                [Document(page_content=d.text, metadata={"citation": d.citation}) for d in docs]
            )
            self._stores[on] = store
        hits = self._stores[on].similarity_search(question, k=k)
        by_citation = {d.citation: d for d in docs}
        return [by_citation[h.metadata["citation"]] for h in hits]
