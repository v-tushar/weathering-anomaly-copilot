"""Knowledge base: markdown chunks plus TF-IDF retrieval.

TF-IDF rather than embeddings keeps the demo offline, deterministic and free.
Search is the same interface either way, so an embedding or hybrid (BM25 +
embeddings) retriever can replace it without touching the agent.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

KB_DIR = Path(__file__).parent / "knowledge"
_CHUNK = re.compile(r"^## \[(?P<id>[A-Z]+-\d+)\] (?P<title>.+)$", re.M)
_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str
    source: str


def section_steps(text: str) -> list[str]:
    """Procedure steps in a manual section, verbatim: numbered lines ("1. ..."), or
    the sentences after "Actions:". Empty if the section only explains."""
    steps: list[str] = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+\.\s+(.*)", line)
        if m:
            steps.append(m.group(1).strip())
        elif steps and line.startswith((" ", "\t")) and line.strip():
            steps[-1] += " " + line.strip()          # wrapped continuation of a step
        elif steps and line.strip():
            break                                    # prose after the list
    if steps:
        return steps
    flat = " ".join(text.split())
    if "Actions:" in flat:
        return [s for s in _SENT.split(flat.split("Actions:", 1)[1].strip()) if s]
    return []


def safety_notes(text: str) -> list[str]:
    """Sentences that restrict who may do the work or how (verbatim from the manual)."""
    flat = " ".join(text.split())
    return [s for s in _SENT.split(flat)
            if re.search(r"technician|high voltage|locked out|powered down", s, re.I)]


def load_chunks(kb_dir: Path = KB_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        body = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S)
        heads = list(_CHUNK.finditer(body))
        for h, nxt in zip(heads, heads[1:] + [None]):
            text = body[h.end(): nxt.start() if nxt else len(body)].strip()
            chunks.append(Chunk(h["id"], h["title"].strip(), text, path.name))
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chunk ids in knowledge base")
    return chunks


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or load_chunks()
        self.by_id = {c.id: c for c in self.chunks}
        # Content hash: stamped on every diagnosis so an audit can tell which
        # version of the manual the copilot was reading.
        self.version = hashlib.sha256(
            "".join(f"{c.id}{c.title}{c.text}" for c in self.chunks).encode()).hexdigest()[:8]
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self._m = self._vec.fit_transform([f"{c.title} {c.title} {c.text}" for c in self.chunks])

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = self._vec.transform([query])
        scores = (self._m @ q.T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]
