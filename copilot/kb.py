"""Knowledge base: markdown chunks plus TF-IDF retrieval.

TF-IDF rather than embeddings keeps the demo offline, deterministic and free.
Search is the same interface either way, so an embedding or hybrid (BM25 +
embeddings) retriever can replace it without touching the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

KB_DIR = Path(__file__).parent / "knowledge"
_CHUNK = re.compile(r"^## \[(?P<id>[A-Z]+-\d+)\] (?P<title>.+)$", re.M)


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str
    source: str


def load_chunks(kb_dir: Path = KB_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        body = re.sub(r"<!--.*?-->", "", path.read_text(), flags=re.S)
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
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self._m = self._vec.fit_transform([f"{c.title} {c.title} {c.text}" for c in self.chunks])

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = self._vec.transform([query])
        scores = (self._m @ q.T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]
