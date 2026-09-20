"""
retriever.py
------------
The retrieval component of the RAG pipeline: turns the 76 preprocessed
chunks (see build_corpus.py / chunks.json) into TF-IDF vectors and answers
"which chunks are most relevant to this question" via cosine similarity.

This replaces the hand-rolled JavaScript TF-IDF implementation from the
browser version of this project with scikit-learn's TfidfVectorizer,
which is the standard, well-tested tool for exactly this job in Python.
Everything here runs server-side now; nothing runs in the user's browser.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNKS_PATH = Path(__file__).parent / "chunks.json"


@dataclass
class RetrievedChunk:
    """One retrieval hit: the chunk plus how well it matched the query."""
    id: str
    paper_id: str
    paper_title: str
    paper_short: str
    year: int
    section: str
    text: str
    score: float


class Retriever:
    """
    Loads chunks.json once and builds a TF-IDF index over it. `retrieve()`
    can then be called many times (once per user question) cheaply, since
    the expensive part -- fitting the vectorizer's vocabulary and IDF
    weights over the corpus -- happens exactly once, at construction time.
    """

    def __init__(self, chunks_path: Path = CHUNKS_PATH):
        self.chunks: List[dict] = json.loads(chunks_path.read_text(encoding="utf-8"))
        if not self.chunks:
            raise ValueError(f"No chunks loaded from {chunks_path}")

        texts = [c["text"] for c in self.chunks]

        # ngram_range=(1,2) means both single words ("attention") and
        # adjacent word pairs ("multi-head") become vocabulary terms.
        # Bigrams help recover some of the phrase-level signal that plain
        # unigram TF-IDF misses (see PROJECT_DOCUMENTATION.md, section 4,
        # for the "few-shot" ranking limitation this partially addresses).
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(texts)  # shape: (n_chunks, n_terms)

    def retrieve(self, query: str, k: int = 8) -> List[RetrievedChunk]:
        """Return the top-k chunks most similar to `query`, ranked descending."""
        query = (query or "").strip()
        if not query:
            return []

        query_vec = self.vectorizer.transform([query])          # shape: (1, n_terms)
        sims = cosine_similarity(query_vec, self.matrix)[0]      # shape: (n_chunks,)

        k = min(k, len(self.chunks))
        top_indices = sims.argsort()[::-1][:k]                   # indices of k largest scores

        results = []
        for idx in top_indices:
            c = self.chunks[idx]
            results.append(RetrievedChunk(
                id=c["id"],
                paper_id=c["paper_id"],
                paper_title=c["paper_title"],
                paper_short=c["paper_short"],
                year=c["year"],
                section=c["section"],
                text=c["text"],
                score=float(sims[idx]),
            ))
        return results

    def highlight(self, text: str, query: str) -> str:
        """
        Wrap query terms found in `text` with <mark> tags, for the sources
        panel in the UI. This mirrors highlightSnippet() from the original
        JS version, but is now plain server-side string processing that
        produces HTML the Jinja template inserts with the `|safe` filter.
        """
        query_terms = {t.lower() for t in re.findall(r"[a-zA-Z0-9\-]+", query) if len(t) > 2}
        if not query_terms:
            return _escape(text)

        pieces = re.split(r"(\s+)", text)
        out = []
        for piece in pieces:
            bare = re.sub(r"[^a-zA-Z0-9\-]", "", piece).lower()
            if bare in query_terms:
                out.append(f"<mark>{_escape(piece)}</mark>")
            else:
                out.append(_escape(piece))
        return "".join(out)


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )
