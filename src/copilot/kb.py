"""Knowledge base: load the Markdown corpus, chunk it, and search it.

Retrieval is deliberately dependency-free -- BM25 over heading-delimited chunks,
scored in-process. A vector DB would add setup friction without changing what
this POC demonstrates: a RETRIEVER span carrying `retrieval.documents`, which is
what Arize's RAG evaluators (relevance, groundedness) bind to.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

from .config import KB_DIR
from .tracing import set_output, span

_TOKEN = re.compile(r"[a-z0-9_]+")
_STOP = frozenset(
    "a an the and or but if is are was were be been being of to in on at for with "
    "from by as it its this that these those i you we they do does did can could "
    "should would will my your our their me us them what how why when where".split()
)


@dataclass(frozen=True)
class Chunk:
    doc_id: str  # e.g. "billing-and-plans.md#Overages"
    source: str  # filename
    heading: str
    text: str


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _split_sections(path: Path) -> list[Chunk]:
    """Split a Markdown file on `##` headings. Each section is one chunk."""
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    title = next((ln[2:].strip() for ln in lines if ln.startswith("# ")), path.stem)
    chunks: list[Chunk] = []
    heading = title
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            chunks.append(
                Chunk(
                    doc_id=f"{path.name}#{heading}",
                    source=path.name,
                    heading=heading,
                    # Prepend the doc title so a chunk is self-describing to the LLM.
                    text=f"[{title} > {heading}]\n{body}",
                )
            )

    for line in lines:
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            buf = []
        elif not line.startswith("# "):
            buf.append(line)
    flush()
    return chunks


@lru_cache(maxsize=1)
def _index() -> tuple[list[Chunk], list[Counter[str]], Counter[str], float]:
    """Build the BM25 index once per process."""
    chunks = [c for p in sorted(KB_DIR.glob("*.md")) for c in _split_sections(p)]
    if not chunks:
        raise RuntimeError(f"No knowledge-base documents found under {KB_DIR}")

    tfs = [Counter(_tokenize(c.text)) for c in chunks]
    df: Counter[str] = Counter()
    for tf in tfs:
        df.update(tf.keys())
    avg_len = sum(sum(tf.values()) for tf in tfs) / len(tfs)
    return chunks, tfs, df, avg_len


def search(query: str, *, top_k: int = 4, min_score: float = 1.0) -> list[tuple[Chunk, float]]:
    """BM25 search. Returns (chunk, score) above `min_score`, best first.

    `min_score` matters for this POC: questions about topics absent from the
    corpus (refunds) must come back empty or near-empty, so the agent's
    grounding behaviour is actually tested rather than papered over by
    marginally-relevant chunks.
    """
    chunks, tfs, df, avg_len = _index()
    n = len(chunks)
    k1, b = 1.5, 0.75

    q_terms = _tokenize(query)
    scored: list[tuple[Chunk, float]] = []
    for chunk, tf in zip(chunks, tfs):
        length = sum(tf.values()) or 1
        score = 0.0
        for term in q_terms:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * length / avg_len))
        if score > min_score:
            scored.append((chunk, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def search_traced(query: str, *, top_k: int = 4) -> list[tuple[Chunk, float]]:
    """`search()` wrapped in a RETRIEVER span with `retrieval.documents` set."""
    with span(
        "kb.search",
        OpenInferenceSpanKindValues.RETRIEVER,
        input_value=query,
        metadata={"top_k": top_k, "retriever": "bm25"},
    ) as sp:
        hits = search(query, top_k=top_k)
        for i, (chunk, score) in enumerate(hits):
            prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}."
            sp.set_attribute(prefix + DocumentAttributes.DOCUMENT_ID, chunk.doc_id)
            sp.set_attribute(prefix + DocumentAttributes.DOCUMENT_CONTENT, chunk.text)
            sp.set_attribute(prefix + DocumentAttributes.DOCUMENT_SCORE, float(score))
        sp.set_attribute("retrieval.hit_count", len(hits))
        set_output(sp, {"hit_count": len(hits), "doc_ids": [c.doc_id for c, _ in hits]})
        return hits


def format_context(hits: list[tuple[Chunk, float]]) -> str:
    """Render retrieved chunks for the LLM prompt."""
    if not hits:
        return "(no matching documentation found)"
    return "\n\n---\n\n".join(f"SOURCE: {c.doc_id}\n{c.text}" for c, _ in hits)


def context_for_ids(doc_ids: Iterable[str]) -> str:
    """Rebuild the retrieved context from doc ids alone.

    Offline evaluation needs the documentation *text*, not just which chunks
    were hit -- a groundedness judge given only ids has to guess, and will.
    The corpus is local and immutable, so ids are enough to reconstruct exactly
    what the agent was shown.
    """
    chunks = {c.doc_id: c for c in _index()[0]}
    wanted = [chunks[i] for i in doc_ids if i in chunks]
    if not wanted:
        return "(no matching documentation found)"
    return "\n\n---\n\n".join(f"SOURCE: {c.doc_id}\n{c.text}" for c in wanted)
