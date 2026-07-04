from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LOG = logging.getLogger("rag_retriever")


def clean_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-]{2,}", (text or "").lower())
        if len(t) > 2
    }


@dataclass(frozen=True)
class RetrievedBlock:
    score: float
    text: str
    title: str
    metadata: dict

    def format(self, max_chars: int = 1400) -> str:
        body = self.text
        if len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0].strip()
        meta = self.metadata or {}
        cite_bits = []
        if meta.get("journal"):
            cite_bits.append(str(meta["journal"]))
        if meta.get("year"):
            cite_bits.append(str(meta["year"]))
        if meta.get("pmid"):
            cite_bits.append(f"PMID {meta['pmid']}")
        if meta.get("doi"):
            cite_bits.append(f"DOI {meta['doi']}")
        cite = "; ".join(cite_bits)
        return f"{body}\n\nCitation: {cite}" if cite else body


class LocalContextRetriever:
    """Sentence-transformer/FAISS retriever with deterministic lexical fallback."""

    def __init__(
        self,
        corpus_path: str | Path,
        index_path: str | Path | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_records: int = 0,
        strict: bool = False,
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.index_path = Path(index_path) if index_path else None
        # The model requested by the caller. The model actually used is read from
        # the index sidecar meta when one exists, because the index is only valid
        # for the model that built it.
        self.requested_embedding_model = embedding_model
        self.embedding_model = embedding_model
        self.strict = strict
        self.records = self._load_records(max_records=max_records)
        self._validate_corpus_schema()
        # Precompute one token set per record so the lexical fallback does not re-tokenize the
        # entire corpus on every query (an O(corpus) regex pass per query at ~7 GB scale).
        self._record_tokens = [
            tokens(f"{rec.get('title', '')} {rec.get('search_text', '')} {rec.get('text', '')}")
            for rec in self.records
        ]
        self._model = None
        self._index = None
        self._load_semantic_index()

    def _validate_corpus_schema(self) -> None:
        """Warn loudly if the corpus looks like a raw harvest file, not the combined corpus.

        The four multi-GB harvest*.jsonl files sit next to the corpus; pointing RAG_CORPUS_PATH
        at one by mistake yields flat records with no 'metadata'/'search_text', so citations go
        empty and lexical scoring loses signal -- silently. Catch that at load."""
        if not self.records:
            return
        first = self.records[0]
        expected_any = ("metadata", "search_text", "chunk_uid")
        if "text" not in first or not any(k in first for k in expected_any):
            LOG.warning(
                "corpus at %s does not look like the combined corpus (missing %s). Did you point "
                "RAG_CORPUS_PATH at a raw harvest file instead of combine_drug_jsonl.py output?",
                self.corpus_path, "/".join(expected_any),
            )

    def _read_index_meta(self) -> dict:
        if not self.index_path:
            return {}
        meta_path = self.index_path.with_name(self.index_path.name + ".meta.json")
        if not meta_path.is_file():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            LOG.warning("could not read index meta %s: %s", meta_path, exc)
            return {}

    def _fallback(self, reason: str) -> None:
        """Disable semantic retrieval. Raise in strict mode, warn otherwise."""
        self._index = None
        self._model = None
        if self.strict:
            raise RuntimeError(
                f"Semantic retrieval unavailable: {reason}. Refusing to fall back to "
                "lexical search because RAG_STRICT_RETRIEVAL is set."
            )
        LOG.warning("%s; using lexical retrieval fallback", reason)

    def _load_records(self, max_records: int) -> list[dict]:
        if not self.corpus_path.is_file():
            raise FileNotFoundError(f"RAG corpus not found: {self.corpus_path}")
        records = []
        bad = 0
        with self.corpus_path.open("r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, start=1):
                if max_records and i > max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines the same way the embed step does, so the record count
                    # and the index vector count stay aligned (the semantic-index parity guard).
                    bad += 1
                    continue
                if not isinstance(obj, dict):
                    # Valid JSON that is not an object (e.g. a bare number/array) would slip past the
                    # decode guard and later blow up the eager rec.get(...) token precompute; drop it
                    # here so a stray scalar line cannot crash retriever construction / serving.
                    bad += 1
                    continue
                records.append(obj)
        if bad:
            LOG.warning("skipped %d malformed corpus line(s) in %s", bad, self.corpus_path)
        LOG.info("loaded %d corpus chunks from %s", len(records), self.corpus_path)
        return records

    def _load_semantic_index(self) -> None:
        if not self.index_path:
            LOG.info("No index path configured; using lexical retrieval")
            return
        if not self.index_path.is_file():
            self._fallback(f"no FAISS index at {self.index_path}")
            return

        meta = self._read_index_meta()
        meta_model = meta.get("embedding_model")
        if meta_model:
            if meta_model != self.requested_embedding_model:
                LOG.warning(
                    "index was built with embedding model %s but %s was requested; "
                    "using the index's model so query and corpus vectors match",
                    meta_model,
                    self.requested_embedding_model,
                )
            self.embedding_model = meta_model

        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self._fallback("FAISS/sentence-transformers not installed")
            return

        index = faiss.read_index(str(self.index_path))
        if index.ntotal != len(self.records):
            self._fallback(
                f"index vectors ({index.ntotal}) != loaded corpus records "
                f"({len(self.records)}). Rebuild the index over the full corpus "
                "without --embed-max-chunks, and load the retriever without a "
                "max_records cap so both sides align"
            )
            return

        model = SentenceTransformer(self.embedding_model)
        model_dim = int(model.get_sentence_embedding_dimension())
        if model_dim != index.d:
            self._fallback(
                f"embedding model {self.embedding_model} produces dim {model_dim} "
                f"but the index expects dim {index.d}. The index and serving model "
                "disagree; rebuild the index with the serving embedding model"
            )
            return

        self._index = index
        self._model = model
        LOG.info(
            "semantic retrieval enabled: model=%s dim=%d vectors=%d",
            self.embedding_model,
            index.d,
            index.ntotal,
        )

    @property
    def semantic_enabled(self) -> bool:
        return self._index is not None and self._model is not None

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedBlock]:
        if not self.records or not clean_text(query):
            return []
        if self.semantic_enabled:
            return self._retrieve_semantic(query, top_k)
        return self._retrieve_lexical(query, top_k)

    def _record_to_block(self, rec: dict, score: float) -> RetrievedBlock:
        return RetrievedBlock(
            score=float(score),
            text=clean_text(rec.get("text")),
            title=clean_text(rec.get("title")),
            metadata=rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {},
        )

    def _retrieve_semantic(self, query: str, top_k: int) -> list[RetrievedBlock]:
        import numpy as np

        q = self._model.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        scores, idxs = self._index.search(q, min(top_k, len(self.records)))
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            out.append(self._record_to_block(self.records[int(idx)], float(score)))
        return out

    def _retrieve_lexical(self, query: str, top_k: int) -> list[RetrievedBlock]:
        qtok = tokens(query)
        if not qtok:
            return []
        scored = []
        for rec, rtok in zip(self.records, self._record_tokens):
            if not rtok:
                continue
            overlap = len(qtok & rtok)
            if overlap <= 0:
                continue
            # BM25-ish small fallback: rewards overlap, dampens very long chunks.
            score = overlap / math.sqrt(len(rtok))
            scored.append((score, rec))
        scored.sort(key=lambda x: (-x[0], clean_text(x[1].get("chunk_uid"))))
        return [self._record_to_block(rec, score) for score, rec in scored[:top_k]]


def format_blocks(blocks: Iterable[RetrievedBlock], max_chars: int = 1400) -> list[str]:
    return [block.format(max_chars=max_chars) for block in blocks]

