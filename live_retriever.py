from __future__ import annotations

import html
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

LOGGER = logging.getLogger("live_retriever")

# A core page of abstracts is a few hundred KB. Cap the read so a misbehaving service cannot make
# us buffer an unbounded body before parsing. Over-cap is treated as a miss (fail-open holds).
MAX_RESPONSE_BYTES = 4 * 1024 * 1024

# Query words that mean the user wants current work, so we sort by publication date instead of
# relevance. Deliberately small and lexical: a miss just falls back to relevance sort, never errors.
RECENCY_HINTS = (
    "latest", "recent", "current", "newest", "this year", "up to date", "up-to-date",
    "emerging", "state of the art", "state-of-the-art", "2024", "2025", "2026",
)

TAG_RE = re.compile(r"<[^>]+>")


def strip_markup(text: str) -> str:
    """Europe PMC titles and abstracts carry HTML entities and inline tags (<sub>, <i>). Unescape
    and drop tags so block text stays on the plain-text distribution the adapter was trained on."""
    if not text:
        return ""
    text = html.unescape(text)
    text = TAG_RE.sub("", text)
    return " ".join(text.split())


def clip(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def wants_recent(query: str) -> bool:
    q = (query or "").lower()
    return any(hint in q for hint in RECENCY_HINTS)


def citation_line(rec: dict) -> str:
    """Journal; year; PMID; DOI, matching rag_retriever.RetrievedBlock.format so live and FAISS
    citations read identically to the adapter. Only present fields are joined."""
    bits = []
    journal_info = rec.get("journalInfo") or {}
    journal = (journal_info.get("journal") or {}).get("title") or rec.get("journalTitle")
    year = rec.get("pubYear") or journal_info.get("yearOfPublication")
    pmid = rec.get("pmid")
    doi = rec.get("doi")
    if journal:
        bits.append(str(journal))
    if year:
        bits.append(str(year))
    if pmid:
        bits.append(f"PMID {pmid}")
    if doi:
        bits.append(f"DOI {doi}")
    return "; ".join(bits)


def format_record(rec: dict, max_chars: int) -> str | None:
    """One Europe PMC core result as a context block matching format_blocks output: body then a
    Citation line. Returns None when there is no abstract, since a bare title is nothing to ground
    on and would just dilute the context budget."""
    abstract = strip_markup(rec.get("abstractText") or "")
    if not abstract:
        return None
    title = strip_markup(rec.get("title") or "")
    if title:
        sep = " " if title.endswith((".", "?", "!")) else ". "
        body = f"{title}{sep}{abstract}"
    else:
        body = abstract
    body = clip(body, max_chars)
    cite = citation_line(rec)
    return f"{body}\n\nCitation: {cite}" if cite else body


@dataclass(frozen=True)
class LiveLiteratureRetriever:
    """Fetches recent + relevant Europe PMC abstracts at query time so the bot knows literature
    published after the FAISS index was built. Fail-open in the ie_client mold: short timeout,
    never raises, returns [] on any problem, so a dead API leaves the answer FAISS-identical."""

    # Empty endpoint => OFF, same convention as the IE endpoint. Defaults to Europe PMC's free REST
    # search so the live leg is ON out of the box; set RAG_LIVE_ENDPOINT="" to disable it.
    endpoint: str = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    max_blocks: int = 4
    timeout_s: float = 2.0
    max_block_chars: int = 1400
    # Over-fetch a bit so abstract-less or duplicate hits can be skipped and still fill max_blocks.
    page_size: int = 12

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint.strip())

    def build_url(self, query: str) -> str:
        params = {
            "query": f"{query.strip()} AND (HAS_ABSTRACT:Y)",
            "format": "json",
            "resultType": "core",
            "pageSize": str(max(self.page_size, self.max_blocks)),
        }
        # Freshest first when the question asks for current work; relevance (API default) otherwise.
        if wants_recent(query):
            params["sort"] = "P_PDATE_D desc"
        return self.endpoint + "?" + urllib.parse.urlencode(params)

    def fetch_blocks(self, query: str) -> list[str]:
        """Recent + relevant abstracts as numbered-ready context blocks. NEVER raises; [] on any
        problem (off, empty query, timeout, bad payload) so generation is unchanged when live is
        unavailable."""
        if not self.enabled or not (query or "").strip():
            return []
        req = urllib.request.Request(
            self.build_url(query), method="GET", headers={"Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                LOGGER.warning("live payload over %d bytes; skipping", MAX_RESPONSE_BYTES)
                return []
            payload = json.loads(raw.decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            LOGGER.warning("live literature unavailable (%s); proceeding without it", exc)
            return []
        except Exception as exc:  # noqa: BLE001 -- last-resort backstop: the live leg must never break a turn
            LOGGER.warning("live literature failed unexpectedly (%s); proceeding without it", exc)
            return []
        results = (((payload or {}).get("resultList") or {}).get("result")) or []
        blocks: list[str] = []
        seen: set = set()
        for rec in results:
            if not isinstance(rec, dict):
                continue
            key = rec.get("pmid") or rec.get("doi") or rec.get("id")
            if key and key in seen:
                continue
            block = format_record(rec, self.max_block_chars)
            if not block:
                continue
            if key:
                seen.add(key)
            blocks.append(block)
            if len(blocks) >= self.max_blocks:
                break
        return blocks
