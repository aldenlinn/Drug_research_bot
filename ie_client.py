from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

LOGGER = logging.getLogger("ie_client")

# Bound the response read: a priors payload is small (a few ranked concepts), so anything larger
# is a misconfiguration or a misbehaving service. Capping keeps a runaway body from being buffered
# whole before parsing. Fail-open holds regardless -- over-cap is treated as a miss.
MAX_RESPONSE_BYTES = 512 * 1024


def clip(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def format_concept_prior(item: dict, max_chars: int) -> str:
    """One concept prior as a labeled block. Tolerates missing fields (fail-soft parsing)."""
    label = str(item.get("label") or item.get("orbit") or "concept").strip()
    score = item.get("score")
    members = item.get("members") or []
    member_labels = [str(m) for m in members if str(m).strip()][:8]
    head = f"Knowledge prior (Intuition Engine): {label}"
    if isinstance(score, (int, float)):
        head += f" (relevance {float(score):.3f})"
    if member_labels:
        head += " -- related: " + ", ".join(member_labels)
    return clip(head, max_chars)


def format_relation_prior(item: dict, max_chars: int) -> str:
    head = str(item.get("label_head") or item.get("head") or "?").strip()
    tail = str(item.get("label_tail") or item.get("tail") or "?").strip()
    direction = item.get("direction_prior")
    line = f"Knowledge prior (relation): {head} -> {tail}"
    if isinstance(direction, (int, float)):
        line += f" (direction {float(direction):.2f})"
    return clip(line, max_chars)


def priors_from_payload(payload: dict, max_blocks: int, max_chars: int) -> list[str]:
    """Turn an IE /v1/retrieve response into labeled text blocks, capped for context budget."""
    blocks: list[str] = []
    for item in (payload.get("concept_priors") or []):
        if isinstance(item, dict):
            blocks.append(format_concept_prior(item, max_chars))
        if len(blocks) >= max_blocks:
            return blocks
    for item in (payload.get("relation_priors") or []):
        if isinstance(item, dict):
            blocks.append(format_relation_prior(item, max_chars))
        if len(blocks) >= max_blocks:
            break
    return blocks


@dataclass(frozen=True)
class IERetrievalClient:
    endpoint: str = ""
    namespace: str = "biomed"
    top_k: int = 3
    timeout_s: float = 1.5
    api_key: str = ""
    max_block_chars: int = 600

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint.strip())

    def fetch_priors(self, query: str, seeds: list[str] | None = None) -> list[str]:
        """Return labeled IE prior blocks for a query. NEVER raises; [] on any problem.

        `seeds` lets the caller pass orbit ids it already extracted (e.g. 'mesh:<UI>'); the
        server resolves free text otherwise. Either way a miss returns [] and generation
        proceeds unchanged.
        """
        if not self.enabled or not (query or "").strip():
            return []
        body = {"query": query, "top_k": self.top_k, "namespace": self.namespace}
        if seeds:
            body["seeds"] = seeds
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            req.add_header("x-api-key", self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            LOGGER.warning("IE priors unavailable (%s); proceeding FAISS-only", exc)
            return []
        except Exception as exc:  # noqa: BLE001 -- last-resort backstop: IE must never break a turn
            LOGGER.warning("IE priors failed unexpectedly (%s); proceeding FAISS-only", exc)
            return []
        if not isinstance(payload, dict):
            return []
        return priors_from_payload(payload, max_blocks=self.top_k, max_chars=self.max_block_chars)
