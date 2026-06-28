from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

LOG = logging.getLogger("build_pairs")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# Output schema is {"prompt": ..., "response": ...} so it drops straight into
# Train_loRa.py create_conversation, which reads sample["prompt"] and
# sample["response"]. No change to that function is needed.

DISCLAIMER = "Educational use only, not medical advice."

# Optional scope filter. Off by default. Turn on with --filter-scope to keep
# only rows whose text mentions one of these terms.
SCOPE_TERMS = [
    "psilocybin", "lsd", "lysergic", "ketamine", "mdma", "psychedelic",
    "neuroprotection", "neurodegeneration", "neuroplasticity", "bdnf",
    "cognition", "cognitive", "depression", "anxiety", "antiaging",
    "anti-aging", "aging", "longevity", "reprogramming", "dementia",
    "alzheimer", "parkinson", "serotonin", "5-ht2a",
]

HARVEST_QUESTION_STEMS = [
    "What does the research say about {topic}?",
    "Summarize the findings on {topic}.",
    "What are the reported effects and benefits related to {topic}?",
    "What is known from recent studies about {topic}?",
]


def in_scope(text: str) -> bool:
    low = (text or "").lower()
    return any(term in low for term in SCOPE_TERMS)


def clean_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def trim_to_chars(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if stop > int(max_chars * 0.5):
        return cut[:stop + 1].strip()
    return cut.strip()


def extract_year(value) -> str:
    match = re.search(r"(19|20)\d{2}", str(value or ""))
    return match.group(0) if match else ""


def citation_line(journal, year, pmid, doi) -> str:
    bits = []
    src = clean_text(journal)
    yr = clean_text(year)
    if src and yr:
        bits.append(f"{src} ({yr})")
    elif src:
        bits.append(src)
    if clean_text(pmid):
        bits.append(f"PMID {clean_text(pmid)}")
    elif clean_text(doi):
        bits.append(f"DOI {clean_text(doi)}")
    return ("Source: " + ", ".join(bits) + ".") if bits else ""


def load_hf_split(name: str, split: str):
    from datasets import load_dataset
    last = None
    for kwargs in ({}, {"trust_remote_code": True}):
        try:
            return load_dataset(name, split=split, **kwargs)
        except Exception as exc:
            last = exc
    LOG.warning("could not load %s: %r", name, last)
    return None


def pairs_from_medchat(filter_scope: bool, cap):
    ds = load_hf_split("ngram/medchat-qa", "train")
    if ds is None:
        return []
    cols = set(ds.column_names)
    if not {"question", "answer"} <= cols:
        LOG.warning("medchat-qa columns unexpected: %s", sorted(cols))
        return []
    out = []
    for row in ds:
        q = clean_text(row.get("question"))
        a = clean_text(row.get("answer"))
        if not q or not a:
            continue
        if filter_scope and not in_scope(q + " " + a):
            continue
        out.append({"prompt": q, "response": a})
        if cap and len(out) >= cap:
            break
    LOG.info("medchat-qa pairs: %d", len(out))
    return out


def pairs_from_biodex(filter_scope: bool, cap, max_chars: int):
    ds = load_hf_split("BioDEX/BioDEX-Reactions", "train")
    if ds is None:
        return []
    cols = set(ds.column_names)
    if not {"title", "abstract", "reactions"} <= cols:
        LOG.warning("BioDEX columns unexpected: %s", sorted(cols))
        return []
    out = []
    for row in ds:
        title = clean_text(row.get("title"))
        abstract = clean_text(row.get("abstract"))
        reactions = clean_text(row.get("reactions"))
        if not abstract or not reactions:
            continue
        if filter_scope and not in_scope(title + " " + abstract + " " + reactions):
            continue
        cite = citation_line(
            row.get("journal"), extract_year(row.get("pubdate")),
            row.get("pmid"), row.get("doi"),
        )
        prompt = (
            "Summarize the adverse drug reactions and clinical findings reported in: "
            f"\"{trim_to_chars(title, 240)}\"."
        )
        response = f"{trim_to_chars(abstract, max_chars)}\n\nReported reactions: {reactions}."
        if cite:
            response += f"\n\n{cite}"
        response += f"\n\n{DISCLAIMER}"
        out.append({"prompt": prompt, "response": response})
        if cap and len(out) >= cap:
            break
    LOG.info("BioDEX pairs: %d", len(out))
    return out


def pairs_from_harvest(path: str, filter_scope: bool, cap, max_chars: int):
    p = Path(path)
    if not p.exists():
        LOG.warning("harvest file not found: %s", p)
        return []
    out = []
    with p.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = clean_text(rec.get("text"))
            title = clean_text(rec.get("title"))
            if not text or not title:
                continue
            if filter_scope and not in_scope(title + " " + text):
                continue
            stem = HARVEST_QUESTION_STEMS[i % len(HARVEST_QUESTION_STEMS)]
            prompt = stem.format(topic=trim_to_chars(title, 200))
            cite = citation_line(
                rec.get("journal"), rec.get("year"),
                rec.get("pmid"), rec.get("doi"),
            )
            response = trim_to_chars(text, max_chars)
            if cite:
                response += f"\n\n{cite}"
            response += f"\n\n{DISCLAIMER}"
            out.append({"prompt": prompt, "response": response})
            if cap and len(out) >= cap:
                break
    LOG.info("harvest pairs: %d", len(out))
    return out


def dedupe(pairs):
    seen = set()
    out = []
    for pair in pairs:
        key = pair["prompt"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(pair)
    return out


def write_pairs(pairs, path: str):
    with Path(path).open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False))
            fh.write("\n")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Build SFT prompt/response pairs for the drug-information LoRA."
    )
    parser.add_argument("--out", default="pairs.jsonl")
    parser.add_argument("--harvest", default="harvest.jsonl")
    parser.add_argument(
        "--sources", default="medchat,biodex,harvest",
        help="Comma list from: medchat, biodex, harvest",
    )
    parser.add_argument(
        "--filter-scope", action="store_true",
        help="Keep only rows mentioning a scope term.",
    )
    parser.add_argument(
        "--max-chars", type=int, default=1200,
        help="Character budget for long answer bodies.",
    )
    parser.add_argument("--medchat-max", type=int, default=0)
    parser.add_argument("--biodex-max", type=int, default=0)
    parser.add_argument("--harvest-max", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    sources = {s.strip().lower() for s in args.sources.split(",") if s.strip()}

    pairs = []
    if "medchat" in sources:
        pairs += pairs_from_medchat(args.filter_scope, args.medchat_max or None)
    if "biodex" in sources:
        pairs += pairs_from_biodex(args.filter_scope, args.biodex_max or None, args.max_chars)
    if "harvest" in sources:
        pairs += pairs_from_harvest(args.harvest, args.filter_scope, args.harvest_max or None, args.max_chars)

    before = len(pairs)
    pairs = dedupe(pairs)
    random.Random(args.seed).shuffle(pairs)
    write_pairs(pairs, args.out)
    LOG.info(
        "total pairs: %d (%d after dedupe), written to %s",
        before, len(pairs), args.out,
    )
    if not pairs:
        LOG.warning("no pairs written, check source availability above")


if __name__ == "__main__":
    main()