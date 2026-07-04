from __future__ import annotations

import argparse
import csv
import json
import re

from Reasearch_Drug_Chatbot import GemmaRagEngine, ServingConfig, configure_logging

PMID_RE = re.compile(r"PMID[:\s#]*([0-9]{6,9})", re.IGNORECASE)
# Abstention signals aligned with SYSTEM_PROMPT ("if the material does not contain the answer,
# say so plainly") and the no-context serving branch ("did not contain enough evidence").
ABSTAIN_RE = re.compile(
    r"(do(es)?\s+not\s+contain|did\s+not\s+contain|not\s+contain\s+enough|"
    r"no\s+(relevant\s+)?evidence|not\s+(found|available|present)\s+in|could\s+not\s+find|"
    r"can(not|'t|\s+not)\s+find|insufficient\s+(evidence|context|information)|"
    r"not\s+enough\s+(evidence|information|context)|(local\s+)?corpus\s+(does|did|do)\s+not|"
    r"outside\s+(the\s+)?(scope|corpus)|unable\s+to\s+(find|answer)|"
    r"no\s+information\s+(in|was|is)|not\s+in\s+the\s+(provided\s+)?context)",
    re.IGNORECASE,
)
DISCLAIMER_HINT = "not diagnosis or treatment"

WORD = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\-]{2,}")
STOP = set(
    "the a an and or of to in for with on at by from as is are was were be been being this that "
    "these those it its their they them we you your our will can may might could should would has "
    "have had not but which who whom into than then also such more most some any each both study "
    "studies finding findings evidence result results according recent research patients clinical "
    "compared versus using used show shows showed suggest suggests reported report".split()
)


def toks(text):
    return {w for w in WORD.findall((text or "").lower()) if w not in STOP and len(w) > 3}


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join((text or "").split())) if s.strip()]


def retrieved(engine, question, top_k):
    blocks = engine.retriever.retrieve(question, top_k=top_k) if engine.retriever else []
    pmids, texts = set(), []
    for b in blocks:
        pm = (b.metadata or {}).get("pmid")
        if pm not in (None, ""):
            pmids.add(str(pm))
        texts.append(b.text or "")
    return pmids, texts


def content_sentences(answer):
    # Grounding applies to factual claims, not the fixed disclaimer or the bare citation tail.
    out = []
    for s in split_sentences(answer):
        if DISCLAIMER_HINT in s.lower():
            continue
        stripped = re.sub(r"\(PMID[^)]*\)", "", s).strip(" .;:[]1")
        if len(toks(stripped)) < 2:
            continue
        out.append(s)
    return out


def grounded_check(answer, ctx_texts, threshold):
    # Lexical proxy: fraction of a claim's content words present in the retrieved context. Flags
    # low-overlap sentences as possible hallucinations for manual review (retrieval/citation are
    # exact; grounding is a heuristic per the eval spec).
    ctx = set()
    for t in ctx_texts:
        ctx |= toks(t)
    flagged = []
    for s in content_sentences(answer):
        st = toks(re.sub(r"\(PMID[^)]*\)", "", s))
        if not st:
            continue
        support = len(st & ctx) / len(st)
        if support < threshold:
            flagged.append((round(support, 2), s))
    return (len(flagged) == 0), flagged


def pct(vals):
    vals = [v for v in vals if v is not None]
    return round(100.0 * sum(1 for v in vals if v) / len(vals), 1) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/gold_eval.jsonl")
    ap.add_argument("--out", default="eval_results.csv")
    ap.add_argument("--top-k", type=int, default=0, help="0 => use serving RAG_RETRIEVAL_TOP_K")
    ap.add_argument("--ground-threshold", type=float, default=0.30)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    configure_logging()
    engine = GemmaRagEngine(ServingConfig()).load()
    if engine.retriever is None:
        print("WARNING: no retriever loaded (RAG_CORPUS_PATH missing). retrieval_hit/citation_ok "
              "will all be False. Point RAG_CORPUS_PATH/RAG_INDEX_PATH at the serving corpus.")
    top_k = args.top_k or engine.config.retrieval_top_k

    gold = [json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()]
    rows = []
    for i, item in enumerate(gold, 1):
        q = item["question"]
        gold_pmid = item.get("gold_pmid")
        in_corpus = bool(item.get("in_corpus", gold_pmid is not None))

        pmids, ctx = retrieved(engine, q, top_k)
        answer = engine.answer(q, max_new_tokens=args.max_new_tokens)
        cited = set(PMID_RE.findall(answer))
        no_invented = cited.issubset(pmids)  # every cited PMID was actually retrieved

        if in_corpus:
            retrieval_hit = str(gold_pmid) in pmids
            citation_ok = bool(cited) and no_invented
            grounded, flagged = grounded_check(answer, ctx, args.ground_threshold)
            abstains_ok = None
        else:
            retrieval_hit = None
            citation_ok = None
            grounded, flagged = None, []
            abstains_ok = bool(ABSTAIN_RE.search(answer)) and no_invented and len(cited) == 0

        rows.append({
            "n": i, "type": "answer" if in_corpus else "abstain",
            "gold_pmid": gold_pmid or "", "retrieved_pmids": ";".join(sorted(pmids)),
            "n_ctx": len(ctx), "cited_pmids": ";".join(sorted(cited)),
            "retrieval_hit": retrieval_hit, "citation_ok": citation_ok,
            "grounded": grounded, "abstains_ok": abstains_ok,
            "n_flagged": len(flagged), "question": q, "answer": answer,
            "flagged": " || ".join(f"[{s}] {t}" for s, t in flagged),
        })

    def mark(v):
        return "-" if v is None else ("PASS" if v else "FAIL")

    print("\n" + "=" * 100)
    print(f"{'#':>2}  {'type':<7} {'gold':<9} {'hit':<5} {'cite':<5} {'grnd':<5} {'abst':<5} {'flg':<3} question")
    print("-" * 100)
    for r in rows:
        print(f"{r['n']:>2}  {r['type']:<7} {str(r['gold_pmid']):<9} "
              f"{mark(r['retrieval_hit']):<5} {mark(r['citation_ok']):<5} {mark(r['grounded']):<5} "
              f"{mark(r['abstains_ok']):<5} {r['n_flagged']:<3} {r['question'][:60]}")
    print("=" * 100)

    ans = [r for r in rows if r["type"] == "answer"]
    absr = [r for r in rows if r["type"] == "abstain"]
    print(f"answerable: {len(ans)} | abstention: {len(absr)}")
    print(f"retrieval_hit : {pct([r['retrieval_hit'] for r in ans])}%")
    print(f"citation_ok   : {pct([r['citation_ok'] for r in ans])}%")
    print(f"grounded      : {pct([r['grounded'] for r in ans])}%  (lexical proxy; review flagged rows)")
    print(f"abstention    : {pct([r['abstains_ok'] for r in absr])}%")

    flagged_any = [r for r in rows if r["n_flagged"]]
    if flagged_any:
        print("\nPossible-hallucination sentences to eyeball (low context overlap):")
        for r in flagged_any:
            print(f"  Q{r['n']} (gold {r['gold_pmid']}): {r['flagged']}")

    fields = ["n", "type", "gold_pmid", "retrieved_pmids", "n_ctx", "cited_pmids",
              "retrieval_hit", "citation_ok", "grounded", "abstains_ok", "n_flagged",
              "question", "answer", "flagged"]
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
