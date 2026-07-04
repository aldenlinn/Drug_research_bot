import argparse
import json
import re

# Out-of-corpus probes: real drug questions with no answer in this neuroprotection/aging corpus.
# The bot must say the local corpus does not contain the answer and NOT fabricate a citation.
ABSTENTION = [
    "What is the target INR range for a patient on warfarin with a mechanical heart valve?",
    "What is the first-line oral antibiotic for an uncomplicated urinary tract infection in adults?",
    "How effective is finasteride at 1 mg daily for treating male pattern hair loss?",
]


def clean(text):
    return " ".join((text or "").split())


def short_point(evidence, limit=190):
    parts = re.split(r"(?<=[.!?])\s+", clean(evidence))
    out = ""
    for p in parts:
        if len(out) + len(p) + 1 > limit and out:
            break
        out = (out + " " + p).strip()
    return out[:limit]


def load(candidates_path, pairs_path):
    cands = {}
    for line in open(candidates_path, encoding="utf-8"):
        if line.strip():
            c = json.loads(line)
            cands[int(c["id"])] = c
    rows = []
    for line in open(pairs_path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            sid = r.get("_src_id")
            if sid is not None and int(sid) in cands:
                rows.append((r, cands[int(sid)]))
    return rows


def build(candidates_path, pairs_path, out_path, n_answerable):
    rows = load(candidates_path, pairs_path)
    by_subject = {}
    for r, cd in rows:
        by_subject.setdefault(cd["subject"], []).append((r, cd))

    picked, used_pmid = [], set()

    # Seed with the three strongest evidence tiers if present, so the gold set spans study types.
    for r, cd in rows:
        if len(picked) >= 3:
            break
        if cd["study_type"] in ("rct", "meta-analysis", "clinical-observational") and cd["pmid"] not in used_pmid:
            picked.append((r, cd))
            used_pmid.add(cd["pmid"])

    # Then round-robin across subjects for topic coverage.
    subjects = sorted(by_subject, key=lambda s: -len(by_subject[s]))
    i = 0
    while len(picked) < n_answerable and subjects:
        subj = subjects[i % len(subjects)]
        bucket = by_subject[subj]
        added = False
        for r, cd in bucket:
            if cd["pmid"] not in used_pmid:
                picked.append((r, cd))
                used_pmid.add(cd["pmid"])
                added = True
                break
        i += 1
        if i > len(subjects) * 40:
            break
        if not added and i % len(subjects) == 0:
            pass

    picked = picked[:n_answerable]
    with open(out_path, "w", encoding="utf-8") as out:
        for r, cd in picked:
            question = r["prompt"].split("\n\nQuestion: ", 1)[1].strip()
            entry = {
                "question": question,
                "gold_pmid": str(cd["pmid"]),
                "expected_point": short_point(cd["evidence"]),
                "subject": cd["subject"],
                "study_type": cd["study_type"],
                "in_corpus": True,
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        for q in ABSTENTION:
            entry = {
                "question": q,
                "gold_pmid": None,
                "expected_point": "Out of corpus scope: the bot should state the local corpus does "
                                  "not contain the answer and must not fabricate a citation.",
                "subject": "out-of-corpus",
                "study_type": "none",
                "in_corpus": False,
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"wrote {len(picked)} answerable + {len(ABSTENTION)} abstention = "
          f"{len(picked) + len(ABSTENTION)} gold questions to {out_path}")
    subj_counts = {}
    for _, cd in picked:
        subj_counts[cd["subject"]] = subj_counts.get(cd["subject"], 0) + 1
    print("coverage:", subj_counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/evidence_candidates_2024.jsonl")
    ap.add_argument("--pairs", default="data/qa_pairs_2024_v2.jsonl")
    ap.add_argument("--out", default="data/gold_eval.jsonl")
    ap.add_argument("--n", type=int, default=17)
    args = ap.parse_args()
    build(args.candidates, args.pairs, args.out, args.n)


if __name__ == "__main__":
    main()
