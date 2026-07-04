import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from drugbot_prompts import SYSTEM_PROMPT  # the canonical persona; identical to the serving repo

DISCLAIMER = ("This is educational information, not diagnosis or treatment, and clinical "
              "decisions belong to a licensed clinician or pharmacist.")


def load_candidates(path):
    by_id = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            by_id[int(c["id"])] = c
    return by_id


def load_shards(shard_dir):
    rows = []
    for shard in sorted(glob.glob(os.path.join(shard_dir, "batch_*.jsonl"))):
        with open(shard, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"  skip malformed {os.path.basename(shard)}:{lineno} ({exc})")
    return rows


def assemble(prompt_evidence, question):
    return f"{SYSTEM_PROMPT}\n\nContext:\n[1] {prompt_evidence}\n\nQuestion: {question}"


def existing_ids(out_path):
    ids = set()
    if not os.path.exists(out_path):
        return ids
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("_src_id")
            if sid is not None:
                ids.add(int(sid))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["w", "a"], default="w")
    ap.add_argument("--tag-source", action="store_true",
                    help="add a hidden _src_id key for cross-batch dedup (stripped for training)")
    args = ap.parse_args()

    cands = load_candidates(args.candidates)
    rows = load_shards(args.shards)
    seen = existing_ids(args.out) if args.mode == "a" else set()

    written = 0
    dropped = {"dup": 0, "no_candidate": 0, "no_pmid_cite": 0, "no_disclaimer": 0,
               "no_marker": 0, "pmid_mismatch": 0, "empty": 0, "em_dash": 0}
    out_lines = []
    for r in rows:
        try:
            rid = int(r["id"])
        except (KeyError, ValueError, TypeError):
            dropped["empty"] += 1
            continue
        if rid in seen:
            dropped["dup"] += 1
            continue
        cand = cands.get(rid)
        if not cand:
            dropped["no_candidate"] += 1
            continue
        question = (r.get("question") or "").strip()
        response = (r.get("response") or "").strip()
        if not question or not response:
            dropped["empty"] += 1
            continue
        true_pmid = str(cand["pmid"])
        if str(r.get("pmid")) != true_pmid:
            dropped["pmid_mismatch"] += 1
            continue
        if f"PMID {true_pmid}" not in response:
            dropped["no_pmid_cite"] += 1
            continue
        if "[1]" not in response:
            dropped["no_marker"] += 1
            continue
        if not response.endswith(DISCLAIMER):
            dropped["no_disclaimer"] += 1
            continue
        if "—" in response or "—" in question:
            dropped["em_dash"] += 1
            continue
        prompt = assemble(cand["evidence"], question)
        rec = {"prompt": prompt, "response": response}
        if args.tag_source:
            rec["_src_id"] = rid
        out_lines.append(json.dumps(rec, ensure_ascii=False))
        seen.add(rid)
        written += 1

    with open(args.out, args.mode, encoding="utf-8") as out:
        for line in out_lines:
            out.write(line + "\n")

    print(f"shards rows in: {len(rows)} | written: {written} | out: {args.out} (mode {args.mode})")
    print("dropped:", {k: v for k, v in dropped.items() if v})


if __name__ == "__main__":
    main()
