from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DISCLAIMER_HINT = "not diagnosis or treatment"
CORPUS_REQUIRED = ("chunk_uid", "doc_id", "chunk_id", "text", "search_text", "metadata")
MAX_REPORTED = 10


def nonempty_str(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def iter_lines(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if stripped:
                yield lineno, stripped


def validate_pairs(path: Path) -> dict:
    errors: list[str] = []
    warnings = 0
    rows = 0
    empty_prompt = empty_response = has_nul = 0
    missing_disclaimer = 0
    resp_len_min = None
    resp_len_max = 0
    resp_len_sum = 0

    for lineno, line in iter_lines(path):
        rows += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            if len(errors) < MAX_REPORTED:
                errors.append(f"line {lineno}: invalid JSON ({exc})")
            continue
        if not isinstance(rec, dict):
            if len(errors) < MAX_REPORTED:
                errors.append(f"line {lineno}: not a JSON object")
            continue
        prompt, response = rec.get("prompt"), rec.get("response")
        if not nonempty_str(prompt):
            empty_prompt += 1
            if len(errors) < MAX_REPORTED:
                errors.append(f"line {lineno}: empty/missing 'prompt'")
        if not nonempty_str(response):
            empty_response += 1
            if len(errors) < MAX_REPORTED:
                errors.append(f"line {lineno}: empty/missing 'response'")
            continue
        if "\x00" in prompt or "\x00" in response:
            has_nul += 1
            if len(errors) < MAX_REPORTED:
                errors.append(f"line {lineno}: contains NUL byte")
        if DISCLAIMER_HINT not in response:
            missing_disclaimer += 1
            warnings += 1
        n = len(response)
        resp_len_min = n if resp_len_min is None else min(resp_len_min, n)
        resp_len_max = max(resp_len_max, n)
        resp_len_sum += n

    ok = not errors and empty_prompt == 0 and empty_response == 0 and has_nul == 0
    return {
        "file": str(path),
        "rows": rows,
        "ok": ok,
        "empty_prompt": empty_prompt,
        "empty_response": empty_response,
        "has_nul": has_nul,
        "missing_disclaimer": missing_disclaimer,
        "response_chars_min": resp_len_min or 0,
        "response_chars_max": resp_len_max,
        "response_chars_avg": round(resp_len_sum / rows) if rows else 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_corpus(path: Path) -> dict:
    errors: list[str] = []
    rows = 0
    empty_text = 0
    missing_fields = 0

    for lineno, line in iter_lines(path):
        rows += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            if len(errors) < MAX_REPORTED:
                errors.append(f"line {lineno}: invalid JSON ({exc})")
            continue
        if not isinstance(rec, dict):
            if len(errors) < MAX_REPORTED:
                errors.append(f"line {lineno}: not a JSON object")
            continue
        absent = [k for k in CORPUS_REQUIRED if k not in rec]
        if absent:
            missing_fields += 1
            if len(errors) < MAX_REPORTED:
                errors.append(f"line {lineno}: missing fields {absent}")
        if not nonempty_str(rec.get("text")):
            empty_text += 1
            if len(errors) < MAX_REPORTED:
                errors.append(f"line {lineno}: empty/missing 'text'")

    ok = not errors and empty_text == 0 and missing_fields == 0
    return {
        "file": str(path),
        "rows": rows,
        "ok": ok,
        "empty_text": empty_text,
        "missing_fields": missing_fields,
        "errors": errors,
    }


def report(title: str, result: dict) -> None:
    print(f"\n=== {title}: {result['file']} ===")
    for key, val in result.items():
        if key in ("file", "errors"):
            continue
        print(f"  {key}: {val}")
    if result.get("errors"):
        print("  first problems:")
        for e in result["errors"]:
            print(f"    - {e}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate LoRA pairs / RAG corpus JSONL.")
    parser.add_argument("--pairs", default=None, help="LoRA prompt/response JSONL")
    parser.add_argument("--corpus", default=None, help="RAG corpus JSONL")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.pairs and not args.corpus:
        print("nothing to validate: pass --pairs and/or --corpus", file=sys.stderr)
        return 2

    all_ok = True
    if args.pairs:
        res = validate_pairs(Path(args.pairs))
        report("PAIRS", res)
        all_ok = all_ok and res["ok"]
    if args.corpus:
        res = validate_corpus(Path(args.corpus))
        report("CORPUS", res)
        all_ok = all_ok and res["ok"]

    print("\nRESULT:", "OK - training ready" if all_ok else "FAILED - see problems above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
