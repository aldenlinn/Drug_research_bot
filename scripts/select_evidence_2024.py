import argparse
import json
import random
import re
from collections import defaultdict

# Boilerplate and non-finding scaffolding that leaked into the old pass. A sentence matching
# any of these is not a result and gets dropped before it can become "evidence".
JUNK = re.compile(
    r"\b(disclaimer|publisher.?s note|copyright|\bmdpi\b|creative commons|licen[sc]e|"
    r"conflict[s]? of interest|competing interests|author[s]?.? contributions|"
    r"data availability|data links|funding|acknowledg|supplementary|corresponding author|"
    r"orcid|c-editor|s-editor|l-editor|t-editor|biorender|"
    r"future (directions|research|studies|work|gwas)|further (studies|research) (are|is) needed|"
    r"this (review|article|paper|study) (aims|seeks|provides an overview)|"
    r"in this (review|article|paper) we|clinicaltrials\.gov|ethics (approval|statement)|"
    r"supplement industry|commercially available|on the market|marketed as)\b",
    re.IGNORECASE,
)
# Section-header words glued to the front of a passage ("Conclusion As genetic data...").
HEADER = re.compile(
    r"^(conclusion[s]?( and outlook)?|result[s]?|discussion|finding[s]?|summary|abstract|"
    r"background|introduction|significance|considerations and future directions|outlook)"
    r"\b[\s:.\-]*",
    re.IGNORECASE,
)
# Figure/table callouts and bare cluster jargon make a sentence unreadable out of context.
NOISE = re.compile(r"\b(panel|cluster\s*\d)\b", re.IGNORECASE)
# A sentence carries a real finding if it states an effect, association, or statistic.
CUE = re.compile(
    r"\b(we (found|show|showed|observed|demonstrate[d]?|report|reveal(ed)?)|significant(ly)?|"
    r"was associated|were associated|associated with|led to|resulted in|compared (with|to)|"
    r"p\s*[<=>]\s*0|95%\s*ci|hazard ratio|odds ratio|relative risk|reduced|increased|improved|"
    r"enhanced|attenuated|decreased|inhibited|suppressed|elevated|ameliorated|efficacy|"
    r"effective|protect(ed|ive)|prevent(ed|s|ion)?|restore[d]?|rescue[d]?|alleviat|mitigat|"
    r"promot(ed|es)|upregulat|downregulat|higher|lower|versus|vs\.)\b",
    re.IGNORECASE,
)

# Parenthetical author-year citations and figure/table refs: remove so the finding reads clean.
CITE_PAREN = re.compile(r"\((?:[^()]*?\bet al\.?,?\s*\d{4}[^()]*)\)")
YEAR_CITE = re.compile(r"\(\s*(?:[A-Z][A-Za-z\-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z\-]+)?,?\s*\d{4}[;,]?\s*)+\)")
FIG_PAREN = re.compile(r"\((?:see\s+)?(?:additional\s+)?(?:fig(?:ure|s)?\.?|table|panel)[^()]*\)", re.IGNORECASE)
FIG_INLINE = re.compile(
    r"\b(?:fig(?:ure|s)?\.?|table)\s*\d+[a-z]?(?:\s*(?:and|,|to|through|[–-])\s*[a-z0-9]+)*",
    re.IGNORECASE,
)
EMPTY_PAREN = re.compile(r"\(\s*[;,]?\s*\)")


def fix_mojibake(s):
    # The harvest was UTF-8 that got decoded as cp1252 somewhere upstream (Muller -> MÃ¼ller).
    # Reverse it only when the tell-tale sequences are present, and only if it round-trips.
    if not s:
        return s
    if any(m in s for m in ("Ã", "Â", "â€", "Å", "Ë")):
        try:
            return s.encode("cp1252", "strict").decode("utf-8", "strict")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
    return s


def clean(text):
    return " ".join((text or "").split())


def scrub(text):
    text = fix_mojibake(text)
    text = CITE_PAREN.sub("", text)
    text = YEAR_CITE.sub("", text)
    text = FIG_PAREN.sub("", text)
    text = FIG_INLINE.sub("", text)
    text = EMPTY_PAREN.sub("", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)   # unglue space-before-punct left by removals
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\b([A-Za-z]+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)  # "in in" -> "in"
    return clean(text)


def split_sentences(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)  # unglue "effect.Discussion"
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean(text))
    return [p.strip() for p in parts if len(p.strip()) > 25]


def pick_evidence(record):
    by_type = defaultdict(list)
    for p in record.get("passages") or []:
        st = (p.get("section_type") or p.get("section") or "").upper()
        by_type[st].append(p.get("text") or "")
    for key in ("CONCL", "RESULTS"):  # task pins evidence to the paper's own conclusion/results
        raw = HEADER.sub("", scrub(" ".join(by_type.get(key, [])).strip()))
        if not raw:
            continue
        sents = [HEADER.sub("", s) for s in split_sentences(raw)]
        keep = [s for s in sents
                if CUE.search(s) and not JUNK.search(s) and not NOISE.search(s)]
        finding = " ".join(keep[:4]).strip()
        if len(finding.split()) >= 18:
            return finding[:1150], key
    return None, None


def detect_study_type(title, evidence, mesh):
    t = (clean(title) + " " + evidence + " " + " ".join(mesh)).lower()
    if re.search(r"\b(randomi[sz]ed|double-blind|placebo-controlled|phase\s*(i{1,3}|[123])\b)", t):
        return "rct"
    if re.search(r"\b(meta-analysis|systematic review|pooled analysis)\b", t):
        return "meta-analysis"
    if re.search(r"\b(mice|mouse|rats?|murine|zebrafish|animals?|in vitro|in vivo|cell line|"
                 r"cultured|primary neurons?|animal model|mouse model|rat model|c57|knockout|"
                 r"knock-?down|transfect|lentivir|aav|organoid|xenograft|hippocampal slices?)\b", t):
        return "preclinical"
    if re.search(r"\b(patients|participants|subjects|cohort|enrolled|retrospective|prospective|"
                 r"case-control|observational)\b", t):
        return "clinical-observational"
    return "review"


def mesh_terms(record):
    out = []
    for m in record.get("mesh") or []:
        term = m.get("term") if isinstance(m, dict) else m
        if term:
            out.append(clean(str(term)))
    return out


def subject_of(record):
    topics = [clean(x) for x in (record.get("topics") or []) if clean(x)]
    return topics[0] if topics else "neuroscience"


def norm_key(evidence):
    return re.sub(r"[^a-z0-9]", "", evidence.lower())[:80]


def build(in_path, out_path, target, per_topic, per_journal, seed):
    seen_ids, seen_findings = set(), set()
    buckets = defaultdict(list)
    read = full = found = 0
    with open(in_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            read += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("has_full_text"):
                continue
            full += 1
            key = rec.get("canonical_key") or rec.get("pmid") or rec.get("id")
            if not key or key in seen_ids:
                continue
            if not rec.get("pmid"):            # need a real PMID to cite and to score retrieval
                continue
            evidence, section = pick_evidence(rec)
            if not evidence:
                continue
            fkey = norm_key(evidence)
            if fkey in seen_findings:           # drop near-duplicate findings
                continue
            seen_ids.add(key)
            seen_findings.add(fkey)
            found += 1
            mesh = mesh_terms(rec)
            cand = {
                "pmid": str(rec.get("pmid")),
                "pmcid": rec.get("pmcid"),
                "doi": rec.get("doi"),
                "journal": clean(rec.get("journal") or ""),
                "year": rec.get("year"),
                "evidence_tier": rec.get("evidence_tier"),
                "study_type": detect_study_type(rec.get("title") or "", evidence, mesh),
                "subject": subject_of(rec),
                "section": section,
                "evidence": evidence,
            }
            buckets[cand["subject"]].append(cand)

    rng = random.Random(seed)
    pool, journal_count = [], defaultdict(int)
    subjects = sorted(buckets, key=lambda s: len(buckets[s]), reverse=True)
    for subj in subjects:
        recs = buckets[subj]
        rng.shuffle(recs)
        taken = 0
        for cand in recs:
            if taken >= per_topic:
                break
            if journal_count[cand["journal"]] >= per_journal:
                continue
            pool.append(cand)
            journal_count[cand["journal"]] += 1
            taken += 1
    rng.shuffle(pool)
    pool = pool[:target]

    with open(out_path, "w", encoding="utf-8") as out:
        for i, cand in enumerate(pool):
            cand["id"] = i
            out.write(json.dumps(cand, ensure_ascii=False) + "\n")

    by_type = defaultdict(int)
    by_subj = defaultdict(int)
    for c in pool:
        by_type[c["study_type"]] += 1
        by_subj[c["subject"]] += 1
    print(f"read {read} | has_full_text {full} | with_finding {found} | pool {len(pool)}")
    print("study_type:", dict(sorted(by_type.items(), key=lambda kv: -kv[1])))
    top = sorted(by_subj.items(), key=lambda kv: -kv[1])[:15]
    print("top subjects:", top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("harvest")
    ap.add_argument("out")
    ap.add_argument("--target", type=int, default=1200)
    ap.add_argument("--per-topic", type=int, default=160)
    ap.add_argument("--per-journal", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.harvest, args.out, args.target, args.per_topic, args.per_journal, args.seed)


if __name__ == "__main__":
    main()
