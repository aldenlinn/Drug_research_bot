

import argparse
import datetime
import hashlib
import json
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET

import requests

TOPICS = [
    "neuroprotection",
    "neurodegeneration",
    "brain aging senescence",
    "antiaging",
    "cognitive enhancement",
    "treatment resistant depression",
    "ketamine depression",
    "psilocybin",
    "BDNF neuroplasticity",
    "neurite outgrowth",
    "dendritic spine plasticity",
    "partial reprogramming",
    "epigenetic reprogramming aging",
    "Yamanaka factors aging",
]

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
OA_SERVICE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{}/unicode"

SOURCE_RANK = {"pubmed": 1, "europepmc": 2, "pmc": 3}
RETRY_STATUS = (429, 500, 502, 503, 504)
DOI_PREFIX = re.compile(r"^(https?://(dx\.)?doi\.org/)", re.IGNORECASE)
YEAR = re.compile(r"(19|20)\d{2}")

PMC_SECTION_MAP = {
    "ABSTRACT": "abstract", "INTRO": "introduction", "METHODS": "methods",
    "RESULTS": "results", "DISCUSS": "conclusions", "CONCL": "conclusions",
}


def log(message):
    print(message, flush=True)


def load_env(path):
    """Load KEY=VALUE lines from a .env file, tolerating CRLF and quotes."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def clean(value):
    if value is None:
        return ""
    return " ".join(str(value).split())


def clean_doi(value):
    if not value:
        return None
    doi = DOI_PREFIX.sub("", str(value).strip()).strip().rstrip(".")
    return doi.lower() or None


def extract_year(value):
    if value is None:
        return None
    match = YEAR.search(str(value))
    return int(match.group(0)) if match else None


def normalize_cc(raw):
    if not raw:
        return "unknown"
    text = " ".join(str(raw).strip().lower().replace("_", "-").split())
    if "cc0" in text or text in ("pd", "public domain", "public-domain"):
        return "public-domain"
    if "nc" in text and "by" in text:
        return "CC-BY-NC"
    if "sa" in text and "by" in text:
        return "CC-BY-SA"
    if "by" in text:
        return "CC-BY"
    return "unknown"


TIER_RULES = (
    ("meta-analysis", "meta-analysis"),
    ("randomized controlled trial", "rct"),
    ("clinical trial", "rct"),
    ("observational study", "observational"),
    ("cohort", "observational"),
    ("systematic review", "review"),
    ("review", "review"),
    ("preprint", "preprint"),
)


def evidence_tier(pub_types, is_preprint=False):
    if is_preprint:
        return "preprint"
    lowered = [str(p).lower() for p in (pub_types or [])]
    for needle, tier in TIER_RULES:
        if any(needle in p for p in lowered):
            return tier
    return "review"


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def date_range(days):
    if not days:
        return None, None
    today = datetime.datetime.now(datetime.timezone.utc).date()
    start = today - datetime.timedelta(days=days)
    return start.isoformat(), today.isoformat()


class Http:
    """Session with per-host pacing and retry/backoff with jitter."""

    def __init__(self, default_params=None):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "csc525-harvest/1.0 (academic; sanctioned API use)"
        self.defaults = default_params or {}
        self.next_allowed = {}

    def get(self, url, params=None, rps=3, timeout=40):
        host = url.split("/")[2]
        merged = dict(self.defaults)
        if params:
            merged.update({k: v for k, v in params.items() if v is not None})
        attempt = 0
        while True:
            now = time.monotonic()
            wait = self.next_allowed.get(host, 0.0) - now
            if wait > 0:
                time.sleep(wait)
            self.next_allowed[host] = time.monotonic() + (1.0 / rps if rps else 0.0)
            try:
                response = self.session.get(url, params=merged, timeout=timeout)
                if response.status_code in RETRY_STATUS:
                    raise RuntimeError("HTTP {}".format(response.status_code))
                response.raise_for_status()
                return response
            except requests.HTTPError:
                raise
            except Exception as exc:
                if attempt >= 5:
                    raise
                time.sleep(random.uniform(0.0, min(30.0, 1.0 * (2 ** attempt))))
                attempt += 1


def pubmed_records(http, term, days, cap, ncbi_rps):
    """Yield abstract records for a term, paging via the History server so a
    high-volume term is not clipped at the 10k esearch ceiling."""
    start_date, end_date = date_range(days)
    search = http.get(EUTILS + "/esearch.fcgi", rps=ncbi_rps, params={
        "db": "pubmed", "term": term, "usehistory": "y", "retmode": "json",
        "retmax": 0, "reldate": days or None, "datetype": "pdat" if days else None,
    }).json().get("esearchresult", {})
    count = int(search.get("count", 0))
    webenv = search.get("webenv")
    query_key = search.get("querykey")
    if not count or not webenv:
        return
    target = count if not cap else min(count, cap)
    batch = 200
    fetched = 0
    while fetched < target:
        size = min(batch, target - fetched)
        xml = http.get(EUTILS + "/efetch.fcgi", rps=ncbi_rps, params={
            "db": "pubmed", "WebEnv": webenv, "query_key": query_key,
            "retstart": fetched, "retmax": size, "retmode": "xml",
        }).content
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            break
        articles = root.findall(".//PubmedArticle")
        if not articles:
            break
        for article in articles:
            record = parse_pubmed_article(article, term)
            if record:
                yield record
        fetched += len(articles)


def parse_pubmed_article(article, term):
    abstract = " ".join(
        ("{}: {}".format(node.get("Label").strip(), clean("".join(node.itertext())))
         if node.get("Label") else clean("".join(node.itertext())))
        for node in article.findall("./MedlineCitation/Article/Abstract/AbstractText")
        if clean("".join(node.itertext())))
    if not abstract:
        return None
    pmid = clean(article.findtext("./MedlineCitation/PMID")) or None
    doi = None
    for node in article.findall("./MedlineCitation/Article/ELocationID"):
        if (node.get("EIdType") or "").lower() == "doi":
            doi = clean_doi(node.text)
            break
    pmcid = None
    for node in article.findall("./PubmedData/ArticleIdList/ArticleId"):
        kind = (node.get("IdType") or "").lower()
        if kind == "doi" and not doi:
            doi = clean_doi(node.text)
        elif kind == "pmc":
            pmcid = clean(node.text) or None
    pub_types = [clean(p.text) for p in article.findall(
        "./MedlineCitation/Article/PublicationTypeList/PublicationType")]
    return {
        "source": "pubmed", "pmid": pmid, "pmcid": pmcid, "doi": doi,
        "title": clean(article.findtext("./MedlineCitation/Article/ArticleTitle")) or None,
        "journal": clean(article.findtext("./MedlineCitation/Article/Journal/Title")) or None,
        "year": extract_year(article.findtext(
            "./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year"))
            or extract_year(article.findtext(
                "./MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate")),
        "license": "abstract-only",
        "is_open_access": bool(pmcid),
        "has_full_text": False,
        "section": "abstract",
        "evidence_tier": evidence_tier(pub_types),
        "text": abstract,
        "url": "https://pubmed.ncbi.nlm.nih.gov/{}/".format(pmid) if pmid else None,
        "topics": {term},
    }


def europepmc_records(http, term, days, cap):
    """Yield abstract records from Europe PMC, paging by cursorMark."""
    start_date, end_date = date_range(days)
    query = term
    if start_date:
        query = "({}) AND (FIRST_PDATE:[{} TO {}])".format(term, start_date, end_date)
    cursor = "*"
    fetched = 0
    while True:
        payload = http.get(EPMC, rps=4, params={
            "query": query, "format": "json", "resultType": "core",
            "pageSize": 100, "cursorMark": cursor, "sort": "P_PDATE_D desc",
        }).json()
        results = (payload.get("resultList") or {}).get("result") or []
        for result in results:
            record = parse_epmc_result(result, term)
            if record:
                yield record
                fetched += 1
                if cap and fetched >= cap:
                    return
        nxt = payload.get("nextCursorMark")
        if not results or not nxt or nxt == cursor:
            return
        cursor = nxt


def parse_epmc_result(result, term):
    abstract = clean(result.get("abstractText"))
    if not abstract:
        return None
    source_db = clean(result.get("source"))
    pmcid = clean(result.get("pmcid")) or None
    is_oa = clean(result.get("isOpenAccess")).upper() == "Y"
    in_epmc = clean(result.get("inEPMC")).upper() == "Y"
    raw_license = result.get("license")
    if raw_license:
        license_name = normalize_cc(raw_license)
    elif is_oa and in_epmc:
        license_name = "unknown"
    else:
        license_name = "abstract-only"
    pub_types = (result.get("pubTypeList") or {}).get("pubType") or []
    pmid = clean(result.get("pmid")) or None
    doi = clean_doi(result.get("doi"))
    return {
        "source": "europepmc", "pmid": pmid, "pmcid": pmcid, "doi": doi,
        "title": clean(result.get("title")) or None,
        "journal": clean(((result.get("journalInfo") or {}).get("journal") or {}).get("title")) or None,
        "year": extract_year(result.get("pubYear")) or extract_year(result.get("firstPublicationDate")),
        "license": license_name,
        "is_open_access": is_oa,
        "has_full_text": False,
        "section": "abstract",
        "evidence_tier": evidence_tier(pub_types, is_preprint=(source_db == "PPR")),
        "text": abstract,
        "url": ("https://europepmc.org/article/MED/{}".format(pmid) if pmid
                else "https://europepmc.org/article/PMC/{}".format(pmcid) if pmcid
                else ("https://doi.org/{}".format(doi) if doi else None)),
        "topics": {term},
    }


def pmc_fulltext(http, pmcid, ncbi_rps):
    """Return a full-text record for an OA PMCID, or None if not OA/usable."""
    try:
        oa_xml = http.get(OA_SERVICE, rps=ncbi_rps, params={"id": pmcid}).text
        root = ET.fromstring(oa_xml.encode("utf-8"))
    except Exception:
        return None
    if root.find("error") is not None:
        return None
    record_el = root.find(".//record")
    if record_el is None:
        return None
    license_name = normalize_cc(record_el.get("license"))
    if license_name == "unknown":
        return None
    try:
        collection = http.get(BIOC.format(pmcid), rps=ncbi_rps).json()
    except Exception:
        return None
    if isinstance(collection, list):
        collection = collection[0] if collection else {}
    docs = (collection or {}).get("documents") or []
    if not docs:
        return None
    title = None
    buckets = {}
    pmid = doi = None
    for passage in docs[0].get("passages") or []:
        infons = passage.get("infons") or {}
        pmid = pmid or clean(infons.get("article-id_pmid")) or None
        doi = doi or clean_doi(infons.get("article-id_doi"))
        text = clean(passage.get("text"))
        if not text:
            continue
        stype = (infons.get("section_type") or "").upper()
        if stype == "TITLE" and title is None:
            title = text
            continue
        mapped = PMC_SECTION_MAP.get(stype)
        if mapped:
            buckets.setdefault(mapped, []).append(text)
    body = " ".join(
        " ".join(buckets.get(section, []))
        for section in ("abstract", "introduction", "methods", "results", "conclusions")).strip()
    if not body:
        return None
    return {
        "source": "pmc", "pmid": pmid, "pmcid": pmcid, "doi": doi,
        "title": title, "journal": None, "year": None,
        "license": license_name, "is_open_access": True, "has_full_text": True,
        "section": "full", "evidence_tier": "review", "text": body,
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/{}/".format(pmcid),
        "topics": set(),
    }


def canonical_key(record):
    if record.get("doi"):
        return "doi:" + record["doi"]
    if record.get("pmid"):
        return "pmid:" + record["pmid"]
    if record.get("pmcid"):
        return "pmcid:" + record["pmcid"]
    title = (record.get("title") or "").lower().strip()
    if title:
        return "title:" + hashlib.sha1(title.encode("utf-8")).hexdigest()
    return "blank:" + hashlib.sha1((record.get("text") or "").encode("utf-8")).hexdigest()


def merge(index, record, stats):
    """Insert or merge a record into the canonical index. Returns nothing."""
    key = canonical_key(record)
    existing = index.get(key)
    if existing is None:
        index[key] = record
        return
    stats["duplicates_collapsed"] += 1
    # Union provenance ids and the topic set across the two source views.
    for field in ("pmid", "pmcid", "doi", "journal", "year"):
        if not existing.get(field) and record.get(field):
            existing[field] = record[field]
    existing["topics"] |= record.get("topics", set())
    existing["is_open_access"] = existing.get("is_open_access") or record.get("is_open_access")
    # Richer source wins the body text, license, and section.
    if SOURCE_RANK[record["source"]] > SOURCE_RANK[existing["source"]]:
        for field in ("source", "license", "has_full_text", "section", "text",
                      "evidence_tier", "url", "title"):
            if record.get(field) is not None:
                existing[field] = record[field]


def harvest(http, topics, days, max_per_topic, fulltext_cap, ncbi_rps):
    index = {}
    stats = {"pubmed_raw": 0, "europepmc_raw": 0, "duplicates_collapsed": 0,
             "fulltext_upgraded": 0, "fulltext_attempted": 0}
    for term in topics:
        before = len(index)
        for record in europepmc_records(http, term, days, max_per_topic):
            stats["europepmc_raw"] += 1
            merge(index, record, stats)
        for record in pubmed_records(http, term, days, max_per_topic, ncbi_rps):
            stats["pubmed_raw"] += 1
            merge(index, record, stats)
        log("  {:<32} +{} unique (total {})".format(term, len(index) - before, len(index)))

    if fulltext_cap:
        candidates = [r for r in index.values()
                      if r.get("pmcid") and not r["has_full_text"] and r.get("is_open_access")]
        log("full text: {} OA candidates, upgrading up to {}".format(len(candidates), fulltext_cap))
        for record in candidates[:fulltext_cap]:
            stats["fulltext_attempted"] += 1
            full = pmc_fulltext(http, record["pmcid"], ncbi_rps)
            if full:
                full["topics"] = set(record.get("topics", set()))
                merge(index, full, stats)
                stats["fulltext_upgraded"] += 1
    return index, stats


def finalize(record):
    stamp = now_iso()
    out = dict(record)
    out["topics"] = sorted(record.get("topics", set()))
    out["id"] = hashlib.sha1(canonical_key(record).encode("utf-8")).hexdigest()
    out["canonical_key"] = canonical_key(record)
    out["retrieved_at"] = stamp
    out["disclaimer"] = ("Educational / harm reduction use only. Not medical advice. "
                         "Always consult a physician.")
    return out


def write_output(index, path):
    with open(path, "w", encoding="utf-8") as handle:
        for record in index.values():
            handle.write(json.dumps(finalize(record), ensure_ascii=False))
            handle.write("\n")


def tally(records, field):
    counts = {}
    for record in records:
        counts[str(record.get(field))] = counts.get(str(record.get(field)), 0) + 1
    return counts


def print_summary(index, stats, path):
    records = list(index.values())
    raw = stats["pubmed_raw"] + stats["europepmc_raw"]
    log("\n===== HARVEST SUMMARY =====")
    log("raw records pulled : europepmc {}, pubmed {} (total {})".format(
        stats["europepmc_raw"], stats["pubmed_raw"], raw))
    log("duplicates collapsed: {}".format(stats["duplicates_collapsed"]))
    log("full text upgraded : {}/{} attempted".format(
        stats["fulltext_upgraded"], stats["fulltext_attempted"]))
    log("unique articles    : {}".format(len(records)))
    if raw:
        log("dedup rate         : {:.1%}".format(stats["duplicates_collapsed"] / raw))
    log("by winning source  : {}".format(tally(records, "source")))
    log("by license         : {}".format(tally(records, "license")))
    log("by evidence tier   : {}".format(tally(records, "evidence_tier")))
    log("with full text     : {}".format(sum(1 for r in records if r["has_full_text"])))
    log("written to         : {}".format(path))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="One-swoop literature harvester (Europe PMC + PubMed + PMC).")
    parser.add_argument("--out", default=None, help="Output JSONL (default harvest.jsonl beside this script).")
    parser.add_argument("--days", type=int, default=365, help="Recency window in days (0 = all time).")
    parser.add_argument("--max-per-topic", type=int, default=800,
                        help="Cap abstracts per topic per source (0 = no cap).")
    parser.add_argument("--fulltext", type=int, default=200,
                        help="Max OA articles to upgrade to full text (0 = abstracts only).")
    parser.add_argument("--from", dest="since", default=None,
                    help="Start date YYYY-MM-DD (overrides --days).")

    return parser


def main(argv):
    args = build_arg_parser().parse_args(argv)
    days = args.days
    if args.since:
        start = datetime.date.fromisoformat(args.since)
        days = (datetime.datetime.now(datetime.timezone.utc).date() - start).days

    script_dir = os.path.dirname(os.path.abspath(__file__))
    load_env(os.path.join(script_dir, ".env"))
    key = os.environ.get("NCBI_API_KEY", "").strip()
    email = os.environ.get("NCBI_EMAIL", "").strip()
    ncbi_rps = 10 if key else 3
    params = {"tool": "csc525-harvest"}
    if email:
        params["email"] = email
    if key:
        params["api_key"] = key
    http = Http(default_params=params)
    out_path = args.out or os.path.join(script_dir, "harvest.jsonl")

    log("topics: {} | days: {} | max/topic: {} | full text: {} | NCBI {} rps".format(
        len(TOPICS), days, args.max_per_topic, args.fulltext,
        ncbi_rps if key else "{} (no key)".format(ncbi_rps)))
    index, stats = harvest(http, TOPICS, days, args.max_per_topic, args.fulltext, ncbi_rps)
    write_output(index, out_path)
    print_summary(index, stats, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
