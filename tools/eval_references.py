"""
SEEKER Evaluator — Reference Hallucination Checker
---------------------------------------------------
Queries the pipeline database for all sources in a run, cross-checks each
against three live verification sources:

  1. Semantic Scholar — academic papers, preprints
  2. OpenAlex — broader academic coverage including books
  3. Claude web search — Anthropic API with web_search tool, catches books,
     old works, and anything the academic APIs miss

Produces a Markdown report grading every reference.

Grades:
  VERIFIED     — Found in API with matching title, author, and year
  PLAUSIBLE    — Found with partial match (title similar, year ±2, author fuzzy)
  UNVERIFIED   — Not found in any source (could be book/old work with no coverage)
  SUSPICIOUS   — Not found and claims look inconsistent
  HALLUCINATED — Found but the actual content contradicts what the pipeline claims

Requires: ANTHROPIC_API_KEY in .env for Claude web search (optional — degrades
gracefully to S2 + OpenAlex only if not set).

Usage:
    python3 tools/eval_references.py --run-id RUN-20260407-022355-242D
    python3 tools/eval_references.py --run-id RUN-20260407-022355-242D --types seminal
    python3 tools/eval_references.py --run-id RUN-20260407-022355-242D --types seminal,historical --limit 20

Output: artifacts/<run_id>_eval_references.md
"""

from __future__ import annotations

import re
import sys
import json
import time
import sqlite3
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).parent.parent
DB_PATH       = _HERE / "db" / "pipeline.db"
ARTIFACTS_DIR = _HERE / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Rate limiting ────────────────────────────────────────────────────────────
_last_call: dict[str, float] = {}

def _rate_wait(api: str, delay: float = 1.0):
    now = time.time()
    last = _last_call.get(api, 0)
    if now - last < delay:
        time.sleep(delay - (now - last))
    _last_call[api] = time.time()


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SourceRecord:
    source_id: str
    title: str
    authors: list[str]
    year: int | None
    source_name: str   # openalex, web_search, openlibrary, etc.
    doi: str
    abstract: str
    source_type: str   # current, seminal, historical
    seminal_reason: str
    historical_reason: str
    active_link: str


@dataclass
class VerifyResult:
    source: SourceRecord
    status: str = "UNVERIFIED"
    confidence: float = 0.0
    api_title: str = ""
    api_year: int | None = None
    api_authors: list[str] = field(default_factory=list)
    api_abstract: str = ""
    api_source: str = ""         # which API found it
    api_url: str = ""
    api_cited_by: int = 0
    title_match: float = 0.0    # 0–1 similarity
    year_match: bool = False
    author_match: bool = False
    claim_check: str = "not_checked"  # consistent / inconsistent / not_checked
    claim_detail: str = ""
    notes: str = ""


# ─── Database reader ──────────────────────────────────────────────────────────

def load_sources(run_id: str, types: list[str] | None = None,
                 db_path: Path = DB_PATH) -> list[SourceRecord]:
    """Load sources from the pipeline database for a given run."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM sources WHERE run_id = ?"
    params: list = [run_id]

    if types:
        placeholders = ",".join("?" * len(types))
        query += f" AND type IN ({placeholders})"
        params.extend(types)

    query += " ORDER BY type, year"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    sources = []
    for r in rows:
        authors_raw = r["authors"] or "[]"
        try:
            authors = json.loads(authors_raw)
        except (json.JSONDecodeError, TypeError):
            authors = [authors_raw] if authors_raw else []

        sources.append(SourceRecord(
            source_id=r["source_id"],
            title=r["title"] or "",
            authors=authors if isinstance(authors, list) else [str(authors)],
            year=r["year"],
            source_name=r["source_name"] or "",
            doi=r["doi"] or "",
            abstract=r["abstract"] or "",
            source_type=r["type"] or "",
            seminal_reason=r["seminal_reason"] or "",
            historical_reason=r["historical_reason"] or "",
            active_link=r["active_link"] or "",
        ))

    return sources


# ─── API verification ─────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _author_match(source_authors: list[str], api_authors: list[str]) -> bool:
    """Check if any author surname appears in both lists."""
    def surnames(authors):
        out = set()
        for a in authors:
            parts = a.strip().split()
            if parts:
                out.add(parts[-1].lower())
        return out

    s1 = surnames(source_authors)
    s2 = surnames(api_authors)
    return bool(s1 & s2)


def search_semantic_scholar(title: str, author: str = "") -> dict | None:
    """Search Semantic Scholar for a paper by title."""
    _rate_wait("semantic_scholar", 1.2)
    query = title[:200]
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": 3,
                    "fields": "title,authors,year,abstract,externalIds,url,citationCount"},
            headers={"User-Agent": "SEEKEREvaluator/1.0"},
            timeout=15,
        )
        if resp.status_code == 429:
            time.sleep(5)
            return None
        if resp.status_code != 200:
            return None
        data = resp.json()
        papers = data.get("data", [])
        return papers[0] if papers else None
    except Exception as e:
        logger.warning(f"[S2] Error: {e}")
        return None


def search_openalex(title: str) -> dict | None:
    """Search OpenAlex for a paper by title."""
    _rate_wait("openalex", 0.3)
    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params={"search": title[:200], "per-page": 3,
                    "sort": "relevance_score:desc",
                    "mailto": "pipeline@research.local"},
            headers={"User-Agent": "SEEKEREvaluator/1.0"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
        return results[0] if results else None
    except Exception as e:
        logger.warning(f"[OpenAlex] Error: {e}")
        return None


def search_claude_web(title: str, author: str = "", year: int | None = None) -> dict | None:
    """
    Use Anthropic API with web_search tool to verify a reference.
    Returns a dict with title, authors, year, abstract, url if found.
    Falls back gracefully if ANTHROPIC_API_KEY is not set.
    """
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    _rate_wait("claude_web", 2.0)  # conservative rate limit

    query_parts = [f'"{title[:100]}"']
    if author:
        query_parts.append(author.split()[-1])  # surname
    if year:
        query_parts.append(str(year))
    search_query = " ".join(query_parts)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{
                "role": "user",
                "content": (
                    f"Verify this academic reference exists. Search for it and tell me:\n"
                    f"Title: {title}\n"
                    f"Author: {author}\n"
                    f"Year: {year}\n\n"
                    f"Respond ONLY with a JSON object, no other text:\n"
                    f'{{"found": true/false, "verified_title": "...", '
                    f'"verified_authors": ["..."], "verified_year": NNNN, '
                    f'"abstract_snippet": "first 200 chars of abstract", '
                    f'"url": "...", "source": "where you found it"}}\n'
                    f"If you cannot find this exact work, set found=false."
                )
            }]
        )

        # Extract text from response
        text_parts = [b.text for b in response.content if hasattr(b, "text") and b.text]
        full_text = "\n".join(text_parts)

        if not full_text:
            return None

        # Parse JSON from response
        match = re.search(r'\{[^{}]*"found"[^{}]*\}', full_text, re.DOTALL)
        if not match:
            return None

        data = json.loads(match.group())
        if not data.get("found"):
            return None

        return {
            "title":    data.get("verified_title", ""),
            "authors":  data.get("verified_authors", []),
            "year":     data.get("verified_year"),
            "abstract": data.get("abstract_snippet", ""),
            "url":      data.get("url", ""),
            "source":   data.get("source", "web"),
        }
    except ImportError:
        logger.info("[Claude] anthropic library not installed — skipping web search verification")
        return None
    except Exception as e:
        logger.warning(f"[Claude] Web search error: {e}")
        return None


# ─── Verification logic ───────────────────────────────────────────────────────

import requests  # placed here so the module is still parseable if requests missing

def verify_source(source: SourceRecord) -> VerifyResult:
    """Verify a single source against live APIs."""
    result = VerifyResult(source=source)

    # Skip very old works and known books — APIs won't have them
    is_pre_modern = source.year is not None and source.year < 1900
    is_book_source = source.source_name in ("openlibrary", "googlebooks", "google_books")

    # 1. Try Semantic Scholar
    s2 = search_semantic_scholar(source.title, source.authors[0] if source.authors else "")
    if s2:
        api_title   = s2.get("title", "")
        api_authors = [a.get("name", "") for a in s2.get("authors", [])]
        api_year    = s2.get("year")
        api_abstract = s2.get("abstract", "") or ""
        api_url     = s2.get("url", "")
        api_cited   = s2.get("citationCount", 0) or 0

        result.api_title    = api_title
        result.api_year     = api_year
        result.api_authors  = api_authors
        result.api_abstract = api_abstract[:500]
        result.api_source   = "semantic_scholar"
        result.api_url      = api_url
        result.api_cited_by = api_cited

        result.title_match  = _title_similarity(source.title, api_title)
        result.year_match   = (api_year is not None and source.year is not None
                               and abs(api_year - source.year) <= 2)
        result.author_match = _author_match(source.authors, api_authors)

        # Grade
        if result.title_match >= 0.80 and result.author_match:
            result.status = "VERIFIED"
            result.confidence = min(1.0, result.title_match * 0.6 + 0.2 * result.year_match + 0.2)
        elif result.title_match >= 0.60:
            result.status = "PLAUSIBLE"
            result.confidence = result.title_match * 0.5 + 0.1 * result.author_match
        else:
            # Title didn't match well — try OpenAlex before giving up
            pass

    # 2. If S2 didn't verify, try OpenAlex
    if result.status not in ("VERIFIED", "PLAUSIBLE"):
        oa = search_openalex(source.title)
        if oa:
            api_title   = oa.get("display_name", "")
            api_authors = [a.get("author", {}).get("display_name", "")
                           for a in oa.get("authorships", [])]
            api_year    = oa.get("publication_year")
            api_cited   = oa.get("cited_by_count", 0)
            api_url     = oa.get("doi", "") or oa.get("id", "")

            # Build abstract from inverted index
            api_abstract = ""
            if oa.get("abstract_inverted_index"):
                inv = oa["abstract_inverted_index"]
                words = {}
                for word, positions in inv.items():
                    for pos in positions:
                        words[pos] = word
                api_abstract = " ".join(words[i] for i in sorted(words.keys()))[:500]

            result.api_title    = api_title
            result.api_year     = api_year
            result.api_authors  = api_authors[:5]
            result.api_abstract = api_abstract
            result.api_source   = "openalex"
            result.api_url      = api_url
            result.api_cited_by = api_cited

            result.title_match  = _title_similarity(source.title, api_title)
            result.year_match   = (api_year is not None and source.year is not None
                                   and abs(api_year - source.year) <= 2)
            result.author_match = _author_match(source.authors, api_authors)

            if result.title_match >= 0.80 and result.author_match:
                result.status = "VERIFIED"
                result.confidence = min(1.0, result.title_match * 0.6 + 0.2 * result.year_match + 0.2)
            elif result.title_match >= 0.60:
                result.status = "PLAUSIBLE"
                result.confidence = result.title_match * 0.5 + 0.1 * result.author_match

    # 3. If still not found, try Claude web search (broadest coverage — books, old works)
    if result.status not in ("VERIFIED", "PLAUSIBLE"):
        claude = search_claude_web(
            source.title,
            source.authors[0] if source.authors else "",
            source.year,
        )
        if claude:
            api_title   = claude.get("title", "")
            api_authors = claude.get("authors", [])
            api_year    = claude.get("year")
            api_abstract = claude.get("abstract", "") or ""
            api_url     = claude.get("url", "")

            result.api_title    = api_title
            result.api_year     = api_year
            result.api_authors  = api_authors[:5] if isinstance(api_authors, list) else []
            result.api_abstract = api_abstract[:500]
            result.api_source   = "claude_web"
            result.api_url      = api_url

            result.title_match  = _title_similarity(source.title, api_title) if api_title else 0.0
            result.year_match   = (api_year is not None and source.year is not None
                                   and abs(api_year - source.year) <= 2)
            result.author_match = _author_match(source.authors, api_authors) if api_authors else False

            if result.title_match >= 0.70 and (result.author_match or result.year_match):
                result.status = "VERIFIED"
                result.confidence = min(1.0, result.title_match * 0.5 + 0.2 * result.year_match + 0.2 * result.author_match + 0.1)
            elif result.title_match >= 0.50 or result.author_match:
                result.status = "PLAUSIBLE"
                result.confidence = result.title_match * 0.4 + 0.1 * result.author_match

    # 4. If still not found after all three sources
    if result.status == "UNVERIFIED":
        if is_pre_modern:
            result.notes = "Pre-1900 work — not expected in academic APIs"
            result.confidence = 0.5  # neutral — can't verify but expected
        elif is_book_source:
            result.notes = "Book source — limited API coverage for monographs"
            result.confidence = 0.4
        else:
            result.notes = "Not found in Semantic Scholar, OpenAlex, or Claude web search"
            result.confidence = 0.0

    # 5. Claim consistency check (for seminal/historical sources with reasons)
    claim_text = source.seminal_reason or source.historical_reason
    # Use the best available abstract: DB-stored first, then API-retrieved
    best_abstract = source.abstract or result.api_abstract
    if claim_text and best_abstract and result.status in ("VERIFIED", "PLAUSIBLE"):
        result.claim_check, result.claim_detail = _check_claim_vs_abstract(
            claim_text, best_abstract, source.title
        )

    return result


def _check_claim_vs_abstract(claim: str, abstract: str, title: str) -> tuple[str, str]:
    """
    Basic consistency check: does the pipeline's claim about a work
    align with the work's actual abstract? Uses keyword overlap as a
    heuristic. Not perfect — but catches obvious misattributions.
    """
    claim_words = set(_normalize(claim).split())
    abstract_words = set(_normalize(abstract).split())
    title_words = set(_normalize(title).split())

    # Remove common stop words
    stops = {"the", "a", "an", "of", "in", "to", "and", "is", "that", "for",
             "this", "are", "was", "with", "on", "by", "from", "as", "at",
             "it", "be", "or", "not", "has", "have", "had", "but", "its",
             "which", "we", "can", "their", "been", "were", "will", "more",
             "than", "also", "about", "other", "into", "some", "these", "those",
             "would", "may", "between", "through", "most", "how", "what"}

    claim_content  = claim_words - stops
    abstract_content = abstract_words - stops
    all_ref_words = abstract_content | title_words - stops

    if not claim_content or not all_ref_words:
        return "not_checked", "Insufficient text for comparison"

    overlap = claim_content & all_ref_words
    ratio = len(overlap) / len(claim_content) if claim_content else 0

    if ratio >= 0.25:
        return "consistent", f"{len(overlap)}/{len(claim_content)} claim terms found in abstract/title ({ratio:.0%})"
    elif ratio >= 0.10:
        return "weak", f"Only {len(overlap)}/{len(claim_content)} claim terms found ({ratio:.0%}) — claim may be loosely grounded"
    else:
        return "inconsistent", f"Only {len(overlap)}/{len(claim_content)} terms overlap ({ratio:.0%}) — claim may not match this work"


# ─── Report generation ────────────────────────────────────────────────────────

def generate_report(run_id: str, results: list[VerifyResult], problem: str) -> str:
    """Generate the evaluation Markdown report."""

    # Stats
    total = len(results)
    verified     = [r for r in results if r.status == "VERIFIED"]
    plausible    = [r for r in results if r.status == "PLAUSIBLE"]
    unverified   = [r for r in results if r.status == "UNVERIFIED"]
    suspicious   = [r for r in results if r.status == "SUSPICIOUS"]

    # Claim checks
    consistent   = [r for r in results if r.claim_check == "consistent"]
    weak_claims  = [r for r in results if r.claim_check == "weak"]
    inconsistent = [r for r in results if r.claim_check == "inconsistent"]

    # By source type
    by_type = {}
    for r in results:
        t = r.source.source_type
        by_type.setdefault(t, []).append(r)

    lines = [
        f"# Reference Hallucination Evaluation Report",
        f"**Run ID:** {run_id}",
        f"**Problem:** {problem}",
        f"**Evaluated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Sources checked:** {total}",
        f"**APIs used:** Semantic Scholar, OpenAlex, Claude web search",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Status | Count | % |",
        f"|--------|-------|---|",
        f"| ✅ VERIFIED | {len(verified)} | {len(verified)/total*100:.0f}% |" if total else "",
        f"| 🟡 PLAUSIBLE | {len(plausible)} | {len(plausible)/total*100:.0f}% |" if total else "",
        f"| ⚪ UNVERIFIED | {len(unverified)} | {len(unverified)/total*100:.0f}% |" if total else "",
        f"| 🔴 SUSPICIOUS | {len(suspicious)} | {len(suspicious)/total*100:.0f}% |" if total else "",
        "",
        f"**Hallucination risk score:** {len(suspicious)}/{total} ({len(suspicious)/total*100:.1f}%)" if total else "",
        "",
    ]

    if consistent or weak_claims or inconsistent:
        lines += [
            "### Claim Consistency (for sources with seminal/historical reasons)",
            "",
            f"| Check | Count |",
            f"|-------|-------|",
            f"| Consistent | {len(consistent)} |",
            f"| Weak match | {len(weak_claims)} |",
            f"| Inconsistent | {len(inconsistent)} |",
            "",
        ]

    # Per source type breakdown
    for stype in ["seminal", "historical", "current"]:
        type_results = by_type.get(stype, [])
        if not type_results:
            continue

        type_verified = sum(1 for r in type_results if r.status == "VERIFIED")
        type_plausible = sum(1 for r in type_results if r.status == "PLAUSIBLE")

        lines += [
            f"---",
            f"",
            f"## {stype.capitalize()} Sources ({len(type_results)} total — {type_verified} verified, {type_plausible} plausible)",
            "",
        ]

        for r in type_results:
            s = r.source
            icon = {"VERIFIED": "✅", "PLAUSIBLE": "🟡",
                    "UNVERIFIED": "⚪", "SUSPICIOUS": "🔴"}.get(r.status, "❓")
            auth_str = ", ".join(s.authors[:2]) or "?"

            lines.append(f"### {icon} [{s.year or '?'}] {s.title[:100]}")
            lines.append(f"")
            lines.append(f"- **Pipeline:** {auth_str} | {s.source_name} | doi: {s.doi[:50] if s.doi else 'none'}")
            lines.append(f"- **Status:** {r.status} (confidence: {r.confidence:.0%})")

            if r.api_title:
                api_auth = ", ".join(r.api_authors[:2]) or "?"
                lines.append(f"- **API match ({r.api_source}):** {r.api_title[:100]}")
                lines.append(f"  - Authors: {api_auth} | Year: {r.api_year} | Cited: {r.api_cited_by}")
                lines.append(f"  - Title similarity: {r.title_match:.0%} | Year match: {'✓' if r.year_match else '✗'} | Author match: {'✓' if r.author_match else '✗'}")

            if s.seminal_reason:
                lines.append(f"- **Pipeline claim:** {s.seminal_reason[:200]}")
            elif s.historical_reason:
                lines.append(f"- **Pipeline claim:** {s.historical_reason[:200]}")

            if r.claim_check != "not_checked":
                claim_icon = {"consistent": "✅", "weak": "⚠️", "inconsistent": "❌"}.get(r.claim_check, "❓")
                lines.append(f"- **Claim check:** {claim_icon} {r.claim_check} — {r.claim_detail}")

            if r.notes:
                lines.append(f"- **Notes:** {r.notes}")

            lines.append("")

    # Flagged items section
    flagged = [r for r in results if r.claim_check == "inconsistent" or r.status == "SUSPICIOUS"]
    if flagged:
        lines += [
            "---",
            "",
            "## ⚠️ Items Requiring Manual Review",
            "",
        ]
        for r in flagged:
            lines.append(f"- **[{r.source.year or '?'}] {r.source.title[:80]}** — {r.status}, claim: {r.claim_check}")
            if r.claim_detail:
                lines.append(f"  {r.claim_detail}")
            lines.append("")

    # Footer
    lines += [
        "---",
        "",
        f"*Report generated by SEEKER Evaluator v1.0 — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        f"*APIs queried: Semantic Scholar, OpenAlex, Claude web search (Anthropic API)*",
        f"*Claim consistency uses keyword overlap heuristic — manual review recommended for flagged items.*",
    ]

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SEEKER Reference Hallucination Evaluator")
    parser.add_argument("--run-id", required=True, help="Run ID to evaluate")
    parser.add_argument("--types", default="seminal,historical",
                        help="Source types to check (comma-separated: seminal,historical,current)")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to pipeline.db")
    parser.add_argument("--limit", type=int, default=0, help="Max sources to check (0=all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    types = [t.strip() for t in args.types.split(",")]
    print(f"\n{'='*60}")
    print(f"  SEEKER Reference Hallucination Evaluator")
    print(f"{'='*60}")
    print(f"  Run ID: {args.run_id}")
    print(f"  Types:  {', '.join(types)}")
    print(f"  DB:     {db_path}")
    print(f"{'='*60}\n")

    # Load sources
    sources = load_sources(args.run_id, types, db_path=db_path)
    if not sources:
        print("No sources found for this run and type filter.")
        sys.exit(1)

    if args.limit > 0:
        sources = sources[:args.limit]

    print(f"  Loaded {len(sources)} sources to verify\n")

    # Get problem from runs table
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    run_row = conn.execute("SELECT problem FROM runs WHERE run_id = ?", (args.run_id,)).fetchone()
    problem = run_row["problem"] if run_row else args.run_id
    conn.close()

    # Verify each source
    results: list[VerifyResult] = []
    for i, source in enumerate(sources, 1):
        auth = source.authors[0] if source.authors else "?"
        print(f"  [{i}/{len(sources)}] [{source.source_type}] {source.title[:60]}...")

        try:
            result = verify_source(source)
            results.append(result)
            icon = {"VERIFIED": "✅", "PLAUSIBLE": "🟡",
                    "UNVERIFIED": "⚪", "SUSPICIOUS": "🔴"}.get(result.status, "?")
            print(f"           {icon} {result.status} "
                  f"(title: {result.title_match:.0%}, "
                  f"claim: {result.claim_check})")
        except Exception as e:
            logger.warning(f"    Error verifying: {e}")
            results.append(VerifyResult(source=source, notes=f"Error: {e}"))

    # Generate report
    report = generate_report(args.run_id, results, problem)
    out_path = ARTIFACTS_DIR / f"{args.run_id}_eval_references.md"
    out_path.write_text(report, encoding="utf-8")

    # Print summary
    verified   = sum(1 for r in results if r.status == "VERIFIED")
    plausible  = sum(1 for r in results if r.status == "PLAUSIBLE")
    unverified = sum(1 for r in results if r.status == "UNVERIFIED")
    suspicious = sum(1 for r in results if r.status == "SUSPICIOUS")

    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  ✅ Verified:   {verified}")
    print(f"  🟡 Plausible:  {plausible}")
    print(f"  ⚪ Unverified: {unverified}")
    print(f"  🔴 Suspicious: {suspicious}")
    print(f"  Report: {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
