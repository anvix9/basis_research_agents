"""
SEEKER Evaluator — Claim Verification
--------------------------------------
Cross-checks the pipeline's claims about each source (seminal_reason,
historical_reason) against what the source actually says.

Uses three verification strategies:
  1. DB-internal: compare claim keywords vs stored abstract
  2. Consensus MCP: search for the work via Consensus and compare
  3. Claude LLM: ask Claude to evaluate claim accuracy given the abstract

Produces a Markdown report grading each claim as:
  CONFIRMED    — claim is substantiated by the source's actual content
  PARTIALLY    — claim is broadly correct but overstates, misstates a detail,
                 or conflates with another work
  UNSUBSTANTIATED — cannot verify (no abstract, old work, insufficient data)
  INACCURATE   — claim contradicts what the source actually says
  METADATA_ERROR — title, author, or year is wrong for the claimed content

Usage:
    python3 tools/eval_claims.py --run-id RUN-20260407-022355-242D
    python3 tools/eval_claims.py --run-id RUN-20260407-022355-242D --types seminal --limit 10

Output: artifacts/<run_id>_eval_claims.md
"""

from __future__ import annotations

import re
import os
import sys
import json
import time
import sqlite3
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_HERE         = Path(__file__).parent.parent
DB_PATH       = _HERE / "db" / "pipeline.db"
ARTIFACTS_DIR = _HERE / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SourceRecord:
    source_id: str
    title: str
    authors: list[str]
    year: int | None
    source_name: str
    doi: str
    abstract: str
    source_type: str
    seminal_reason: str
    historical_reason: str
    active_link: str


@dataclass
class ClaimResult:
    source: SourceRecord
    claim_text: str = ""
    verdict: str = "PENDING"       # CONFIRMED / PARTIALLY / UNSUBSTANTIATED / INACCURATE / METADATA_ERROR
    confidence: float = 0.0
    # Verification details
    method: str = ""               # db_internal / consensus / claude_llm
    evidence: str = ""             # what the verifier found
    issues: list[str] = field(default_factory=list)
    # Metadata accuracy
    title_accurate: str = "unknown"
    author_accurate: str = "unknown"
    year_accurate: str = "unknown"
    correct_title: str = ""        # if metadata is wrong, what's the real title?
    notes: str = ""


# ─── DB reader ────────────────────────────────────────────────────────────────

def load_sources(db_path: Path, run_id: str,
                 types: list[str] | None = None) -> list[SourceRecord]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM sources WHERE run_id = ?"
    params: list = [run_id]
    if types:
        placeholders = ",".join("?" * len(types))
        query += f" AND type IN ({placeholders})"
        params.extend(types)

    # Only sources with claims
    query += " AND (seminal_reason IS NOT NULL AND seminal_reason != '' "
    query += "  OR historical_reason IS NOT NULL AND historical_reason != '')"
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


def get_problem(db_path: Path, run_id: str) -> str:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT problem FROM runs WHERE run_id=?", (run_id,)).fetchone()
    conn.close()
    return row["problem"] if row else run_id


# ─── Verification: Claude LLM ─────────────────────────────────────────────────

CLAIM_CHECK_SYSTEM = """You are an academic fact-checker evaluating whether a research pipeline's claim about a scholarly work is accurate.

You will receive:
- The TITLE, AUTHOR, and YEAR the pipeline attributes to the work
- The CLAIM the pipeline makes about what this work established or contributed
- An ABSTRACT or description of the work (may be from the pipeline itself — treat with skepticism)
- Optionally, SEARCH RESULTS from web search to help you verify

Your job:
1. Verify the METADATA: Is the title correct? Is it attributed to the right author? Is the year right?
   Common errors: using a chapter title instead of the book title, wrong publication year, wrong author for a co-authored work.

2. Verify the CLAIM: Does this work actually do what the pipeline claims? Is the claim accurate, overstated, or wrong?
   Common errors: attributing an idea to the wrong work, overstating what a work "established" vs what it merely discussed, conflating two different works.

Respond ONLY with a JSON object:
{
  "verdict": "CONFIRMED|PARTIALLY|INACCURATE|METADATA_ERROR|UNSUBSTANTIATED",
  "confidence": 0.0-1.0,
  "title_accurate": true/false,
  "correct_title": "actual title if different, empty string if correct",
  "author_accurate": true/false,
  "year_accurate": true/false,
  "claim_assessment": "1-3 sentence explanation of whether the claim is accurate",
  "issues": ["list of specific problems found, empty if none"]
}"""


def verify_claim_with_claude(source: SourceRecord, claim: str) -> ClaimResult | None:
    """Use Claude with web search to verify a claim about a source."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    try:
        import anthropic
    except ImportError:
        logger.info("[Claude] anthropic library not installed")
        return None

    time.sleep(1.5)  # rate limit

    author_str = ", ".join(source.authors[:3]) if source.authors else "unknown"

    prompt = (
        f"REFERENCE TO VERIFY:\n"
        f"  Title: {source.title}\n"
        f"  Author: {author_str}\n"
        f"  Year: {source.year}\n\n"
        f"PIPELINE'S CLAIM ABOUT THIS WORK:\n"
        f"  \"{claim}\"\n\n"
        f"ABSTRACT STORED IN PIPELINE (may be LLM-generated — verify independently):\n"
        f"  \"{source.abstract[:500]}\"\n\n"
        f"Search for this work and verify: (1) Is the metadata correct? "
        f"(2) Does this work actually do what the claim says?\n"
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=CLAIM_CHECK_SYSTEM,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )

        text_parts = [b.text for b in response.content if hasattr(b, "text") and b.text]
        full_text = "\n".join(text_parts)

        if not full_text:
            return None

        # Extract JSON
        match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', full_text, re.DOTALL)
        if not match:
            # Try a broader match for nested JSON
            match = re.search(r'\{.*?"verdict".*?\}', full_text, re.DOTALL)
        if not match:
            logger.warning(f"[Claude] Could not parse JSON from response for: {source.title[:50]}")
            return None

        data = json.loads(match.group())

        result = ClaimResult(source=source, claim_text=claim)
        result.verdict    = data.get("verdict", "UNSUBSTANTIATED")
        result.confidence = data.get("confidence", 0.5)
        result.method     = "claude_llm"
        result.evidence   = data.get("claim_assessment", "")
        result.issues     = data.get("issues", [])

        result.title_accurate  = "yes" if data.get("title_accurate", True) else "no"
        result.author_accurate = "yes" if data.get("author_accurate", True) else "no"
        result.year_accurate   = "yes" if data.get("year_accurate", True) else "no"
        result.correct_title   = data.get("correct_title", "")

        return result

    except Exception as e:
        logger.warning(f"[Claude] Error verifying claim for {source.title[:50]}: {e}")
        return None


# ─── Verification: DB-internal keyword check ──────────────────────────────────

def verify_claim_internal(source: SourceRecord, claim: str) -> ClaimResult:
    """Basic keyword overlap check between claim and stored abstract."""
    result = ClaimResult(source=source, claim_text=claim, method="db_internal")

    if not source.abstract or len(source.abstract) < 50:
        result.verdict = "UNSUBSTANTIATED"
        result.evidence = "Abstract too short or missing for internal verification"
        result.confidence = 0.2
        return result

    # Normalize and extract content words
    stops = {"the","a","an","of","in","to","and","is","that","for","this","are",
             "was","with","on","by","from","as","at","it","be","or","not","has",
             "have","had","but","its","which","we","can","their","been","were",
             "will","more","than","also","about","other","into","some","these",
             "would","may","between","through","most","how","what","do","does"}

    def content_words(text):
        words = re.sub(r"[^\w\s]", " ", text.lower()).split()
        return set(w for w in words if w not in stops and len(w) > 2)

    claim_words   = content_words(claim)
    abstract_words = content_words(source.abstract)
    title_words   = content_words(source.title)
    all_ref_words = abstract_words | title_words

    if not claim_words:
        result.verdict = "UNSUBSTANTIATED"
        result.evidence = "Claim too short to analyze"
        result.confidence = 0.1
        return result

    overlap = claim_words & all_ref_words
    ratio = len(overlap) / len(claim_words)

    if ratio >= 0.30:
        result.verdict = "CONFIRMED"
        result.confidence = min(0.6, ratio)  # cap at 0.6 — keyword check is weak
        result.evidence = (f"{len(overlap)}/{len(claim_words)} claim terms found in "
                          f"abstract/title ({ratio:.0%})")
    elif ratio >= 0.15:
        result.verdict = "PARTIALLY"
        result.confidence = ratio * 0.5
        result.evidence = (f"Only {len(overlap)}/{len(claim_words)} terms overlap "
                          f"({ratio:.0%}) — claim may overstate")
    else:
        result.verdict = "UNSUBSTANTIATED"
        result.confidence = 0.2
        result.evidence = (f"Only {len(overlap)}/{len(claim_words)} terms overlap "
                          f"({ratio:.0%}) — claim may not match this work")

    return result


# ─── Main verification orchestrator ───────────────────────────────────────────

def verify_claim(source: SourceRecord) -> ClaimResult:
    """Verify a source's claim using the best available method."""
    claim = source.seminal_reason or source.historical_reason
    if not claim:
        return ClaimResult(source=source, verdict="UNSUBSTANTIATED",
                          evidence="No claim to verify", method="none")

    # Try Claude first (most accurate)
    claude_result = verify_claim_with_claude(source, claim)
    if claude_result:
        return claude_result

    # Fall back to DB-internal keyword check
    return verify_claim_internal(source, claim)


# ─── Report generation ────────────────────────────────────────────────────────

def generate_report(run_id: str, results: list[ClaimResult], problem: str) -> str:
    total = len(results)
    confirmed   = [r for r in results if r.verdict == "CONFIRMED"]
    partially   = [r for r in results if r.verdict == "PARTIALLY"]
    inaccurate  = [r for r in results if r.verdict == "INACCURATE"]
    metadata_err = [r for r in results if r.verdict == "METADATA_ERROR"]
    unsub       = [r for r in results if r.verdict == "UNSUBSTANTIATED"]

    # Metadata issues
    title_issues  = [r for r in results if r.title_accurate == "no"]
    author_issues = [r for r in results if r.author_accurate == "no"]
    year_issues   = [r for r in results if r.year_accurate == "no"]

    lines = [
        f"# Claim Verification Report",
        f"**Run ID:** {run_id}",
        f"**Problem:** {problem}",
        f"**Evaluated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Claims checked:** {total}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Verdict | Count | % |",
        f"|---------|-------|---|",
    ]
    if total > 0:
        for label, items, icon in [
            ("CONFIRMED", confirmed, "✅"),
            ("PARTIALLY", partially, "🟡"),
            ("METADATA_ERROR", metadata_err, "🔶"),
            ("INACCURATE", inaccurate, "❌"),
            ("UNSUBSTANTIATED", unsub, "⚪"),
        ]:
            lines.append(f"| {icon} {label} | {len(items)} | {len(items)/total*100:.0f}% |")

    lines.append("")

    if title_issues or author_issues or year_issues:
        lines += [
            "### Metadata Accuracy",
            "",
            f"| Issue | Count |",
            f"|-------|-------|",
            f"| Wrong title | {len(title_issues)} |",
            f"| Wrong author | {len(author_issues)} |",
            f"| Wrong year | {len(year_issues)} |",
            "",
        ]

    accuracy = (len(confirmed) + len(partially)) / total * 100 if total else 0
    lines.append(f"**Overall claim accuracy:** {accuracy:.0f}% (confirmed + partially correct)")
    lines.append("")

    # Group by source type
    by_type = {}
    for r in results:
        by_type.setdefault(r.source.source_type, []).append(r)

    for stype in ["seminal", "historical", "current"]:
        type_results = by_type.get(stype, [])
        if not type_results:
            continue

        lines += [
            "---",
            "",
            f"## {stype.capitalize()} Sources — Claim Verification ({len(type_results)} claims)",
            "",
        ]

        for r in type_results:
            s = r.source
            icon = {"CONFIRMED": "✅", "PARTIALLY": "🟡", "METADATA_ERROR": "🔶",
                    "INACCURATE": "❌", "UNSUBSTANTIATED": "⚪"}.get(r.verdict, "❓")
            auth_str = ", ".join(s.authors[:2]) or "?"

            lines.append(f"### {icon} [{s.year or '?'}] {s.title[:100]}")
            lines.append("")
            lines.append(f"- **Author:** {auth_str} | **Source:** {s.source_name}")
            lines.append(f"- **Claim:** \"{r.claim_text[:200]}\"")
            lines.append(f"- **Verdict:** {r.verdict} (confidence: {r.confidence:.0%}) — via {r.method}")

            if r.evidence:
                lines.append(f"- **Assessment:** {r.evidence}")

            # Metadata issues
            meta_flags = []
            if r.title_accurate == "no":
                meta_flags.append(f"Title incorrect → actual: \"{r.correct_title}\"" if r.correct_title
                                  else "Title may be incorrect")
            if r.author_accurate == "no":
                meta_flags.append("Author attribution may be wrong")
            if r.year_accurate == "no":
                meta_flags.append("Year may be incorrect")
            if meta_flags:
                lines.append(f"- **Metadata issues:** {'; '.join(meta_flags)}")

            if r.issues:
                lines.append(f"- **Issues found:**")
                for issue in r.issues:
                    lines.append(f"  - {issue}")

            if r.notes:
                lines.append(f"- **Notes:** {r.notes}")

            lines.append("")

    # Flagged items
    flagged = [r for r in results if r.verdict in ("INACCURATE", "METADATA_ERROR")]
    if flagged:
        lines += [
            "---",
            "",
            "## ⚠️  Items Requiring Correction",
            "",
        ]
        for r in flagged:
            lines.append(f"- **[{r.source.year or '?'}] {r.source.title[:80]}** — {r.verdict}")
            if r.evidence:
                lines.append(f"  {r.evidence}")
            if r.correct_title:
                lines.append(f"  Correct title: \"{r.correct_title}\"")
            for issue in r.issues:
                lines.append(f"  - {issue}")
            lines.append("")

    lines += [
        "---",
        "",
        f"*Report generated by SEEKER Claim Evaluator v1.0 — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        f"*Primary method: Claude Haiku + web search | Fallback: DB-internal keyword overlap*",
    ]

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SEEKER Claim Verification Evaluator")
    parser.add_argument("--run-id", required=True, help="Run ID to evaluate")
    parser.add_argument("--types", default="seminal,historical",
                        help="Source types to check (comma-separated)")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to pipeline.db")
    parser.add_argument("--limit", type=int, default=0, help="Max claims to check (0=all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    types = [t.strip() for t in args.types.split(",")]

    print(f"\n{'='*60}")
    print(f"  SEEKER Claim Verification Evaluator")
    print(f"{'='*60}")
    print(f"  Run ID: {args.run_id}")
    print(f"  Types:  {', '.join(types)}")
    print(f"  DB:     {db_path}")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    print(f"  Claude: {'available' if has_key else 'not available (ANTHROPIC_API_KEY not set — using DB-internal only)'}")
    print(f"{'='*60}\n")

    sources = load_sources(db_path, args.run_id, types)
    if not sources:
        print("No sources with claims found for this run.")
        sys.exit(1)

    if args.limit > 0:
        sources = sources[:args.limit]

    print(f"  Loaded {len(sources)} sources with claims to verify\n")

    problem = get_problem(db_path, args.run_id)

    results: list[ClaimResult] = []
    for i, source in enumerate(sources, 1):
        claim = source.seminal_reason or source.historical_reason
        auth = source.authors[0] if source.authors else "?"
        print(f"  [{i}/{len(sources)}] [{source.source_type}] {source.title[:55]}...")

        try:
            result = verify_claim(source)
            results.append(result)
            icon = {"CONFIRMED": "✅", "PARTIALLY": "🟡", "METADATA_ERROR": "🔶",
                    "INACCURATE": "❌", "UNSUBSTANTIATED": "⚪"}.get(result.verdict, "?")
            detail = ""
            if result.title_accurate == "no":
                detail = f" | title wrong → {result.correct_title[:40]}"
            print(f"           {icon} {result.verdict} ({result.method}){detail}")
        except Exception as e:
            logger.warning(f"    Error: {e}")
            results.append(ClaimResult(source=source, verdict="UNSUBSTANTIATED",
                                       notes=f"Error: {e}", method="error"))

    # Generate report
    report = generate_report(args.run_id, results, problem)
    out_path = ARTIFACTS_DIR / f"{args.run_id}_eval_claims.md"
    out_path.write_text(report, encoding="utf-8")

    # Summary
    confirmed  = sum(1 for r in results if r.verdict == "CONFIRMED")
    partially  = sum(1 for r in results if r.verdict == "PARTIALLY")
    inaccurate = sum(1 for r in results if r.verdict == "INACCURATE")
    meta_err   = sum(1 for r in results if r.verdict == "METADATA_ERROR")
    unsub      = sum(1 for r in results if r.verdict == "UNSUBSTANTIATED")

    print(f"\n{'='*60}")
    print(f"  CLAIM VERIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"  ✅ Confirmed:       {confirmed}")
    print(f"  🟡 Partially:       {partially}")
    print(f"  🔶 Metadata error:  {meta_err}")
    print(f"  ❌ Inaccurate:      {inaccurate}")
    print(f"  ⚪ Unsubstantiated: {unsub}")
    total = len(results)
    accuracy = (confirmed + partially) / total * 100 if total else 0
    print(f"  Accuracy: {accuracy:.0f}%")
    print(f"  Report:   {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
