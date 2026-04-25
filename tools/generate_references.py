"""
Reference Section Generator
-----------------------------
Walks the argument tree, collects all source_ids referenced by any node,
queries the sources table for full metadata, and produces a formatted
reference section.

Supports: APA, Chicago, and simple list formats.

Usage:
    python3 tools/generate_references.py --run-id RUN-20260407-022355-242D
    python3 tools/generate_references.py --run-id RUN-20260407-022355-242D --format chicago
    python3 tools/generate_references.py --run-id RUN-20260407-022355-242D --format apa --output refs.md

Output: artifacts/<run_id>_references.md
"""

from __future__ import annotations

import sys
import json
import sqlite3
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_HERE         = Path(__file__).parent.parent
DB_PATH       = _HERE / "db" / "pipeline.db"
ARTIFACTS_DIR = _HERE / "artifacts"


def load_tree_sources(db_path: Path, run_id: str) -> list[str]:
    """Get all source_ids from the argument tree for a run."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT source_ids FROM argument_tree WHERE run_id = ?", (run_id,)
    ).fetchall()
    conn.close()

    all_ids = set()
    for row in rows:
        try:
            ids = json.loads(row["source_ids"])
            all_ids.update(i for i in ids if i)
        except (json.JSONDecodeError, TypeError):
            pass
    return sorted(all_ids)


def load_sources_by_ids(db_path: Path, source_ids: list[str]) -> list[dict]:
    """Load full source records by their IDs."""
    if not source_ids:
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(source_ids))
    rows = conn.execute(
        f"SELECT * FROM sources WHERE source_id IN ({placeholders})",
        source_ids
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_all_run_sources(db_path: Path, run_id: str) -> list[dict]:
    """Fallback: load all sources for a run if no tree exists."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM sources WHERE run_id = ? ORDER BY type, year",
        (run_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def format_apa(source: dict) -> str:
    """Format a source in APA 7th edition style."""
    authors_raw = source.get("authors", "[]")
    try:
        authors = json.loads(authors_raw) if isinstance(authors_raw, str) else authors_raw
    except (json.JSONDecodeError, TypeError):
        authors = []

    if not authors:
        author_str = "Unknown"
    elif len(authors) == 1:
        author_str = _apa_author(authors[0])
    elif len(authors) == 2:
        author_str = f"{_apa_author(authors[0])}, & {_apa_author(authors[1])}"
    elif len(authors) <= 20:
        parts = [_apa_author(a) for a in authors[:-1]]
        author_str = ", ".join(parts) + f", & {_apa_author(authors[-1])}"
    else:
        parts = [_apa_author(a) for a in authors[:19]]
        author_str = ", ".join(parts) + f", ... {_apa_author(authors[-1])}"

    year = source.get("year") or "n.d."
    title = source.get("title", "Untitled")
    doi = source.get("doi", "")
    doi_str = f" https://doi.org/{doi}" if doi else ""

    return f"{author_str} ({year}). {title}.{doi_str}"


def _apa_author(name: str) -> str:
    """Convert 'First Last' to 'Last, F.' for APA."""
    parts = name.strip().split()
    if len(parts) >= 2:
        surname = parts[-1]
        initials = " ".join(f"{p[0]}." for p in parts[:-1])
        return f"{surname}, {initials}"
    return name


def format_chicago(source: dict) -> str:
    """Format a source in Chicago style."""
    authors_raw = source.get("authors", "[]")
    try:
        authors = json.loads(authors_raw) if isinstance(authors_raw, str) else authors_raw
    except (json.JSONDecodeError, TypeError):
        authors = []

    if not authors:
        author_str = "Unknown"
    elif len(authors) == 1:
        parts = authors[0].strip().split()
        if len(parts) >= 2:
            author_str = f"{parts[-1]}, {' '.join(parts[:-1])}"
        else:
            author_str = authors[0]
    elif len(authors) <= 3:
        first = authors[0].strip().split()
        if len(first) >= 2:
            first_str = f"{first[-1]}, {' '.join(first[:-1])}"
        else:
            first_str = authors[0]
        rest = [a.strip() for a in authors[1:]]
        author_str = f"{first_str}, " + ", and ".join(rest)
    else:
        first = authors[0].strip().split()
        if len(first) >= 2:
            first_str = f"{first[-1]}, {' '.join(first[:-1])}"
        else:
            first_str = authors[0]
        author_str = f"{first_str}, et al."

    year = source.get("year") or "n.d."
    title = source.get("title", "Untitled")

    return f'{author_str}. {year}. "{title}."'


def format_simple(source: dict) -> str:
    """Simple format: Author (Year). Title."""
    authors_raw = source.get("authors", "[]")
    try:
        authors = json.loads(authors_raw) if isinstance(authors_raw, str) else authors_raw
    except (json.JSONDecodeError, TypeError):
        authors = []
    author_str = ", ".join(authors[:3])
    if len(authors) > 3:
        author_str += " et al."
    year = source.get("year") or "n.d."
    title = source.get("title", "Untitled")
    doi = source.get("doi", "")
    doi_str = f" doi:{doi}" if doi else ""
    return f"{author_str} ({year}). {title}.{doi_str}"


FORMATTERS = {
    "apa":     format_apa,
    "chicago": format_chicago,
    "simple":  format_simple,
}


def generate_reference_section(
    db_path: Path, run_id: str,
    fmt: str = "apa",
    tree_only: bool = True,
) -> str:
    """
    Generate a formatted reference section.

    If tree_only=True (default), only includes sources referenced in the
    argument tree. Otherwise includes all sources for the run.
    """
    formatter = FORMATTERS.get(fmt, format_apa)

    if tree_only:
        source_ids = load_tree_sources(db_path, run_id)
        if source_ids:
            sources = load_sources_by_ids(db_path, source_ids)
        else:
            # No tree — fall back to all sources
            sources = load_all_run_sources(db_path, run_id)
            tree_only = False
    else:
        sources = load_all_run_sources(db_path, run_id)

    if not sources:
        return "# References\n\nNo sources found.\n"

    # Sort by author surname then year
    def sort_key(s):
        authors_raw = s.get("authors", "[]")
        try:
            authors = json.loads(authors_raw) if isinstance(authors_raw, str) else authors_raw
        except:
            authors = []
        first_author = authors[0] if authors else "ZZZ"
        surname = first_author.strip().split()[-1] if first_author.strip() else "ZZZ"
        year = s.get("year") or 9999
        return (surname.lower(), year)

    sources.sort(key=sort_key)

    # Group by source type
    by_type = {}
    for s in sources:
        stype = s.get("type", "other")
        by_type.setdefault(stype, []).append(s)

    lines = [
        "# References",
        f"**Run ID:** {run_id}",
        f"**Format:** {fmt.upper()}",
        f"**Sources:** {len(sources)} "
        f"({'from argument tree' if tree_only else 'all run sources'})",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        "",
    ]

    for stype in ["seminal", "historical", "current", "other"]:
        type_sources = by_type.get(stype, [])
        if not type_sources:
            continue

        label = {
            "seminal": "Seminal Works",
            "historical": "Historical Sources",
            "current": "Contemporary Literature",
            "other": "Other Sources",
        }.get(stype, stype.capitalize())

        lines.append(f"## {label}")
        lines.append("")
        for s in type_sources:
            ref = formatter(s)
            lines.append(f"- {ref}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SEEKER Reference Section Generator")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--format", default="apa", choices=["apa", "chicago", "simple"])
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--output", default="", help="Output file (default: artifacts/<run_id>_references.md)")
    parser.add_argument("--all-sources", action="store_true",
                        help="Include all sources, not just tree-referenced")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    report = generate_reference_section(
        db_path, args.run_id,
        fmt=args.format,
        tree_only=not args.all_sources,
    )

    out_path = Path(args.output) if args.output else (ARTIFACTS_DIR / f"{args.run_id}_references.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    # Count
    ref_count = report.count("\n- ")
    print(f"\n  Reference section generated: {ref_count} references ({args.format.upper()})")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
