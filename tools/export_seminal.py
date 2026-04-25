#!/usr/bin/env python3
"""
export_seminal.py — ARANEA Seminal Works Exporter
==================================================
Reads all seminal papers from the pipeline SQLite database and exports them
in three formats for your blog:

  1. seminal_references.json   — full structured data (for programmatic use)
  2. seminal_references.csv    — spreadsheet-friendly, grouped by theme
  3. jekyll/_posts/            — one .md file per paper (optional, see --jekyll)

Usage:
  python3 export_seminal.py                    # JSON + CSV to ./exports/
  python3 export_seminal.py --jekyll           # also write Jekyll posts
  python3 export_seminal.py --db path/to/x.db # custom DB path
  python3 export_seminal.py --run RUN-XXXXXXXX # filter by run

Output: ./exports/
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DB = Path(__file__).parent / "db" / "pipeline.db"
EXPORTS_DIR = Path(__file__).parent / "exports"
JEKYLL_DIR  = Path(__file__).parent / "exports" / "jekyll" / "_posts"

# Maps raw theme_tag strings → human-readable category labels for the blog.
# Extend this as your theme bank grows.
THEME_LABEL = {
    "AI":                         "Artificial Intelligence",
    "artificial_intelligence":    "Artificial Intelligence",
    "machine_learning":           "Machine Learning",
    "deep_learning":              "Deep Learning",
    "NLP":                        "Natural Language Processing",
    "natural_language_processing":"Natural Language Processing",
    "philosophy":                 "Philosophy",
    "philosophy_of_mind":         "Philosophy of Mind",
    "philosophy_of_language":     "Philosophy of Language",
    "epistemology":               "Epistemology",
    "ethics":                     "Ethics & AI",
    "AI_ethics":                  "Ethics & AI",
    "cognitive_science":          "Cognitive Science",
    "neuroscience":               "Neuroscience",
    "political_science":          "Political Science",
    "social_science":             "Social Science",
    "complexity":                 "Complexity Science",
    "systems_theory":             "Systems Theory",
    "information_theory":         "Information Theory",
    "computation":                "Computation & Logic",
    "logic":                      "Computation & Logic",
    "mathematics":                "Mathematics",
    "biology":                    "Biology",
    "psychology":                 "Psychology",
    "sociology":                  "Sociology",
    "economics":                  "Economics",
    "history_of_science":         "History of Science",
    "semiotics":                  "Semiotics",
    "linguistics":                "Linguistics",
    "governance":                 "Governance & Policy",
    "power":                      "Power & Politics",
    "democracy":                  "Democracy",
    "methodology":                "Research Methodology",
}


def pretty_theme(raw: str) -> str:
    """Return a human-readable label for a raw theme tag."""
    return THEME_LABEL.get(raw, raw.replace("_", " ").title())


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_seminal(db_path: Path, run_id: str | None = None) -> list[dict]:
    """
    Return all seminal sources from the pipeline DB as a list of dicts.
    Optionally filter by run_id.
    """
    if not db_path.exists():
        sys.exit(f"[ERROR] Database not found: {db_path}\n"
                 f"        Make sure you run the pipeline at least once first, "
                 f"or pass --db with the correct path.")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if run_id:
        rows = conn.execute(
            "SELECT * FROM sources WHERE type='seminal' AND run_id=? ORDER BY year ASC",
            (run_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM sources WHERE type='seminal' ORDER BY year ASC"
        ).fetchall()

    conn.close()

    papers = []
    for row in rows:
        d = dict(row)

        # Parse JSON fields safely
        for field in ("authors", "theme_tags", "intersection_tags"):
            raw = d.get(field)
            if isinstance(raw, str):
                try:
                    d[field] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    d[field] = [raw] if raw else []
            elif raw is None:
                d[field] = []

        # Human-readable theme categories (deduplicated)
        d["categories"] = list(dict.fromkeys(
            pretty_theme(t) for t in d["theme_tags"]
        ))

        # Primary category (first theme tag)
        d["primary_category"] = d["categories"][0] if d["categories"] else "Uncategorized"

        # Formatted author string: "Last1, First1; Last2, First2"
        d["authors_str"] = "; ".join(d["authors"]) if d["authors"] else "Unknown"

        papers.append(d)

    return papers


# ─────────────────────────────────────────────────────────────────────────────
# Export: JSON
# ─────────────────────────────────────────────────────────────────────────────

def export_json(papers: list[dict], out_dir: Path) -> Path:
    """
    Write seminal_references.json with papers grouped by primary category.
    Structure:
      {
        "generated_at": "...",
        "total": N,
        "by_category": {
          "Artificial Intelligence": [...],
          ...
        },
        "all": [...]
      }
    """
    by_category = defaultdict(list)
    for p in papers:
        by_category[p["primary_category"]].append(_paper_for_export(p))

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total": len(papers),
        "note": "Seminal works excavated by ARANEA pipeline — Grounder agent",
        "by_category": {k: v for k, v in sorted(by_category.items())},
        "all": [_paper_for_export(p) for p in papers],
    }

    out_path = out_dir / "seminal_references.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


def _paper_for_export(p: dict) -> dict:
    """Return a clean export dict for one paper."""
    return {
        "id":                p.get("source_id", ""),
        "title":             p.get("title", ""),
        "authors":           p.get("authors", []),
        "authors_str":       p.get("authors_str", ""),
        "year":              p.get("year"),
        "source":            p.get("source_name", ""),
        "doi":               p.get("doi", ""),
        "link":              p.get("active_link", ""),
        "abstract":          p.get("abstract", ""),
        "seminal_reason":    p.get("seminal_reason", ""),
        "categories":        p.get("categories", []),
        "primary_category":  p.get("primary_category", ""),
        "theme_tags":        p.get("theme_tags", []),
        "intersection_tags": p.get("intersection_tags", []),
        "run_id":            p.get("run_id", ""),
        "date_collected":    p.get("date_collected", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Export: CSV
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(papers: list[dict], out_dir: Path) -> Path:
    """
    Write seminal_references.csv — one row per paper, sorted by category then year.
    """
    fieldnames = [
        "primary_category", "title", "authors_str", "year",
        "seminal_reason", "link", "doi", "source",
        "all_categories", "theme_tags", "run_id",
    ]

    out_path = out_dir / "seminal_references.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        sorted_papers = sorted(papers, key=lambda p: (p["primary_category"], p.get("year") or 0))
        for p in sorted_papers:
            writer.writerow({
                "primary_category": p["primary_category"],
                "title":            p.get("title", ""),
                "authors_str":      p.get("authors_str", ""),
                "year":             p.get("year", ""),
                "seminal_reason":   p.get("seminal_reason", ""),
                "link":             p.get("active_link", ""),
                "doi":              p.get("doi", ""),
                "source":           p.get("source_name", ""),
                "all_categories":   " | ".join(p.get("categories", [])),
                "theme_tags":       " | ".join(p.get("theme_tags", [])),
                "run_id":           p.get("run_id", ""),
            })

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Export: Jekyll Markdown posts
# ─────────────────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Convert a title to a URL-safe Jekyll slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text[:80]  # Jekyll filenames shouldn't be too long


def export_jekyll(papers: list[dict], jekyll_dir: Path) -> list[Path]:
    """
    Write one Jekyll post per seminal paper.
    Filename: YYYY-MM-DD-slug.md (using date_collected or today)
    """
    jekyll_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    written = []

    for p in papers:
        title     = p.get("title", "Untitled")
        slug      = slugify(title)
        post_date = (p.get("date_collected") or today)[:10]  # keep YYYY-MM-DD only
        filename  = f"{post_date}-seminal-{slug}.md"
        filepath  = jekyll_dir / filename

        # Build YAML front matter
        categories_yaml = "\n".join(f'  - "{c}"' for c in p.get("categories", ["Uncategorized"]))
        tags_yaml       = "\n".join(f'  - "{t}"' for t in p.get("theme_tags", []))

        authors_list = p.get("authors", [])
        authors_display = ", ".join(authors_list) if authors_list else "Unknown"

        doi   = p.get("doi", "")
        link  = p.get("active_link", "")
        year  = p.get("year", "")
        src   = p.get("source_name", "")
        reason = p.get("seminal_reason", "")
        abstract = p.get("abstract", "")

        # Format optional fields
        doi_line  = f'doi: "{doi}"' if doi else 'doi: ""'
        link_line = f'link: "{link}"' if link else 'link: ""'

        # Truncate abstract for front matter if very long
        abstract_short = abstract[:500] + "…" if len(abstract) > 500 else abstract
        abstract_safe  = abstract_short.replace('"', "'")

        post_content = f"""---
layout: seminal
title: "{title.replace('"', "'")}"
date: {post_date}
authors: "{authors_display.replace('"', "'")}"
year: {year}
source: "{src}"
{doi_line}
{link_line}
abstract: "{abstract_safe}"
seminal_reason: "{reason.replace('"', "'")}"
categories:
{categories_yaml}
tags:
{tags_yaml}
run_id: "{p.get('run_id', '')}"
---

## {title}

**Authors:** {authors_display}  
**Year:** {year}  
**Source:** {src}  
{f"**DOI:** [{doi}](https://doi.org/{doi})  " if doi else ""}
{f"**Link:** [{link}]({link})  " if link else ""}

### Why this work is foundational

{reason or "_No reason provided._"}

### Abstract

{abstract or "_Abstract not available._"}

---
*Excavated by ARANEA — Grounder agent · Run `{p.get('run_id', 'N/A')}`*
"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(post_content)

        written.append(filepath)

    return written


# ─────────────────────────────────────────────────────────────────────────────
# Summary report (printed to terminal)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(papers: list[dict]) -> None:
    by_cat: dict[str, list] = defaultdict(list)
    for p in papers:
        by_cat[p["primary_category"]].append(p)

    print(f"\n{'─'*60}")
    print(f"  ARANEA — Seminal Works Export")
    print(f"  {len(papers)} papers across {len(by_cat)} categories")
    print(f"{'─'*60}")
    for cat in sorted(by_cat):
        entries = by_cat[cat]
        print(f"\n  {cat}  ({len(entries)} papers)")
        for p in sorted(entries, key=lambda x: x.get("year") or 0):
            year   = f"({p['year']})" if p.get("year") else "(year?)"
            link   = "🔗" if p.get("active_link") else "  "
            title  = p["title"][:55] + "…" if len(p["title"]) > 55 else p["title"]
            print(f"    {link} {year:6}  {title}")
    print(f"\n{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ARANEA seminal works to JSON, CSV, and optional Jekyll posts."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"Path to pipeline.db  (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        metavar="RUN_ID",
        help="Filter to a specific run ID  (default: all runs)",
    )
    parser.add_argument(
        "--jekyll",
        action="store_true",
        help="Also generate Jekyll Markdown posts in exports/jekyll/_posts/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=EXPORTS_DIR,
        help=f"Output directory  (default: {EXPORTS_DIR})",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all runs that have seminal papers and exit.",
    )
    args = parser.parse_args()

    # ── list runs mode ───────────────────────────────────────────────────────
    if args.list_runs:
        if not args.db.exists():
            sys.exit(f"[ERROR] Database not found: {args.db}")
        conn = sqlite3.connect(args.db)
        rows = conn.execute(
            "SELECT run_id, COUNT(*) as n FROM sources "
            "WHERE type='seminal' GROUP BY run_id ORDER BY run_id"
        ).fetchall()
        conn.close()
        if not rows:
            print("No seminal papers found in the database yet.")
            print("Run the pipeline first: python3 main.py run --problem '...'")
        else:
            print(f"\nRuns with seminal papers:")
            for run_id, count in rows:
                print(f"  {run_id}  —  {count} papers")
            print()
        return

    # ── load papers ──────────────────────────────────────────────────────────
    papers = load_seminal(args.db, run_id=args.run)

    if not papers:
        msg = "No seminal papers found"
        if args.run:
            msg += f" for run {args.run}"
        msg += ".\nTip: run --list-runs to see available runs."
        sys.exit(msg)

    # ── create output dir ────────────────────────────────────────────────────
    args.out.mkdir(parents=True, exist_ok=True)

    # ── print terminal summary ───────────────────────────────────────────────
    print_summary(papers)

    # ── JSON ─────────────────────────────────────────────────────────────────
    json_path = export_json(papers, args.out)
    print(f"  ✓  JSON  →  {json_path}")

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = export_csv(papers, args.out)
    print(f"  ✓  CSV   →  {csv_path}")

    # ── Jekyll (optional) ────────────────────────────────────────────────────
    if args.jekyll:
        jekyll_posts = export_jekyll(papers, JEKYLL_DIR)
        print(f"  ✓  Jekyll posts ({len(jekyll_posts)})  →  {JEKYLL_DIR}/")

    print(f"\n  Done. Drop seminal_references.json in your blog's _data/ folder")
    print(f"  and use it with Liquid to render the references page.\n")


if __name__ == "__main__":
    main()
