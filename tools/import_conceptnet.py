"""
ConceptNet CSV Filter and SQLite Importer
-----------------------------------------
One-time script to convert the raw ConceptNet 5.7 CSV dump
into a lean, indexed SQLite database for local concept expansion.

Usage:
  python3 tools/import_conceptnet.py --input /path/to/conceptnet-assertions-5.7.0.csv.gz
  python3 tools/import_conceptnet.py --input /path/to/conceptnet-assertions-5.7.0.csv.gz --dry-run
  python3 tools/import_conceptnet.py --stats   (after import — show what was loaded)

The script reads the .csv.gz directly without decompressing to disk.
Progress is printed every 500k lines.

Output:
  pipeline/db/conceptnet.db  — SQLite database, ~300-500MB
  Table: edges (term, relation, target, weight)
  Index: on term column for fast lookups

Filter criteria:
  - Both term AND target must be English (/c/en/)
  - Relation must be in the useful set (13 relations)
  - Weight >= MIN_WEIGHT (default 1.0)
  - Term text max MAX_WORDS words (default 4)
  - Skips reflexive edges (term == target)
"""

import gzip
import json
import sqlite3
import argparse
import sys
import re
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — adjust these if you want more or less data
# ---------------------------------------------------------------------------

# Minimum weight to keep an edge. ConceptNet weights are positive floats.
# 1.0 is the typical weight — keeping >=1.0 removes the noisiest edges.
# Lower to 0.5 to get more coverage, raise to 2.0 for higher confidence only.
MIN_WEIGHT = 1.0

# Maximum number of words in a term (after underscore splitting).
# "artificial_intelligence" = 2 words — keep.
# "a_long_multi_word_phrase_that_is_not_useful" = 8 words — skip.
MAX_WORDS = 4

# Relations to keep — these are useful for semantic expansion.
# Full list at: https://github.com/commonsense/conceptnet5/wiki/Relations
USEFUL_RELATIONS = {
    "/r/RelatedTo",       # General relatedness
    "/r/IsA",             # Hypernymy / taxonomy
    "/r/HasContext",       # Domain context (e.g. "biology", "philosophy")
    "/r/PartOf",          # Meronymy
    "/r/SimilarTo",       # Similarity
    "/r/DefinedAs",       # Definition
    "/r/HasProperty",     # Properties
    "/r/InstanceOf",      # Instance of a class
    "/r/MannerOf",        # Is a manner of
    "/r/Causes",          # Causal relations
    "/r/CapableOf",       # Capabilities
    "/r/UsedFor",         # Purpose / function
    "/r/Synonym",         # Synonymy
}

# Output database path
DB_PATH = Path(__file__).parent.parent / "db" / "conceptnet.db"

# Batch size for SQLite inserts — larger = faster, more memory
BATCH_SIZE = 10_000

# Progress report every N lines
PROGRESS_EVERY = 500_000


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def extract_term(uri: str) -> str:
    """
    Extract the human-readable term from a ConceptNet URI.
    /c/en/artificial_intelligence/n -> "artificial intelligence"
    /c/en/mind_body_problem       -> "mind body problem"
    Returns empty string if not English or malformed.
    """
    # Must be English
    if not uri.startswith("/c/en/"):
        return ""
    # Strip /c/en/ prefix and any trailing part-of-speech tags
    # URI format: /c/en/TERM or /c/en/TERM/POS or /c/en/TERM/POS/dataset/...
    parts = uri[6:].split("/")  # strip "/c/en/"
    raw = parts[0]
    if not raw:
        return ""
    # Convert underscores to spaces
    term = raw.replace("_", " ").strip()
    return term


def count_words(term: str) -> int:
    return len(term.split())


def extract_relation(rel_uri: str) -> str:
    """
    Extract relation label from URI.
    /r/IsA -> /r/IsA  (keep as-is for clarity)
    """
    return rel_uri.strip()


def parse_weight(metadata_json: str) -> float:
    """Extract weight from the JSON metadata field."""
    try:
        data = json.loads(metadata_json)
        return float(data.get("weight", 1.0))
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS edges (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    term     TEXT NOT NULL,
    relation TEXT NOT NULL,
    target   TEXT NOT NULL,
    weight   REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edges_term ON edges(term);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
"""

STATS_TABLE = """
CREATE TABLE IF NOT EXISTS import_stats (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
    conn.executescript(SCHEMA)
    conn.executescript(STATS_TABLE)
    return conn


# ---------------------------------------------------------------------------
# Main filter loop
# ---------------------------------------------------------------------------

def run_import(input_path: Path, dry_run: bool = False, verbose: bool = True):
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  ConceptNet CSV Filter + SQLite Import")
    print(f"{'='*60}")
    print(f"  Input:      {input_path}")
    print(f"  Output:     {DB_PATH}")
    print(f"  Min weight: {MIN_WEIGHT}")
    print(f"  Max words:  {MAX_WORDS}")
    print(f"  Relations:  {len(USEFUL_RELATIONS)}")
    print(f"  Dry run:    {dry_run}")
    print(f"{'='*60}\n")

    if dry_run:
        print("DRY RUN — reading first 2,000,000 lines only, not writing to DB\n")

    conn = None
    if not dry_run:
        print(f"Initializing database at {DB_PATH}...")
        conn = init_db(DB_PATH)
        # Clear existing edges if re-running
        conn.execute("DELETE FROM edges")
        conn.commit()
        print("Database ready.\n")

    # Counters
    total_lines    = 0
    kept           = 0
    skip_lang      = 0
    skip_rel       = 0
    skip_weight    = 0
    skip_words     = 0
    skip_reflexive = 0
    skip_malformed = 0

    # Relation frequency counter (for stats)
    rel_counts = {}

    batch = []
    start_time = time.time()

    # Open gzipped or plain CSV
    opener = gzip.open if str(input_path).endswith(".gz") else open
    mode   = "rt" if str(input_path).endswith(".gz") else "r"

    print("Reading file...")
    with opener(input_path, mode, encoding="utf-8", errors="replace") as f:
        for line in f:
            total_lines += 1

            # Progress report
            if verbose and total_lines % PROGRESS_EVERY == 0:
                elapsed = time.time() - start_time
                rate    = total_lines / elapsed if elapsed > 0 else 0
                print(
                    f"  Line {total_lines:>12,} | "
                    f"Kept: {kept:>8,} | "
                    f"Rate: {rate:>8,.0f} lines/sec | "
                    f"Elapsed: {elapsed:>6.0f}s"
                )

            # Dry run limit
            if dry_run and total_lines > 2_000_000:
                print("\nDry run limit reached (2M lines).")
                break

            # Parse TSV line: edge_uri, relation, start, end, metadata
            line = line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) < 5:
                skip_malformed += 1
                continue

            _, relation, start_uri, end_uri, metadata = parts[0], parts[1], parts[2], parts[3], parts[4]

            # Filter 1: Relation must be useful
            if relation not in USEFUL_RELATIONS:
                skip_rel += 1
                continue

            # Filter 2: Both nodes must be English
            term   = extract_term(start_uri)
            target = extract_term(end_uri)
            if not term or not target:
                skip_lang += 1
                continue

            # Filter 3: Weight
            weight = parse_weight(metadata)
            if weight < MIN_WEIGHT:
                skip_weight += 1
                continue

            # Filter 4: Word count
            if count_words(term) > MAX_WORDS or count_words(target) > MAX_WORDS:
                skip_words += 1
                continue

            # Filter 5: No reflexive edges
            if term.lower() == target.lower():
                skip_reflexive += 1
                continue

            # Passed all filters
            kept += 1
            rel_counts[relation] = rel_counts.get(relation, 0) + 1

            if not dry_run:
                batch.append((term.lower(), relation, target.lower(), weight))
                if len(batch) >= BATCH_SIZE:
                    conn.executemany(
                        "INSERT INTO edges (term, relation, target, weight) VALUES (?,?,?,?)",
                        batch
                    )
                    conn.commit()
                    batch.clear()

    # Flush remaining batch
    if not dry_run and batch:
        conn.executemany(
            "INSERT INTO edges (term, relation, target, weight) VALUES (?,?,?,?)",
            batch
        )
        conn.commit()

    elapsed = time.time() - start_time

    # ---------------------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Import Complete")
    print(f"{'='*60}")
    print(f"  Total lines read:    {total_lines:>12,}")
    print(f"  Edges kept:          {kept:>12,}")
    print(f"  Keep rate:           {kept/total_lines*100:>11.1f}%")
    print(f"")
    print(f"  Filtered out:")
    print(f"    Non-English:       {skip_lang:>12,}")
    print(f"    Wrong relation:    {skip_rel:>12,}")
    print(f"    Low weight:        {skip_weight:>12,}")
    print(f"    Too many words:    {skip_words:>12,}")
    print(f"    Reflexive:         {skip_reflexive:>12,}")
    print(f"    Malformed:         {skip_malformed:>12,}")
    print(f"")
    print(f"  Time elapsed:        {elapsed:>10.0f}s ({elapsed/60:.1f} min)")
    print(f"  Throughput:          {total_lines/elapsed:>10,.0f} lines/sec")
    print(f"")
    print(f"  Relation breakdown (kept edges):")
    for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
        rel_short = rel.replace("/r/", "")
        print(f"    {rel_short:<25} {count:>10,}")

    if not dry_run:
        # Save stats to DB
        stats = {
            "total_lines":    str(total_lines),
            "edges_kept":     str(kept),
            "import_date":    time.strftime("%Y-%m-%d %H:%M:%S"),
            "min_weight":     str(MIN_WEIGHT),
            "max_words":      str(MAX_WORDS),
            "source_file":    str(input_path.name),
        }
        for k, v in stats.items():
            conn.execute(
                "INSERT OR REPLACE INTO import_stats (key, value) VALUES (?,?)",
                (k, v)
            )
        conn.commit()

        # Database file size
        db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)
        print(f"\n  Database size:       {db_size_mb:>10.1f} MB")
        print(f"  Database path:       {DB_PATH}")
        conn.close()

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Stats command — show what is in the DB
# ---------------------------------------------------------------------------

def show_stats():
    if not DB_PATH.exists():
        print("No conceptnet.db found. Run import first.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    print(f"\n{'='*60}")
    print(f"  ConceptNet SQLite Database Stats")
    print(f"{'='*60}")

    # Import stats
    rows = conn.execute("SELECT key, value FROM import_stats").fetchall()
    if rows:
        print(f"\n  Import metadata:")
        for r in rows:
            print(f"    {r['key']:<20} {r['value']}")

    # Edge count
    total = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    print(f"\n  Total edges:         {total:>12,}")

    # Relation breakdown
    print(f"\n  By relation:")
    rels = conn.execute(
        "SELECT relation, COUNT(*) as cnt FROM edges GROUP BY relation ORDER BY cnt DESC"
    ).fetchall()
    for r in rels:
        rel_short = r["relation"].replace("/r/", "")
        print(f"    {rel_short:<25} {r['cnt']:>10,}")

    # Sample lookup
    print(f"\n  Sample lookup: 'intelligence'")
    samples = conn.execute(
        "SELECT relation, target, weight FROM edges WHERE term = 'intelligence' ORDER BY weight DESC LIMIT 10"
    ).fetchall()
    for s in samples:
        print(f"    [{s['relation'].replace('/r/','')}] → {s['target']} (w={s['weight']})")

    db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"\n  Database size:       {db_size_mb:>10.1f} MB")
    print(f"  Database path:       {DB_PATH}")
    print(f"{'='*60}\n")
    conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global MIN_WEIGHT, MAX_WORDS
    parser = argparse.ArgumentParser(
        description="Filter and import ConceptNet CSV dump into local SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full import (reads .gz directly)
  python3 tools/import_conceptnet.py --input ~/downloads/conceptnet-assertions-5.7.0.csv.gz

  # Dry run — test filters on first 2M lines without writing
  python3 tools/import_conceptnet.py --input ~/downloads/conceptnet-assertions-5.7.0.csv.gz --dry-run

  # Show stats after import
  python3 tools/import_conceptnet.py --stats

  # Adjust weight threshold (keep more edges)
  python3 tools/import_conceptnet.py --input file.csv.gz --min-weight 0.5
        """
    )
    parser.add_argument("--input",      type=Path, help="Path to conceptnet-assertions-5.7.0.csv.gz")
    parser.add_argument("--dry-run",    action="store_true", help="Read first 2M lines, don't write to DB")
    parser.add_argument("--stats",      action="store_true", help="Show DB stats (after import)")
    parser.add_argument("--min-weight", type=float, default=MIN_WEIGHT,
                        help=f"Minimum edge weight to keep (default: {MIN_WEIGHT})")
    parser.add_argument("--max-words",  type=int, default=MAX_WORDS,
                        help=f"Maximum words in a term (default: {MAX_WORDS})")
    parser.add_argument("--quiet",      action="store_true", help="Suppress per-line progress")

    args = parser.parse_args()

    # Apply CLI overrides
    MIN_WEIGHT = args.min_weight
    MAX_WORDS  = args.max_words

    if args.stats:
        show_stats()
        return

    if not args.input:
        parser.print_help()
        print("\nERROR: --input is required unless using --stats")
        sys.exit(1)

    run_import(args.input, dry_run=args.dry_run, verbose=not args.quiet)


if __name__ == "__main__":
    main()
