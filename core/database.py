"""
Database Interface
------------------
Single SQLite file: db/pipeline.db
Handles all agent output storage with full metadata.

Tables:
  - runs             Pipeline run registry
  - sources          active / seminal / historical entries
  - dead_links       Dead link archive
  - gaps             Gaper output
  - implications     Vision output
  - proposals        Theorist output
  - evaluations      Rude output
  - syntheses        Synthesizer output
  - directions       Thinker output
  - artifacts        Scribe output
  - seminal_bank     Grounder proposed themes
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "db" / "pipeline.db"


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
-- Pipeline runs registry
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    problem         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    status          TEXT DEFAULT 'active',
    break0_done     INTEGER DEFAULT 0,
    break1_done     INTEGER DEFAULT 0,
    break2_done     INTEGER DEFAULT 0,
    completed_at    TEXT
);

-- Sources: current (Social), seminal (Grounder), historical (Historian)
CREATE TABLE IF NOT EXISTS sources (
    source_id       TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    authors         TEXT,                -- JSON array
    year            INTEGER,
    source_name     TEXT,
    doi             TEXT,
    abstract        TEXT,
    active_link     TEXT,
    theme_tags      TEXT,                -- JSON array
    type            TEXT NOT NULL,       -- current / seminal / historical
    relevance_rating TEXT,              -- High / Medium / Low (current)
    relevance_reason TEXT,
    seminal_reason  TEXT,               -- Grounder
    historical_reason TEXT,             -- Historian
    phase_tag       TEXT,               -- Historian phase classification
    intersection_tags TEXT,             -- JSON array
    added_by        TEXT,
    date_collected  TEXT,
    last_checked    TEXT,
    link_status     TEXT DEFAULT 'active', -- active / redirected / dead / flagged
    run_id          TEXT
);

-- Dead links archive
CREATE TABLE IF NOT EXISTS dead_links (
    dead_id         TEXT PRIMARY KEY,
    source_id       TEXT,
    title           TEXT,
    original_link   TEXT,
    theme_tags      TEXT,
    type            TEXT,
    date_collected  TEXT,
    date_confirmed_dead TEXT,
    last_active     TEXT
);

-- Gaper: gaps
CREATE TABLE IF NOT EXISTS gaps (
    gap_id          TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    problem_origin  TEXT,
    gap_type        TEXT,               -- unstudied / incomplete / contradicted etc.
    description     TEXT NOT NULL,
    significance    TEXT,               -- High / Medium / Low
    significance_reason TEXT,
    primary_evaluation TEXT,            -- answered / partial / unanswered
    references_grounder TEXT,           -- JSON array of source_ids
    references_historian TEXT,          -- JSON array of source_ids
    references_social TEXT,             -- JSON array of source_ids
    dead_end_revisit INTEGER DEFAULT 0,
    recurring_pattern INTEGER DEFAULT 0,
    recurring_reason TEXT,
    added_by        TEXT DEFAULT 'Gaper',
    date_identified TEXT,
    status          TEXT DEFAULT 'open' -- open / addressed / resolved / deferred
);

-- Vision: implications
CREATE TABLE IF NOT EXISTS implications (
    implication_id  TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    problem_origin  TEXT,
    implication     TEXT NOT NULL,
    implication_type TEXT,              -- direct / logical_chain / second_order etc.
    strength        TEXT,               -- Strong / Moderate / Speculative
    strength_reason TEXT,
    scope           TEXT,               -- immediate / second_order
    derived_grounder TEXT,              -- JSON array
    derived_historian TEXT,             -- JSON array
    derived_gaper   TEXT,               -- JSON array
    derived_social  TEXT,               -- JSON array
    hidden_assumption INTEGER DEFAULT 0,
    assumption_note TEXT,
    currently_pursued INTEGER DEFAULT 0,
    pursuit_reference TEXT,
    added_by        TEXT DEFAULT 'Vision',
    date_identified TEXT,
    status          TEXT DEFAULT 'active'
);

-- Theorist: proposals
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id     TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    problem_origin  TEXT,
    proposal        TEXT NOT NULL,
    proposal_type   TEXT,               -- novel / extension / revival / hybrid
    addresses_gaps  TEXT,               -- JSON array of gap_ids
    addresses_implications TEXT,        -- JSON array of implication_ids
    addresses_foundations TEXT,         -- JSON array of source_ids
    assumptions     TEXT,               -- JSON array
    requirements    TEXT,               -- JSON array
    predictions     TEXT,               -- JSON array
    dead_end_reassessment INTEGER DEFAULT 0,
    dead_end_reference TEXT,
    dead_end_reason TEXT,
    interdependencies TEXT,             -- JSON array of proposal_ids
    promise_rating  TEXT,               -- High / Medium / Low
    promise_reason  TEXT,
    novel_vs_extension TEXT,
    scope           TEXT,
    added_by        TEXT DEFAULT 'Theorist',
    date_proposed   TEXT,
    status          TEXT DEFAULT 'proposed'
);

-- Rude: evaluations
CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id   TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    proposal_id     TEXT NOT NULL,
    problem_origin  TEXT,
    verdict         TEXT NOT NULL,      -- feasible / partially_feasible / unfeasible / insufficient_evidence
    verdict_reason  TEXT,
    weakest_empirical_link TEXT,
    dead_end_references TEXT,           -- JSON array
    social_evidence_references TEXT,    -- JSON array
    evidence_to_change_verdict TEXT,
    added_by        TEXT DEFAULT 'Rude',
    date_evaluated  TEXT,
    status          TEXT DEFAULT 'active'
);

-- Synthesizer: research narratives
CREATE TABLE IF NOT EXISTS syntheses (
    synthesis_id    TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    problem_origin  TEXT,
    sharpened_problem TEXT,
    trajectory_statement TEXT,
    key_tensions    TEXT,               -- JSON array
    override_log    TEXT,               -- JSON array
    viable_proposal_ids TEXT,           -- JSON array
    top_gap_ids     TEXT,               -- JSON array
    top_implication_ids TEXT,           -- JSON array
    full_narrative  TEXT,               -- full text of the narrative
    added_by        TEXT DEFAULT 'Synthesizer',
    date_produced   TEXT,
    status          TEXT DEFAULT 'draft'
);

-- Thinker: new directions
CREATE TABLE IF NOT EXISTS directions (
    direction_id    TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    problem_origin  TEXT,
    direction       TEXT NOT NULL,
    direction_type  TEXT,               -- new_research / new_framing / adjacent_field etc.
    grounding_reference TEXT,
    distance_rating TEXT,               -- Near / Mid / Far
    synthesis_id    TEXT,
    added_by        TEXT DEFAULT 'Thinker',
    date_proposed   TEXT,
    status          TEXT DEFAULT 'proposed'
);

-- Scribe: produced artifacts
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id     TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    problem_origin  TEXT,
    output_type     TEXT,               -- blog_post / research_brief / paper_section etc.
    format          TEXT,               -- md / tex
    title           TEXT,
    audience        TEXT,
    synthesis_id    TEXT,
    directions_used TEXT,               -- JSON array of direction_ids
    file_path       TEXT,
    word_count      INTEGER,
    added_by        TEXT DEFAULT 'Scribe',
    date_produced   TEXT,
    status          TEXT DEFAULT 'draft'
);

-- Grounder: proposed themes for seminal bank
CREATE TABLE IF NOT EXISTS seminal_bank (
    bank_id         TEXT PRIMARY KEY,
    proposed_theme  TEXT NOT NULL,
    proposed_by     TEXT DEFAULT 'Grounder',
    problem_origin  TEXT,
    reason          TEXT,
    suggested_keywords TEXT,            -- JSON array
    suggested_sources TEXT,             -- JSON array
    date_proposed   TEXT,
    status          TEXT DEFAULT 'pending_review' -- pending_review / approved / rejected
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_sources_type     ON sources(type);
CREATE INDEX IF NOT EXISTS idx_sources_run      ON sources(run_id);
CREATE INDEX IF NOT EXISTS idx_gaps_run         ON gaps(run_id);
CREATE INDEX IF NOT EXISTS idx_gaps_significance ON gaps(significance);
CREATE INDEX IF NOT EXISTS idx_implications_run ON implications(run_id);
CREATE INDEX IF NOT EXISTS idx_proposals_run    ON proposals(run_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_run  ON evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_proposal ON evaluations(proposal_id);
CREATE INDEX IF NOT EXISTS idx_syntheses_run    ON syntheses(run_id);
CREATE INDEX IF NOT EXISTS idx_directions_run   ON directions(run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_run    ON artifacts(run_id);
"""


def init_db():
    """Initialize database — create all tables if they don't exist."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)
    logger.info(f"Database initialized at {DB_PATH}")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _json(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)

def _from_json(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def insert(table: str, data: dict) -> bool:
    """Generic insert into any table."""
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    sql = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})"
    try:
        with get_connection() as conn:
            conn.execute(sql, list(data.values()))
        return True
    except Exception as e:
        logger.error(f"Insert into {table} failed: {e}")
        return False


def fetch(table: str, where: dict = None, limit: int = None) -> list[dict]:
    """Generic fetch from any table."""
    sql = f"SELECT * FROM {table}"
    params = []
    if where:
        conditions = " AND ".join([f"{k} = ?" for k in where.keys()])
        sql += f" WHERE {conditions}"
        params = list(where.values())
    if limit:
        sql += f" LIMIT {limit}"
    try:
        with get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Fetch from {table} failed: {e}")
        return []


def update(table: str, data: dict, where: dict) -> bool:
    """Generic update on any table."""
    set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
    where_clause = " AND ".join([f"{k} = ?" for k in where.keys()])
    sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
    params = list(data.values()) + list(where.values())
    try:
        with get_connection() as conn:
            conn.execute(sql, params)
        return True
    except Exception as e:
        logger.error(f"Update {table} failed: {e}")
        return False


def count(table: str, where: dict = None) -> int:
    """Count rows in a table."""
    sql = f"SELECT COUNT(*) FROM {table}"
    params = []
    if where:
        conditions = " AND ".join([f"{k} = ?" for k in where.keys()])
        sql += f" WHERE {conditions}"
        params = list(where.values())
    try:
        with get_connection() as conn:
            return conn.execute(sql, params).fetchone()[0]
    except Exception as e:
        logger.error(f"Count {table} failed: {e}")
        return 0


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def create_run(run_id: str, problem: str) -> bool:
    return insert("runs", {
        "run_id":     run_id,
        "problem":    problem,
        "created_at": _now(),
        "status":     "active"
    })


def get_run(run_id: str) -> Optional[dict]:
    rows = fetch("runs", {"run_id": run_id})
    return rows[0] if rows else None


def update_run_status(run_id: str, status: str) -> bool:
    data = {"status": status}
    if status == "completed":
        data["completed_at"] = _now()
    return update("runs", data, {"run_id": run_id})


def mark_break_done(run_id: str, break_num: int) -> bool:
    col = f"break{break_num}_done"
    return update("runs", {col: 1}, {"run_id": run_id})


# ---------------------------------------------------------------------------
# Source management (Social / Grounder / Historian)
# ---------------------------------------------------------------------------

def upsert_source(source: dict) -> bool:
    """Insert or update a source entry."""
    for field in ["authors", "theme_tags", "intersection_tags"]:
        if field in source:
            source[field] = _json(source[field])
    return insert("sources", source)


def get_sources_by_type(source_type: str, run_id: str = None) -> list[dict]:
    where = {"type": source_type}
    if run_id:
        where["run_id"] = run_id
    return fetch("sources", where)


def archive_dead_link(source: dict) -> bool:
    """Move a dead source to dead_links table."""
    dead = {
        "dead_id":             f"DEAD-{source['source_id']}",
        "source_id":           source["source_id"],
        "title":               source.get("title"),
        "original_link":       source.get("active_link"),
        "theme_tags":          source.get("theme_tags"),
        "type":                source.get("type"),
        "date_collected":      source.get("date_collected"),
        "date_confirmed_dead": _now(),
        "last_active":         source.get("last_checked")
    }
    ok = insert("dead_links", dead)
    if ok:
        update("sources", {"link_status": "dead"}, {"source_id": source["source_id"]})
    return ok


# ---------------------------------------------------------------------------
# Gap management (Gaper)
# ---------------------------------------------------------------------------

def insert_gap(gap: dict) -> bool:
    for field in ["references_grounder", "references_historian", "references_social"]:
        if field in gap:
            gap[field] = _json(gap[field])
    if "date_identified" not in gap:
        gap["date_identified"] = _now()
    return insert("gaps", gap)


def get_gaps(run_id: str, significance: str = None) -> list[dict]:
    where = {"run_id": run_id}
    if significance:
        where["significance"] = significance
    return fetch("gaps", where)


# ---------------------------------------------------------------------------
# Implication management (Vision)
# ---------------------------------------------------------------------------

def insert_implication(imp: dict) -> bool:
    for field in ["derived_grounder", "derived_historian", "derived_gaper", "derived_social"]:
        if field in imp:
            imp[field] = _json(imp[field])
    if "date_identified" not in imp:
        imp["date_identified"] = _now()
    return insert("implications", imp)


def get_implications(run_id: str, strength: str = None) -> list[dict]:
    where = {"run_id": run_id}
    if strength:
        where["strength"] = strength
    return fetch("implications", where)


# ---------------------------------------------------------------------------
# Proposal management (Theorist)
# ---------------------------------------------------------------------------

def insert_proposal(proposal: dict) -> bool:
    for field in ["addresses_gaps", "addresses_implications", "addresses_foundations",
                  "assumptions", "requirements", "predictions", "interdependencies"]:
        if field in proposal:
            proposal[field] = _json(proposal[field])
    if "date_proposed" not in proposal:
        proposal["date_proposed"] = _now()
    return insert("proposals", proposal)


def get_proposals(run_id: str, status: str = None) -> list[dict]:
    where = {"run_id": run_id}
    if status:
        where["status"] = status
    return fetch("proposals", where)


# ---------------------------------------------------------------------------
# Evaluation management (Rude)
# ---------------------------------------------------------------------------

def insert_evaluation(evaluation: dict) -> bool:
    for field in ["dead_end_references", "social_evidence_references"]:
        if field in evaluation:
            evaluation[field] = _json(evaluation[field])
    if "date_evaluated" not in evaluation:
        evaluation["date_evaluated"] = _now()
    # Update proposal status
    if "proposal_id" in evaluation:
        verdict_to_status = {
            "feasible":              "feasible",
            "partially_feasible":    "feasible",
            "unfeasible":            "rejected",
            "insufficient_evidence": "deferred"
        }
        new_status = verdict_to_status.get(evaluation.get("verdict"), "under_review")
        update("proposals", {"status": new_status}, {"proposal_id": evaluation["proposal_id"]})
    return insert("evaluations", evaluation)


def get_evaluations(run_id: str, verdict: str = None) -> list[dict]:
    where = {"run_id": run_id}
    if verdict:
        where["verdict"] = verdict
    return fetch("evaluations", where)


# ---------------------------------------------------------------------------
# Synthesis management (Synthesizer)
# ---------------------------------------------------------------------------

def insert_synthesis(synthesis: dict) -> bool:
    for field in ["key_tensions", "override_log", "viable_proposal_ids",
                  "top_gap_ids", "top_implication_ids"]:
        if field in synthesis:
            synthesis[field] = _json(synthesis[field])
    if "date_produced" not in synthesis:
        synthesis["date_produced"] = _now()
    return insert("syntheses", synthesis)


def get_synthesis(run_id: str) -> Optional[dict]:
    rows = fetch("syntheses", {"run_id": run_id})
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Direction management (Thinker)
# ---------------------------------------------------------------------------

def insert_direction(direction: dict) -> bool:
    if "date_proposed" not in direction:
        direction["date_proposed"] = _now()
    return insert("directions", direction)


def get_directions(run_id: str) -> list[dict]:
    return fetch("directions", {"run_id": run_id})


# ---------------------------------------------------------------------------
# Artifact management (Scribe)
# ---------------------------------------------------------------------------

def insert_artifact(artifact: dict) -> bool:
    if field := artifact.get("directions_used"):
        artifact["directions_used"] = _json(field)
    if "date_produced" not in artifact:
        artifact["date_produced"] = _now()
    return insert("artifacts", artifact)


def get_artifacts(run_id: str) -> list[dict]:
    return fetch("artifacts", {"run_id": run_id})


# ---------------------------------------------------------------------------
# Seminal bank (Grounder proposals)
# ---------------------------------------------------------------------------

def insert_seminal_proposal(proposal: dict) -> bool:
    for field in ["suggested_keywords", "suggested_sources"]:
        if field in proposal:
            proposal[field] = _json(proposal[field])
    if "date_proposed" not in proposal:
        proposal["date_proposed"] = _now()
    return insert("seminal_bank", proposal)


def get_seminal_bank(status: str = "pending_review") -> list[dict]:
    return fetch("seminal_bank", {"status": status})
