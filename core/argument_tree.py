"""
Argument Tree — Core Data Structure
------------------------------------
A persistent tree that grows across agents during a pipeline run.
Every claim traces to evidence. Every evidence node links to a verifiable
source. The tree is the single source of truth — downstream agents read it,
not raw text dumps.

Node types:
  root        — the research question
  question    — sub-question from Grounder decomposition
  claim       — assertion derived from evidence
  evidence    — link to a verifiable source (see evidence_type below)
  bridge      — connects two distant nodes across time/discipline
  counter     — contradicts a parent claim
  historical  — historical context or event
  external    — external factor (war, policy, funding shift, institutional change)
  audit_note  — Historian's assessment of a node or branch solidity

Evidence types (what "evidence" can be):
  paper            — peer-reviewed journal article, conference paper
  book             — monograph, edited volume, book chapter
  report           — government report, NGO report, technical report
  legal_document   — law, statute, treaty, court decision, legal opinion
  court_decision   — specific judicial ruling (ICJ, ICC, domestic court)
  news_article     — verified news from established outlet
  archival         — primary source, archive document, historical record
  testimony        — witness testimony, oral history, deposition
  resolution       — UN resolution, organizational resolution
  dataset          — statistical data, survey, census
  other            — anything else verifiable

Status values:
  supported     — claim has sufficient evidence
  contested     — claim has both supporting and contradicting evidence
  unsupported   — claim lacks evidence (not necessarily wrong)
  weak          — claim has evidence but low confidence
  solid         — audited by Historian, confirmed strong
  contradicted  — evidence directly contradicts the claim
  bridged       — gap between nodes has been connected

Usage:
    from core.argument_tree import TreeBuilder

    tree = TreeBuilder(run_id)
    root = tree.create_root(problem)
    q1 = tree.add_question(root, "What is identity?")
    c1 = tree.add_claim(q1, "Identity is socially constructed", confidence=0.8)
    tree.add_evidence(c1, source_id="SRC-xxx", evidence_type="book",
                      relationship="establishes", snippet="Mead argues...")
    tree.add_bridge(from_node=c1, to_node=c9, source_id="SRC-bbb",
                    bridge_type="temporal")

    # Get full tree for context building
    full = tree.get_tree()

    # Get branches relevant to a specific question
    branch = tree.get_branch(q1)

    # Find gaps
    gaps = tree.find_gaps()

    # Find bridge needs
    needs = tree.find_bridge_needs(min_gap_years=15)
"""

from __future__ import annotations

import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from core.utils import generate_id

logger = logging.getLogger(__name__)

_HERE   = Path(__file__).parent.parent
DB_PATH = _HERE / "db" / "pipeline.db"


# ─── Schema ───────────────────────────────────────────────────────────────────

TREE_SCHEMA = """
CREATE TABLE IF NOT EXISTS argument_tree (
    node_id         TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    parent_node_id  TEXT,
    node_type       TEXT NOT NULL,
    depth           INTEGER DEFAULT 0,
    content         TEXT NOT NULL,
    status          TEXT DEFAULT 'unsupported',
    confidence      REAL DEFAULT 0.0,
    source_ids      TEXT DEFAULT '[]',
    agent_origin    TEXT DEFAULT '',
    created_at      TEXT NOT NULL,
    metadata        TEXT DEFAULT '{}',

    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (parent_node_id) REFERENCES argument_tree(node_id)
);

CREATE INDEX IF NOT EXISTS idx_tree_run ON argument_tree(run_id);
CREATE INDEX IF NOT EXISTS idx_tree_parent ON argument_tree(parent_node_id);
CREATE INDEX IF NOT EXISTS idx_tree_type ON argument_tree(run_id, node_type);
"""

VALID_NODE_TYPES = {
    "root", "question", "claim", "evidence", "bridge",
    "counter", "historical", "external", "audit_note",
}

VALID_EVIDENCE_TYPES = {
    "paper", "book", "report", "legal_document", "court_decision",
    "news_article", "archival", "testimony", "resolution", "dataset", "other",
}

VALID_STATUSES = {
    "supported", "contested", "unsupported", "weak",
    "solid", "contradicted", "bridged",
}


# ─── Initialize ───────────────────────────────────────────────────────────────

def init_tree_table():
    """Create the argument_tree table if it doesn't exist."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(TREE_SCHEMA)
    conn.commit()
    conn.close()
    logger.debug("[Tree] Table initialized")


# ─── TreeBuilder ──────────────────────────────────────────────────────────────

class TreeBuilder:
    """
    Builds and maintains the argument tree for a pipeline run.
    All mutations go through this class. Thread-safe per run_id.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._conn = sqlite3.connect(str(DB_PATH))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Ensure table exists
        self._conn.executescript(TREE_SCHEMA)
        self._conn.commit()

    def close(self):
        self._conn.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _insert(self, node_id: str, parent_id: Optional[str], node_type: str,
                content: str, status: str = "unsupported", confidence: float = 0.0,
                source_ids: list = None, agent: str = "",
                metadata: dict = None) -> str:
        depth = 0
        if parent_id:
            row = self._conn.execute(
                "SELECT depth FROM argument_tree WHERE node_id = ?", (parent_id,)
            ).fetchone()
            if row:
                depth = row["depth"] + 1

        self._conn.execute(
            """INSERT OR REPLACE INTO argument_tree
               (node_id, run_id, parent_node_id, node_type, depth,
                content, status, confidence, source_ids,
                agent_origin, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node_id, self.run_id, parent_id, node_type, depth,
                content, status, confidence,
                json.dumps(source_ids or []),
                agent, self._now(),
                json.dumps(metadata or {}),
            )
        )
        self._conn.commit()
        return node_id

    # ── Creation methods ──────────────────────────────────────────────────

    def create_root(self, problem: str) -> str:
        """Create the root node from the research question."""
        node_id = generate_id("ROOT")
        return self._insert(
            node_id, None, "root", problem,
            status="unsupported", agent="system",
        )

    def add_question(self, parent_id: str, question_text: str,
                     question_level: str = "foundational",
                     agent: str = "grounder") -> str:
        """Add a sub-question node under a parent (usually root)."""
        node_id = generate_id("Q")
        return self._insert(
            node_id, parent_id, "question", question_text,
            agent=agent,
            metadata={"question_level": question_level},
        )

    def add_claim(self, parent_question_id: str, claim_text: str,
                  confidence: float = 0.5, agent: str = "grounder",
                  source_ids: list = None) -> str:
        """Add a claim node under a question."""
        node_id = generate_id("CLM")
        return self._insert(
            node_id, parent_question_id, "claim", claim_text,
            status="supported" if source_ids else "unsupported",
            confidence=confidence, source_ids=source_ids,
            agent=agent,
        )

    def add_evidence(self, parent_claim_id: str, source_id: str,
                     evidence_type: str = "paper",
                     relationship: str = "supports",
                     snippet: str = "",
                     agent: str = "grounder",
                     metadata: dict = None) -> str:
        """
        Add an evidence node under a claim. Links to a source_id in the
        sources table.

        evidence_type: paper, book, report, legal_document, court_decision,
                       news_article, archival, testimony, resolution, dataset, other
        relationship: supports, establishes, illustrates, partially_supports,
                      contradicts, qualifies, extends
        """
        node_id = generate_id("EV")
        meta = metadata or {}
        meta["evidence_type"] = evidence_type
        meta["relationship"] = relationship
        meta["snippet"] = snippet[:500]
        return self._insert(
            node_id, parent_claim_id, "evidence", f"[{evidence_type}] {snippet[:200]}",
            status="supported", confidence=0.8,
            source_ids=[source_id], agent=agent,
            metadata=meta,
        )

    def add_bridge(self, from_node_id: str, to_node_id: str,
                   source_id: str, bridge_type: str = "temporal",
                   description: str = "",
                   agent: str = "social") -> str:
        """
        Add a bridge node connecting two distant parts of the tree.
        bridge_type: temporal (fills a time gap), disciplinary (connects fields),
                     conceptual (connects ideas), methodological (connects methods)
        """
        node_id = generate_id("BRG")
        return self._insert(
            node_id, from_node_id, "bridge", description or f"Bridge to {to_node_id}",
            status="bridged", confidence=0.6,
            source_ids=[source_id], agent=agent,
            metadata={
                "bridge_type": bridge_type,
                "bridges_to": to_node_id,
            },
        )

    def add_counter(self, parent_claim_id: str, counter_text: str,
                    source_id: str, agent: str = "grounder") -> str:
        """Add a counter-argument to a claim. Updates parent status to contested."""
        node_id = generate_id("CTR")
        self._insert(
            node_id, parent_claim_id, "counter", counter_text,
            status="supported", confidence=0.7,
            source_ids=[source_id], agent=agent,
        )
        # Update parent claim status to contested
        self._conn.execute(
            "UPDATE argument_tree SET status = 'contested' WHERE node_id = ?",
            (parent_claim_id,)
        )
        self._conn.commit()
        return node_id

    def add_historical(self, parent_id: str, content: str,
                       year: int = None, source_id: str = "",
                       agent: str = "historian") -> str:
        """Add a historical context node."""
        node_id = generate_id("HIST")
        return self._insert(
            node_id, parent_id, "historical", content,
            source_ids=[source_id] if source_id else [],
            agent=agent,
            metadata={"year": year},
        )

    def add_external(self, parent_id: str, content: str,
                     factor_type: str = "event",
                     year: int = None, source_id: str = "",
                     agent: str = "historian") -> str:
        """
        Add an external factor node — events, policies, wars, institutional
        changes that shaped the intellectual trajectory but aren't published
        academic work.

        factor_type: event, policy, war, institutional, funding, technology,
                     social_movement, legal_change, crisis
        """
        node_id = generate_id("EXT")
        return self._insert(
            node_id, parent_id, "external", content,
            source_ids=[source_id] if source_id else [],
            agent=agent,
            metadata={"factor_type": factor_type, "year": year},
        )

    def add_audit_note(self, target_node_id: str, assessment: str,
                       new_status: str = None,
                       new_confidence: float = None,
                       agent: str = "historian") -> str:
        """
        Add an audit note from the Historian. Optionally updates the target
        node's status and confidence.
        """
        node_id = generate_id("AUD")
        self._insert(
            node_id, target_node_id, "audit_note", assessment,
            agent=agent,
            metadata={
                "audited_node": target_node_id,
                "previous_status": self._get_field(target_node_id, "status"),
                "previous_confidence": self._get_field(target_node_id, "confidence"),
            },
        )

        # Update the target node if new values provided
        if new_status:
            self._conn.execute(
                "UPDATE argument_tree SET status = ? WHERE node_id = ?",
                (new_status, target_node_id)
            )
        if new_confidence is not None:
            self._conn.execute(
                "UPDATE argument_tree SET confidence = ? WHERE node_id = ?",
                (new_confidence, target_node_id)
            )
        self._conn.commit()
        return node_id

    # ── Update methods ────────────────────────────────────────────────────

    def update_status(self, node_id: str, status: str):
        """Update a node's status."""
        self._conn.execute(
            "UPDATE argument_tree SET status = ? WHERE node_id = ?",
            (status, node_id)
        )
        self._conn.commit()

    def update_confidence(self, node_id: str, confidence: float):
        """Update a node's confidence."""
        self._conn.execute(
            "UPDATE argument_tree SET confidence = ? WHERE node_id = ?",
            (confidence, node_id)
        )
        self._conn.commit()

    def add_source_to_node(self, node_id: str, source_id: str):
        """Append a source_id to an existing node."""
        row = self._conn.execute(
            "SELECT source_ids FROM argument_tree WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row:
            ids = json.loads(row["source_ids"])
            if source_id not in ids:
                ids.append(source_id)
                self._conn.execute(
                    "UPDATE argument_tree SET source_ids = ? WHERE node_id = ?",
                    (json.dumps(ids), node_id)
                )
                self._conn.commit()

    # ── Query methods ─────────────────────────────────────────────────────

    def _get_field(self, node_id: str, field: str):
        row = self._conn.execute(
            f"SELECT {field} FROM argument_tree WHERE node_id = ?", (node_id,)
        ).fetchone()
        return row[0] if row else None

    def get_node(self, node_id: str) -> Optional[dict]:
        """Get a single node as a dict."""
        row = self._conn.execute(
            "SELECT * FROM argument_tree WHERE node_id = ?", (node_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_children(self, node_id: str) -> list[dict]:
        """Get all direct children of a node."""
        rows = self._conn.execute(
            "SELECT * FROM argument_tree WHERE parent_node_id = ? ORDER BY created_at",
            (node_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_tree(self) -> dict:
        """
        Get the full tree as a nested dict. Efficient: one query, then
        build in memory.
        """
        rows = self._conn.execute(
            "SELECT * FROM argument_tree WHERE run_id = ? ORDER BY depth, created_at",
            (self.run_id,)
        ).fetchall()

        nodes = {r["node_id"]: dict(r) for r in rows}
        for nid, node in nodes.items():
            node["children"] = []
            node["source_ids"] = json.loads(node.get("source_ids", "[]"))
            node["metadata"] = json.loads(node.get("metadata", "{}"))

        root = None
        for nid, node in nodes.items():
            parent = node.get("parent_node_id")
            if parent and parent in nodes:
                nodes[parent]["children"].append(node)
            elif node["node_type"] == "root":
                root = node

        return root or {}

    def get_branch(self, node_id: str) -> dict:
        """Get a subtree starting from a specific node."""
        node = self.get_node(node_id)
        if not node:
            return {}
        node["source_ids"] = json.loads(node.get("source_ids", "[]"))
        node["metadata"] = json.loads(node.get("metadata", "{}"))
        node["children"] = []

        children = self.get_children(node_id)
        for child in children:
            child_branch = self.get_branch(child["node_id"])
            if child_branch:
                node["children"].append(child_branch)

        return node

    def get_nodes_by_type(self, node_type: str) -> list[dict]:
        """Get all nodes of a specific type."""
        rows = self._conn.execute(
            "SELECT * FROM argument_tree WHERE run_id = ? AND node_type = ? ORDER BY created_at",
            (self.run_id, node_type)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_source_ids(self) -> list[str]:
        """Get all unique source_ids referenced in the tree."""
        rows = self._conn.execute(
            "SELECT source_ids FROM argument_tree WHERE run_id = ?",
            (self.run_id,)
        ).fetchall()
        all_ids = set()
        for row in rows:
            ids = json.loads(row["source_ids"])
            all_ids.update(ids)
        return sorted(all_ids - {""})

    # ── Analysis methods ──────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Summary statistics of the tree."""
        rows = self._conn.execute(
            "SELECT node_type, COUNT(*) as cnt FROM argument_tree WHERE run_id = ? GROUP BY node_type",
            (self.run_id,)
        ).fetchall()
        type_counts = {r["node_type"]: r["cnt"] for r in rows}

        status_rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM argument_tree WHERE run_id = ? AND node_type = 'claim' GROUP BY status",
            (self.run_id,)
        ).fetchall()
        claim_statuses = {r["status"]: r["cnt"] for r in status_rows}

        total = sum(type_counts.values())
        return {
            "total_nodes": total,
            "by_type": type_counts,
            "claim_statuses": claim_statuses,
            "unique_sources": len(self.get_all_source_ids()),
        }

    def find_gaps(self) -> list[dict]:
        """
        Find structural gaps in the tree:
        - Questions with no claims
        - Claims with no evidence
        - Claims with low confidence
        """
        gaps = []

        # Questions with no claims
        questions = self.get_nodes_by_type("question")
        for q in questions:
            children = self.get_children(q["node_id"])
            claims = [c for c in children if c["node_type"] == "claim"]
            if not claims:
                gaps.append({
                    "gap_type": "unanswered_question",
                    "node_id": q["node_id"],
                    "content": q["content"],
                })

        # Claims with no evidence
        claims = self.get_nodes_by_type("claim")
        for c in claims:
            children = self.get_children(c["node_id"])
            evidence = [ch for ch in children if ch["node_type"] == "evidence"]
            if not evidence:
                gaps.append({
                    "gap_type": "unsupported_claim",
                    "node_id": c["node_id"],
                    "content": c["content"],
                })

        # Claims with low confidence
        for c in claims:
            if c["confidence"] and c["confidence"] < 0.3:
                gaps.append({
                    "gap_type": "weak_claim",
                    "node_id": c["node_id"],
                    "content": c["content"],
                    "confidence": c["confidence"],
                })

        return gaps

    def find_bridge_needs(self, min_gap_years: int = 15) -> list[dict]:
        """
        Find pairs of evidence nodes under the same question that have a
        large temporal gap — these need bridge papers.
        """
        needs = []
        questions = self.get_nodes_by_type("question")

        for q in questions:
            # Collect all evidence years under this question
            evidence_years = []
            claims = self.get_children(q["node_id"])
            for claim in claims:
                if claim["node_type"] != "claim":
                    continue
                evidences = self.get_children(claim["node_id"])
                for ev in evidences:
                    meta = json.loads(ev.get("metadata", "{}"))
                    # Try to get year from linked source
                    source_ids = json.loads(ev.get("source_ids", "[]"))
                    if source_ids:
                        # We'd need to query sources table for year
                        # For now use metadata if available
                        year = meta.get("year")
                        if year:
                            evidence_years.append((year, ev["node_id"], claim["node_id"]))

            if len(evidence_years) < 2:
                continue

            evidence_years.sort()
            for i in range(len(evidence_years) - 1):
                y1, ev1, cl1 = evidence_years[i]
                y2, ev2, cl2 = evidence_years[i + 1]
                if y2 - y1 >= min_gap_years:
                    needs.append({
                        "question_id": q["node_id"],
                        "question": q["content"][:100],
                        "earlier_node": ev1,
                        "earlier_year": y1,
                        "later_node": ev2,
                        "later_year": y2,
                        "gap_years": y2 - y1,
                    })

        return needs

    # ── Context building for LLM prompts ──────────────────────────────────

    def to_context(self, max_depth: int = 4, include_evidence: bool = True) -> str:
        """
        Serialize the tree into a structured text context for LLM prompts.
        """
        tree = self.get_tree()
        if not tree:
            return "(empty tree)"

        lines = [f"ARGUMENT TREE — {tree.get('content', '')}"]
        stats = self.get_stats()
        lines.append(f"Stats: {stats['total_nodes']} nodes, "
                     f"{stats['unique_sources']} sources, "
                     f"claims: {stats.get('claim_statuses', {})}")
        lines.append("")

        self._render_node(tree, lines, depth=0, max_depth=max_depth,
                         include_evidence=include_evidence)
        return "\n".join(lines)

    def _render_node(self, node: dict, lines: list, depth: int,
                     max_depth: int, include_evidence: bool):
        if depth > max_depth:
            return

        indent = "  " * depth
        ntype = node.get("node_type", "?")
        status = node.get("status", "?")
        conf = node.get("confidence", 0)
        content = node.get("content", "")[:200]

        if ntype == "root":
            lines.append(f"{indent}ROOT: {content}")
        elif ntype == "question":
            lines.append(f"{indent}Q: {content}")
        elif ntype == "claim":
            lines.append(f"{indent}CLAIM [{status}] (conf:{conf:.0%}): {content}")
        elif ntype == "evidence" and include_evidence:
            meta = node.get("metadata", {})
            ev_type = meta.get("evidence_type", "?")
            rel = meta.get("relationship", "?")
            src_ids = node.get("source_ids", [])
            lines.append(f"{indent}EVIDENCE [{ev_type}] ({rel}): {content} | sources: {src_ids}")
        elif ntype == "bridge":
            meta = node.get("metadata", {})
            lines.append(f"{indent}BRIDGE [{meta.get('bridge_type','')}] → {meta.get('bridges_to','')}: {content}")
        elif ntype == "counter":
            lines.append(f"{indent}COUNTER: {content}")
        elif ntype == "historical":
            lines.append(f"{indent}HISTORICAL: {content}")
        elif ntype == "external":
            meta = node.get("metadata", {})
            lines.append(f"{indent}EXTERNAL [{meta.get('factor_type','')}]: {content}")
        elif ntype == "audit_note":
            lines.append(f"{indent}AUDIT: {content}")

        for child in node.get("children", []):
            self._render_node(child, lines, depth + 1, max_depth, include_evidence)

    def to_reference_list(self) -> list[str]:
        """
        Walk the tree, collect all source_ids, query the sources table,
        and return a formatted reference list.
        """
        source_ids = self.get_all_source_ids()
        if not source_ids:
            return []

        placeholders = ",".join("?" * len(source_ids))
        rows = self._conn.execute(
            f"SELECT * FROM sources WHERE source_id IN ({placeholders})",
            source_ids
        ).fetchall()

        refs = []
        for r in rows:
            authors = json.loads(r["authors"]) if r["authors"] else []
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
            year = r["year"] or "n.d."
            title = r["title"] or "Untitled"
            doi = r["doi"] or ""
            doi_str = f" doi:{doi}" if doi else ""
            refs.append(f"{author_str} ({year}). {title}.{doi_str}")

        return sorted(refs)
