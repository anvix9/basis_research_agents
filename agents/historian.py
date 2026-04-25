"""
Historian Agent
---------------
Builds the chronological map of the problem.
Starts from Grounder's seminal works, extends forward in time.
Tracks phases, turning points, dead ends, key actors, methods evolution.
Saves to historical_database with phase_tags.
"""

import re
import json
import time
import logging
import requests
from datetime import datetime, timezone

from core import database as db
from core import llm
from core.utils import generate_id

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Historian agent in a multi-agent research pipeline.

Your role is to build the complete chronological map of the problem — from its intellectual origins to its current state.

You begin from the seminal works provided (from Grounder) and extend forward in time.

You will:
1. Identify the major phases of the problem's evolution and what drove transitions
2. Map key actors — researchers, institutions, schools of thought per phase
3. Document methods evolution — how approaches and tools changed across time
4. Flag turning points — moments the field changed direction
5. Track citation velocity signals — sudden shifts indicate breakthroughs
6. Track keyword evolution — terminology shifts signal paradigm changes
7. Document failures and dead ends explicitly — what was tried, why it failed, what was learned
8. Identify recurring patterns — problems that keep resurfacing
9. Note where current intelligence represents continuity or a break from historical trajectory

CRITICAL: Failures and dead ends are equal data to successes. Document them thoroughly.

Do NOT identify gaps, propose solutions, or draw logical consequences.

Output ONLY a valid JSON object:
{
  "phases": [
    {
      "name": "phase name",
      "period": "e.g. 1950-1970",
      "description": "what characterized this phase",
      "transition_driver": "what caused shift to next phase"
    }
  ],
  "historical_works": [
    {
      "title": "full title",
      "authors": ["Author Name"],
      "year": 1980,
      "source": "source name",
      "doi": "",
      "abstract": "brief description",
      "active_link": "url",
      "historical_reason": "one line — what it changed or represented",
      "phase_tag": "breakthrough|paradigm_shift|dead_end|methodological_evolution|recurring_pattern|turning_point",
      "theme_tags": ["theme1"],
      "intersection_tags": ["theme1 x theme2"]
    }
  ],
  "key_actors": [
    {
      "name": "Person or Institution",
      "phase": "phase name",
      "contribution": "what they contributed"
    }
  ],
  "dead_ends": [
    {
      "approach": "what was tried",
      "period": "when",
      "actors": ["who tried it"],
      "failure_reason": "why it failed",
      "lesson": "what was learned"
    }
  ],
  "recurring_patterns": [
    {
      "pattern": "description of recurring question or problem",
      "appearances": ["era1", "era2"],
      "structural_reason": "why it keeps resurfacing"
    }
  ],
  "methods_evolution": "narrative of how approaches and tools changed over time",
  "trajectory_vs_current": "assessment of whether current intelligence is continuity or break"
}"""


def _verify_link(url: str) -> str:
    if not url:
        return "dead"
    try:
        resp = requests.head(url, timeout=8, allow_redirects=True,
                             headers={"User-Agent": "PipelineResearchBot/1.0"})
        return "active" if resp.status_code < 400 else "dead"
    except Exception:
        return "dead"


def run(context: str, run_id: str, **kwargs):
    logger.info(f"[Historian] Starting for run {run_id}")

    problem = ""
    if "PROBLEM:" in context:
        problem = context.split("PROBLEM:")[1].split("\n\n")[0].strip()

    # ── Job 1: Tree Audit ─────────────────────────────────────────────────
    from core.argument_tree import TreeBuilder
    tree = TreeBuilder(run_id)
    stats = tree.get_stats()

    if stats.get("total_nodes", 0) > 0:
        print("  [Historian] Job 1 — auditing argument tree...")
        gaps = tree.find_gaps()
        claims = tree.get_nodes_by_type("claim")

        # Assess each claim's solidity
        audited = 0
        for claim in claims:
            children = tree.get_children(claim["node_id"])
            evidence_nodes = [c for c in children if c["node_type"] == "evidence"]
            counter_nodes = [c for c in children if c["node_type"] == "counter"]

            if len(evidence_nodes) >= 2 and not counter_nodes:
                tree.add_audit_note(
                    claim["node_id"],
                    f"Well-supported: {len(evidence_nodes)} evidence nodes, no counter-arguments",
                    new_status="solid", new_confidence=0.85,
                    agent="historian",
                )
            elif counter_nodes:
                tree.add_audit_note(
                    claim["node_id"],
                    f"Contested: {len(evidence_nodes)} evidence, {len(counter_nodes)} counter-arguments",
                    new_status="contested",
                    agent="historian",
                )
            elif len(evidence_nodes) == 1:
                tree.add_audit_note(
                    claim["node_id"],
                    "Single evidence source — needs corroboration",
                    new_status="weak", new_confidence=0.4,
                    agent="historian",
                )
            else:
                tree.add_audit_note(
                    claim["node_id"],
                    "No evidence found — unsupported",
                    new_status="unsupported", new_confidence=0.1,
                    agent="historian",
                )
            audited += 1

        unanswered = len([g for g in gaps if g["gap_type"] == "unanswered_question"])
        unsupported = len([g for g in gaps if g["gap_type"] == "unsupported_claim"])
        weak_claims = len([g for g in gaps if g["gap_type"] == "weak_claim"])

        print(f"  [Historian] Audit: {audited} claims assessed | "
              f"{unanswered} unanswered questions | {unsupported} unsupported claims | "
              f"{weak_claims} weak claims")
    else:
        print("  [Historian] No tree found — skipping audit")

    # ── Job 2: Historical search (existing logic) ──────────────────────────
    print("  [Historian] Job 2 — building historical map...")

    # Enrich context with tree summary for better historical search
    enriched_context = context
    if stats.get("total_nodes", 0) > 0:
        tree_ctx = tree.to_context(max_depth=2, include_evidence=False)
        enriched_context += f"\n\n=== ARGUMENT TREE (for historical context) ===\n{tree_ctx}"

    try:
        response = llm.call(enriched_context, SYSTEM_PROMPT, agent_name="historian")
    except Exception as e:
        logger.error(f"[Historian] LLM call failed: {e}")
        tree.close()
        raise

    try:
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        data = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("[Historian] JSON parse failed — partial extraction")
        data = {"phases": [], "historical_works": [], "key_actors": [],
                "dead_ends": [], "recurring_patterns": [],
                "methods_evolution": response[:2000], "trajectory_vs_current": ""}

    # Save historical works to DB
    saved = 0
    source_id_map = {}  # title → source_id for tree linkage
    for work in data.get("historical_works", []):
        if not work.get("title"):
            continue
        link_status = _verify_link(work.get("active_link", ""))
        source_id = generate_id("HIST")
        ok = db.upsert_source({
            "source_id":          source_id,
            "title":              work.get("title", ""),
            "authors":            work.get("authors", []),
            "year":               work.get("year"),
            "source_name":        work.get("source", "historian"),
            "doi":                work.get("doi", ""),
            "abstract":           work.get("abstract", ""),
            "active_link":        work.get("active_link", ""),
            "theme_tags":         work.get("theme_tags", []),
            "type":               "historical",
            "historical_reason":  work.get("historical_reason", ""),
            "phase_tag":          work.get("phase_tag", ""),
            "intersection_tags":  work.get("intersection_tags", []),
            "added_by":           "Historian",
            "date_collected":     datetime.now(timezone.utc).isoformat(),
            "last_checked":       datetime.now(timezone.utc).isoformat(),
            "link_status":        link_status,
            "run_id":             run_id,
        })
        if ok:
            saved += 1
            source_id_map[work.get("title", "")] = source_id
        time.sleep(0.1)

    # ── Job 3: Extend tree with historical + external nodes ────────────────
    if stats.get("total_nodes", 0) > 0:
        print("  [Historian] Job 3 — extending tree with historical context...")

        # Add historical works to tree
        questions = tree.get_nodes_by_type("question")
        root_node = tree.get_nodes_by_type("root")
        parent_for_hist = root_node[0]["node_id"] if root_node else None

        for work in data.get("historical_works", []):
            title = work.get("title", "")
            sid = source_id_map.get(title, "")
            if parent_for_hist:
                tree.add_historical(
                    parent_for_hist,
                    f"[{work.get('year','?')}] {title} — {work.get('historical_reason','')}",
                    year=work.get("year"),
                    source_id=sid,
                    agent="historian",
                )

        # Add external factors from dead ends and phase transitions
        for phase in data.get("phases", []):
            driver = phase.get("transition_driver", "")
            if driver and parent_for_hist:
                tree.add_external(
                    parent_for_hist,
                    f"Phase transition: {phase.get('name','')} — {driver}",
                    factor_type="institutional",
                    agent="historian",
                )

        for dead_end in data.get("dead_ends", []):
            if parent_for_hist:
                tree.add_external(
                    parent_for_hist,
                    f"Dead end: {dead_end.get('approach','')} — {dead_end.get('failure_reason','')}",
                    factor_type="institutional",
                    agent="historian",
                )

        final_stats = tree.get_stats()
        print(f"  [Historian] Tree extended: {final_stats['total_nodes']} nodes total")

    tree.close()

    # Save document
    _save_doc(run_id, problem, data)

    print(f"  [Historian] {saved} historical works saved | "
          f"{len(data.get('phases',[]))} phases | "
          f"{len(data.get('dead_ends',[]))} dead ends documented")
    logger.info("[Historian] Complete")


def _save_doc(run_id: str, problem: str, data: dict):
    from pathlib import Path
    path = Path(__file__).parent.parent / "artifacts" / f"{run_id}_historian_map.md"
    path.parent.mkdir(exist_ok=True)
    lines = [
        f"# Historical Map — Historian",
        f"**Run:** {run_id} | **Problem:** {problem}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "", "---", "", "## Phases", ""
    ]
    for p in data.get("phases", []):
        lines.append(f"### {p.get('name')} ({p.get('period','')})")
        lines.append(p.get("description", ""))
        lines.append(f"*Transition driver: {p.get('transition_driver','')}*")
        lines.append("")
    lines += ["---", "", "## Key Actors", ""]
    for a in data.get("key_actors", []):
        lines.append(f"- **{a.get('name')}** [{a.get('phase','')}]: {a.get('contribution','')}")
    lines += ["", "---", "", "## Dead Ends", ""]
    for d in data.get("dead_ends", []):
        lines.append(f"- **{d.get('approach')}** ({d.get('period','')})")
        lines.append(f"  Reason: {d.get('failure_reason','')}")
        lines.append(f"  Lesson: {d.get('lesson','')}")
    lines += ["", "---", "", "## Recurring Patterns", ""]
    for r in data.get("recurring_patterns", []):
        lines.append(f"- **{r.get('pattern','')}**")
        lines.append(f"  Appearances: {', '.join(r.get('appearances',[]))}")
        lines.append(f"  Why: {r.get('structural_reason','')}")
    lines += ["", "---", "", "## Methods Evolution", "", data.get("methods_evolution",""), ""]
    lines += ["", "---", "", "## Historical Works", ""]
    for w in data.get("historical_works", []):
        lines.append(f"- [{w.get('year','n.d.')}] **{w.get('title','')}** [{w.get('phase_tag','')}]")
        lines.append(f"  {w.get('historical_reason','')}")
    path.write_text("\n".join(lines))
    logger.info(f"[Historian] Historical map saved: {path}")
