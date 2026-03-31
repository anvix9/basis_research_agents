"""
Context Builder
---------------
Assembles the accumulated pipeline context for each agent.
Each agent receives exactly what it needs — no more, no less.
"""

import json
import logging
from typing import Optional
from . import database as db

logger = logging.getLogger(__name__)


def _fmt(label: str, content: str) -> str:
    """Format a context section."""
    return f"\n\n=== {label.upper()} ===\n{content}"


def _sources_summary(sources: list[dict], max_items: int = 20) -> str:
    """Format a list of sources into readable text."""
    if not sources:
        return "None available."
    lines = []
    for s in sources[:max_items]:
        authors = json.loads(s.get("authors") or "[]") if s.get("authors") else []
        author_str = ", ".join(authors[:3]) + ("..." if len(authors) > 3 else "")
        lines.append(
            f"- [{s.get('year', 'n.d.')}] {s.get('title', 'Untitled')} "
            f"({author_str}) | {s.get('source_name', '')} | "
            f"{s.get('seminal_reason') or s.get('historical_reason') or s.get('relevance_reason', '')}"
        )
    return "\n".join(lines)


def _gaps_summary(gaps: list[dict]) -> str:
    if not gaps:
        return "No gaps identified yet."
    lines = []
    for g in gaps:
        lines.append(
            f"- [{g.get('gap_id')}] [{g.get('significance')}] "
            f"[{g.get('gap_type')}] {g.get('description')} "
            f"| Primary eval: {g.get('primary_evaluation')}"
        )
    return "\n".join(lines)


def _implications_summary(implications: list[dict]) -> str:
    if not implications:
        return "No implications identified yet."
    lines = []
    for i in implications:
        lines.append(
            f"- [{i.get('implication_id')}] [{i.get('strength')}] "
            f"[{i.get('implication_type')}] {i.get('implication')}"
        )
    return "\n".join(lines)


def _proposals_summary(proposals: list[dict]) -> str:
    if not proposals:
        return "No proposals yet."
    lines = []
    for p in proposals:
        lines.append(
            f"- [{p.get('proposal_id')}] [{p.get('promise_rating')}] "
            f"[{p.get('proposal_type')}] {p.get('proposal')[:200]}..."
        )
    return "\n".join(lines)


def _evaluations_summary(evaluations: list[dict]) -> str:
    if not evaluations:
        return "No evaluations yet."
    lines = []
    for e in evaluations:
        lines.append(
            f"- [{e.get('evaluation_id')}] Proposal {e.get('proposal_id')} → "
            f"[{e.get('verdict')}] {e.get('verdict_reason', '')}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context builders per agent
# ---------------------------------------------------------------------------

def for_grounder(run_id: str, problem: str, social_sources: list[dict]) -> str:
    """Context for Grounder — problem + Social intelligence."""
    ctx = f"PROBLEM:\n{problem}"
    ctx += _fmt("Social Intelligence (current sources)", _sources_summary(social_sources))
    return ctx


def for_historian(run_id: str, problem: str) -> str:
    """Context for Historian — problem + seminal works from Grounder."""
    seminal = db.get_sources_by_type("seminal", run_id)
    social  = db.get_sources_by_type("current", run_id)
    ctx  = f"PROBLEM:\n{problem}"
    ctx += _fmt("Seminal Works (from Grounder)", _sources_summary(seminal))
    ctx += _fmt("Social Intelligence (current sources)", _sources_summary(social))
    return ctx


def for_gaper(run_id: str, problem: str, break1_instructions: str = None) -> str:
    """Context for Gaper — all prior outputs."""
    seminal    = db.get_sources_by_type("seminal",    run_id)
    historical = db.get_sources_by_type("historical", run_id)
    social     = db.get_sources_by_type("current",    run_id)
    ctx  = f"PROBLEM:\n{problem}"
    ctx += _fmt("Seminal Works (Grounder)", _sources_summary(seminal))
    ctx += _fmt("Historical Map (Historian)", _sources_summary(historical))
    ctx += _fmt("Current Intelligence (Social)", _sources_summary(social))
    if break1_instructions:
        ctx += _fmt("Your Break 1 Instructions", break1_instructions)
    return ctx


def for_vision(run_id: str, problem: str, break1_instructions: str = None) -> str:
    """Context for Vision — all prior outputs + Break 1 instructions."""
    seminal    = db.get_sources_by_type("seminal",    run_id)
    historical = db.get_sources_by_type("historical", run_id)
    social     = db.get_sources_by_type("current",    run_id)
    gaps       = db.get_gaps(run_id)
    ctx  = f"PROBLEM:\n{problem}"
    ctx += _fmt("Seminal Works (Grounder)", _sources_summary(seminal))
    ctx += _fmt("Historical Map (Historian)", _sources_summary(historical))
    ctx += _fmt("Gap Map (Gaper)", _gaps_summary(gaps))
    ctx += _fmt("Current Intelligence (Social)", _sources_summary(social))
    if break1_instructions:
        ctx += _fmt("Break 1 Instructions (Human)", break1_instructions)
    return ctx


def for_theorist(run_id: str, problem: str, break1_instructions: str = None) -> str:
    """Context for Theorist — all prior outputs."""
    seminal     = db.get_sources_by_type("seminal",    run_id)
    historical  = db.get_sources_by_type("historical", run_id)
    social      = db.get_sources_by_type("current",    run_id)
    gaps        = db.get_gaps(run_id)
    implications = db.get_implications(run_id)
    ctx  = f"PROBLEM:\n{problem}"
    ctx += _fmt("Seminal Works (Grounder)", _sources_summary(seminal))
    ctx += _fmt("Historical Map (Historian)", _sources_summary(historical))
    ctx += _fmt("Gap Map (Gaper)", _gaps_summary(gaps))
    ctx += _fmt("Implications Map (Vision)", _implications_summary(implications))
    ctx += _fmt("Current Intelligence (Social)", _sources_summary(social))
    if break1_instructions:
        ctx += _fmt("Break 1 Instructions (Human)", break1_instructions)
    return ctx


def for_rude(run_id: str, problem: str, break1_instructions: str = None) -> str:
    """Context for Rude — proposals + historical dead ends + social."""
    historical = db.get_sources_by_type("historical", run_id)
    social     = db.get_sources_by_type("current",    run_id)
    proposals  = db.get_proposals(run_id)
    gaps       = db.get_gaps(run_id)
    ctx  = f"PROBLEM:\n{problem}"
    ctx += _fmt("Proposals (Theorist)", _proposals_summary(proposals))
    ctx += _fmt("Historical Dead Ends (Historian)", _sources_summary(
        [s for s in historical if s.get("phase_tag") == "dead_end"]
    ))
    ctx += _fmt("Current Intelligence (Social)", _sources_summary(social))
    ctx += _fmt("Gap Map (Gaper)", _gaps_summary(gaps))
    if break1_instructions:
        ctx += _fmt("Break 1 Instructions (Human)", break1_instructions)
    return ctx


def for_synthesizer(run_id: str, problem: str, break1_instructions: str = None) -> str:
    """Context for Synthesizer — everything."""
    seminal     = db.get_sources_by_type("seminal",    run_id)
    historical  = db.get_sources_by_type("historical", run_id)
    social      = db.get_sources_by_type("current",    run_id)
    gaps        = db.get_gaps(run_id)
    implications = db.get_implications(run_id)
    proposals   = db.get_proposals(run_id)
    evaluations = db.get_evaluations(run_id)
    ctx  = f"PROBLEM:\n{problem}"
    ctx += _fmt("Seminal Works (Grounder)", _sources_summary(seminal))
    ctx += _fmt("Historical Map (Historian)", _sources_summary(historical))
    ctx += _fmt("Current Intelligence (Social)", _sources_summary(social))
    ctx += _fmt("Gap Map (Gaper)", _gaps_summary(gaps))
    ctx += _fmt("Implications Map (Vision)", _implications_summary(implications))
    ctx += _fmt("Proposals (Theorist)", _proposals_summary(proposals))
    ctx += _fmt("Feasibility Evaluations (Rude)", _evaluations_summary(evaluations))
    if break1_instructions:
        ctx += _fmt("Break 1 Instructions (Human)", break1_instructions)
    return ctx


def for_thinker(run_id: str, problem: str, break2_instructions: str = None) -> str:
    """Context for Thinker — synthesis + full pipeline."""
    synthesis   = db.get_synthesis(run_id)
    gaps        = db.get_gaps(run_id)
    implications = db.get_implications(run_id)
    proposals   = db.get_proposals(run_id, status="feasible")
    evaluations = db.get_evaluations(run_id)
    ctx  = f"PROBLEM:\n{problem}"
    if synthesis:
        ctx += _fmt("Research Narrative (Synthesizer)", synthesis.get("full_narrative", ""))
        ctx += _fmt("Trajectory Statement", synthesis.get("trajectory_statement", ""))
        ctx += _fmt("Key Tensions", str(synthesis.get("key_tensions", "")))
    ctx += _fmt("Gap Map (Gaper)", _gaps_summary(gaps))
    ctx += _fmt("Implications Map (Vision)", _implications_summary(implications))
    ctx += _fmt("Viable Proposals (post-Rude)", _proposals_summary(proposals))
    if break2_instructions:
        ctx += _fmt("Break 2 Instructions (Human)", break2_instructions)
    return ctx


def for_scribe(
    run_id: str,
    problem: str,
    output_type: str,
    audience: str,
    break2_instructions: str = None
) -> str:
    """Context for Scribe — synthesis + directions + output spec."""
    synthesis  = db.get_synthesis(run_id)
    directions = db.get_directions(run_id)
    proposals  = db.get_proposals(run_id, status="feasible")
    gaps       = db.get_gaps(run_id, significance="High")
    ctx  = f"PROBLEM:\n{problem}"
    ctx += _fmt("Requested Output Type", output_type)
    ctx += _fmt("Intended Audience", audience)
    if synthesis:
        ctx += _fmt("Research Narrative (Synthesizer)", synthesis.get("full_narrative", ""))
        ctx += _fmt("Trajectory Statement", synthesis.get("trajectory_statement", ""))
    ctx += _fmt("New Directions (Thinker)", "\n".join(
        [f"- [{d.get('distance_rating')}] {d.get('direction')}" for d in directions]
    ))
    ctx += _fmt("Viable Proposals", _proposals_summary(proposals))
    ctx += _fmt("High Significance Gaps", _gaps_summary(gaps))
    if break2_instructions:
        ctx += _fmt("Break 2 Instructions (Human)", break2_instructions)
    return ctx
