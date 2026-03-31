"""
Break Mechanics
---------------
Hard stops in the pipeline.
Each break:
  1. Produces a structured review document (saved to artifacts/)
  2. Waits for the human to upload an instruction document
  3. Reads the instruction document
  4. Checks for contradictions with prior agent outputs
  5. Returns instructions to inject into the next agent's context
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from . import database as db

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Document production helpers
# ---------------------------------------------------------------------------

def _now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def _produce_break0_doc(run_id: str, problem: str, selected_themes: list, excluded_themes: list) -> Path:
    """Produce Break 0 review document."""
    path = ARTIFACTS_DIR / f"{run_id}_break0_review.md"
    lines = [
        f"# Break 0 — Theme Confirmation",
        f"**Run ID:** {run_id}",
        f"**Problem:** {problem}",
        f"**Generated:** {_now_str()}",
        "",
        "---",
        "",
        "## Selected Themes",
        "The following themes were matched to your problem and pulled from the database:",
        "",
    ]
    for t in selected_themes:
        lines.append(f"- **{t['theme_id']}**: {t.get('label', '')} — Keywords: {', '.join([k.get('seed','') for k in t.get('keywords', [])])}")
    lines += [
        "",
        "## Excluded Themes",
        "The following themes were excluded with reasons:",
        "",
    ]
    for t in excluded_themes:
        lines.append(f"- **{t['theme_id']}**: {t.get('reason', 'Not relevant to problem')}")
    lines += [
        "",
        "---",
        "",
        "## Your Instructions",
        "Please review the above and provide your instructions below.",
        "You can:",
        "- Confirm the selection (write: `CONFIRMED`)",
        "- Override exclusions (write: `ADD THEME: <theme_id>`)",
        "- Remove a selected theme (write: `REMOVE THEME: <theme_id>`)",
        "- Add free-form instructions for the pipeline",
        "",
        "**Your instructions:**",
        "",
    ]
    path.write_text("\n".join(lines))
    logger.info(f"Break 0 document produced: {path}")
    return path


def _produce_break1_doc(run_id: str, problem: str) -> Path:
    """Produce Break 1 review document — Grounder + Historian + Gaper outputs."""
    seminal    = db.get_sources_by_type("seminal",    run_id)
    historical = db.get_sources_by_type("historical", run_id)
    gaps       = db.get_gaps(run_id)
    path = ARTIFACTS_DIR / f"{run_id}_break1_review.md"

    lines = [
        f"# Break 1 — Ground Truth Validation",
        f"**Run ID:** {run_id}",
        f"**Problem:** {problem}",
        f"**Generated:** {_now_str()}",
        "",
        "---",
        "",
        "## Seminal Works (Grounder)",
        f"*{len(seminal)} seminal works identified.*",
        "",
    ]
    for s in seminal[:30]:
        lines.append(f"- [{s.get('year','n.d.')}] **{s.get('title','')}** — {s.get('seminal_reason','')}")

    lines += ["", "---", "", "## Historical Map (Historian)", f"*{len(historical)} historical entries.*", ""]
    for s in historical[:30]:
        lines.append(f"- [{s.get('year','n.d.')}] **{s.get('title','')}** [{s.get('phase_tag','')}] — {s.get('historical_reason','')}")

    lines += ["", "---", "", "## Gap Map (Gaper)", f"*{len(gaps)} gaps identified.*", ""]
    for g in gaps:
        lines.append(
            f"- **[{g.get('gap_id')}]** [{g.get('significance')}] [{g.get('gap_type')}]"
            f"\n  {g.get('description','')}"
            f"\n  *Primary evaluation: {g.get('primary_evaluation','')}*"
        )

    lines += [
        "",
        "---",
        "",
        "## Your Instructions",
        "Please review the above and provide your instructions.",
        "The pipeline will resume with your corrections injected into Vision.",
        "",
        "You can:",
        "- Confirm everything is correct (write: `CONFIRMED`)",
        "- Correct a gap (write: `CORRECT GAP <gap_id>: <your correction>`)",
        "- Remove a gap (write: `REMOVE GAP <gap_id>`)",
        "- Add a gap (write: `ADD GAP: <description>`)",
        "- Override a seminal work assessment (write: `OVERRIDE SEMINAL <source_id>: <your note>`)",
        "- Add free-form instructions for Vision and beyond",
        "",
        "**Your instructions:**",
        "",
    ]
    path.write_text("\n".join(lines))
    logger.info(f"Break 1 document produced: {path}")
    return path


def _produce_break2_doc(run_id: str, problem: str) -> Path:
    """Produce Break 2 review document — full synthesis for trajectory evaluation."""
    synthesis   = db.get_synthesis(run_id)
    proposals   = db.get_proposals(run_id)
    evaluations = db.get_evaluations(run_id)
    path = ARTIFACTS_DIR / f"{run_id}_break2_review.md"

    lines = [
        f"# Break 2 — Trajectory Evaluation",
        f"**Run ID:** {run_id}",
        f"**Problem:** {problem}",
        f"**Generated:** {_now_str()}",
        "",
        "---",
        "",
    ]

    if synthesis:
        lines += [
            "## Sharpened Problem Statement",
            synthesis.get("sharpened_problem", ""),
            "",
            "---",
            "",
            "## Research Narrative",
            synthesis.get("full_narrative", ""),
            "",
            "---",
            "",
            "## Trajectory Statement",
            synthesis.get("trajectory_statement", ""),
            "",
            "---",
            "",
            "## Key Tensions",
            str(synthesis.get("key_tensions", "")),
            "",
            "---",
            "",
        ]

    lines += ["## Feasibility Verdicts (Rude)", ""]
    for e in evaluations:
        p = next((p for p in proposals if p["proposal_id"] == e["proposal_id"]), {})
        lines.append(
            f"- **[{e.get('proposal_id')}]** [{e.get('verdict')}]"
            f"\n  Proposal: {p.get('proposal','')[:200]}..."
            f"\n  Reason: {e.get('verdict_reason','')}"
            f"\n  Weakest link: {e.get('weakest_empirical_link','')}"
        )

    lines += [
        "",
        "---",
        "",
        "## Your Instructions",
        "Please evaluate the trajectory and provide your instructions.",
        "The pipeline will pass your instructions to Thinker and Scribe.",
        "",
        "You can:",
        "- Confirm the trajectory (write: `CONFIRMED`)",
        "- Override a verdict (write: `OVERRIDE VERDICT <evaluation_id>: <your reasoning>`)",
        "- Request specific output types from Scribe:",
        "  `SCRIBE OUTPUT: blog_post | audience: general public`",
        "  `SCRIBE OUTPUT: paper_section | audience: specialists`",
        "  `SCRIBE OUTPUT: research_brief | audience: collaborators`",
        "- Add free-form instructions for Thinker and Scribe",
        "",
        "**Your instructions:**",
        "",
    ]
    path.write_text("\n".join(lines))
    logger.info(f"Break 2 document produced: {path}")
    return path


# ---------------------------------------------------------------------------
# Instruction ingestion
# ---------------------------------------------------------------------------

def _wait_for_instruction_file(expected_path: Path, break_name: str) -> str:
    """
    Wait for the human to upload an instruction file.
    Polls every 10 seconds. Press Ctrl+C to abort.
    """
    print(f"\n{'='*60}")
    print(f"  {break_name} — PIPELINE PAUSED")
    print(f"{'='*60}")
    print(f"\nReview document saved to:\n  {expected_path}\n")
    print("When ready:")
    print(f"  1. Open the document above")
    print(f"  2. Fill in your instructions at the bottom")
    print(f"  3. Save the file")
    print(f"  4. Press ENTER here to continue\n")
    print("Or upload a separate instruction file and enter its path.")
    print("(Press Ctrl+C to abort the pipeline)\n")

    while True:
        try:
            user_input = input("Press ENTER when ready (or enter path to instruction file): ").strip()

            if user_input == "":
                # Read from the review document itself
                if expected_path.exists():
                    content = expected_path.read_text()
                    instructions = _extract_instructions(content)
                    if instructions:
                        logger.info(f"{break_name}: Instructions read from review document")
                        return instructions
                    else:
                        print("No instructions found in the document. Please fill in the instructions section.")
                else:
                    print(f"Document not found at {expected_path}")

            else:
                # Read from a separate file
                instruction_path = Path(user_input)
                if instruction_path.exists():
                    instructions = instruction_path.read_text().strip()
                    if instructions:
                        logger.info(f"{break_name}: Instructions read from {instruction_path}")
                        return instructions
                    else:
                        print("The file is empty. Please add your instructions.")
                else:
                    print(f"File not found: {instruction_path}")

        except KeyboardInterrupt:
            print("\n\nPipeline aborted by user.")
            sys.exit(0)


def _extract_instructions(document_content: str) -> str:
    """Extract the instructions section from a review document."""
    marker = "**Your instructions:**"
    if marker in document_content:
        parts = document_content.split(marker)
        if len(parts) > 1:
            instructions = parts[-1].strip()
            return instructions if instructions else ""
    return ""


def _check_contradictions(instructions: str, run_id: str, break_num: int) -> list[str]:
    """
    Check if instructions contradict prior agent outputs.
    Returns a list of contradiction notices to inject into the next agent's context.
    """
    contradictions = []

    if break_num == 1:
        # Check gap overrides
        gaps = db.get_gaps(run_id)
        for gap in gaps:
            gap_id = gap.get("gap_id", "")
            if f"REMOVE GAP {gap_id}" in instructions:
                contradictions.append(
                    f"CONTRADICTION NOTICE: Your instruction removes {gap_id} "
                    f"which Gaper identified as a '{gap.get('significance')}' significance gap "
                    f"of type '{gap.get('gap_type')}'. "
                    f"Original: {gap.get('description','')[:100]}. "
                    f"This override is respected — downstream agents will not use this gap."
                )

    if break_num == 2:
        # Check verdict overrides
        evaluations = db.get_evaluations(run_id)
        for ev in evaluations:
            ev_id = ev.get("evaluation_id", "")
            if f"OVERRIDE VERDICT {ev_id}" in instructions:
                contradictions.append(
                    f"CONTRADICTION NOTICE: Your instruction overrides verdict for {ev_id} "
                    f"which Rude rated as '{ev.get('verdict')}'. "
                    f"Rude's reason: {ev.get('verdict_reason','')[:100]}. "
                    f"Your override is respected — Thinker and Scribe will proceed with your assessment."
                )

    return contradictions


# ---------------------------------------------------------------------------
# Public break interface
# ---------------------------------------------------------------------------

def break0(run_id: str, problem: str, selected_themes: list, excluded_themes: list) -> str:
    """
    Break 0 — Theme confirmation.
    Returns the human's instructions as a string.
    """
    doc_path = _produce_break0_doc(run_id, problem, selected_themes, excluded_themes)
    instructions = _wait_for_instruction_file(doc_path, "BREAK 0")
    contradictions = _check_contradictions(instructions, run_id, 0)
    db.mark_break_done(run_id, 0)
    if contradictions:
        logger.info(f"Break 0: {len(contradictions)} contradiction(s) noted")
    return instructions


def break1(run_id: str, problem: str) -> str:
    """
    Break 1 — Ground truth validation.
    Returns the human's instructions as a string.
    """
    doc_path = _produce_break1_doc(run_id, problem)
    instructions = _wait_for_instruction_file(doc_path, "BREAK 1")
    contradictions = _check_contradictions(instructions, run_id, 1)
    db.mark_break_done(run_id, 1)
    if contradictions:
        contradiction_log = "\n".join(contradictions)
        logger.info(f"Break 1: {len(contradictions)} contradiction(s) logged")
        instructions = instructions + f"\n\n--- CONTRADICTION LOG ---\n{contradiction_log}"
    return instructions


def break2(run_id: str, problem: str) -> str:
    """
    Break 2 — Trajectory evaluation.
    Returns the human's instructions as a string.
    """
    doc_path = _produce_break2_doc(run_id, problem)
    instructions = _wait_for_instruction_file(doc_path, "BREAK 2")
    contradictions = _check_contradictions(instructions, run_id, 2)
    db.mark_break_done(run_id, 2)
    if contradictions:
        contradiction_log = "\n".join(contradictions)
        logger.info(f"Break 2: {len(contradictions)} contradiction(s) logged")
        instructions = instructions + f"\n\n--- CONTRADICTION LOG ---\n{contradiction_log}"
    return instructions


def parse_scribe_requests(instructions: str) -> list[dict]:
    """
    Parse SCRIBE OUTPUT directives from Break 2 instructions.
    Returns list of {output_type, audience} dicts.
    
    Format: SCRIBE OUTPUT: <output_type> | audience: <audience>
    """
    requests = []
    for line in instructions.split("\n"):
        line = line.strip()
        if line.upper().startswith("SCRIBE OUTPUT:"):
            parts = line[len("SCRIBE OUTPUT:"):].strip()
            output_type = parts.split("|")[0].strip()
            audience = "general"
            if "|" in parts:
                aud_part = parts.split("|")[1]
                if "audience:" in aud_part.lower():
                    audience = aud_part.lower().replace("audience:", "").strip()
            requests.append({"output_type": output_type, "audience": audience})
    
    # Default if no SCRIBE OUTPUT directives found
    if not requests:
        requests.append({"output_type": "research_brief", "audience": "researcher"})
    
    return requests
