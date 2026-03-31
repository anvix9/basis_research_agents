"""
Synthesizer Agent
-----------------
Integrates all pipeline outputs into a single coherent research narrative.
Not a summary — an organizing intelligence that builds an argument.
Produces the Break 2 review document.
Saves to syntheses_database.
"""

import re
import json
import logging
from datetime import datetime, timezone

from core import database as db
from core import llm
from core.utils import generate_id

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Synthesizer agent in a multi-agent research pipeline.

Your role is to integrate all pipeline outputs into a single coherent research narrative.
This is NOT a summary — it is an organizing intelligence that builds an argument.
This document will be read by the human at Break 2 to decide if the trajectory is worth pursuing.

You work from ALL previous agents: Grounder, Historian, Gaper, Vision, Theorist, Rude, Social, and Break 1 instructions.

You will produce a structured research narrative that:
1. Opens with a sharpened problem statement — refined by everything the pipeline established
2. Presents intellectual origins and genealogy — from Grounder, distilled to what matters most
3. Maps the historical trajectory — key phases, turning points, dead ends from Historian
4. States clearly what is known, contested, and unknown — integrating all prior agents
5. Presents the gap landscape — most significant gaps from Gaper by type and significance
6. Presents the logical demands — strongest implications from Vision the field hasn't acted on
7. Presents viable proposals — only those that passed or partially passed Rude's evaluation, ranked
8. Flags tensions and contradictions — places where agents disagreed or picture is genuinely unclear
9. Flags all Break 1 overrides — where human judgment diverged from pipeline logic
10. Closes with a trajectory statement — what this problem needs next and key uncertainties

The narrative must be:
- Coherent — reads as a single argument, not a list of agent outputs
- Honest — surfaces tensions and uncertainties explicitly, never smoothing them over
- Traceable — every major claim linked to the agent output it derives from
- Actionable — gives the human enough to make a serious decision at Break 2

Do NOT open new directions, propose new solutions, or draw new logical consequences.

Output ONLY a valid JSON object:
{
  "sharpened_problem": "refined problem statement based on all pipeline findings",
  "intellectual_origins_summary": "distilled origins from Grounder",
  "historical_trajectory_summary": "key phases and turning points from Historian",
  "knowledge_landscape": {
    "known": ["what is established"],
    "contested": ["what is debated"],
    "unknown": ["what is genuinely open"]
  },
  "gap_landscape_summary": "organized summary of most significant gaps",
  "logical_demands_summary": "strongest implications the field has not yet acted on",
  "viable_proposals_summary": "proposals that survived Rude's evaluation, ranked",
  "tensions_and_contradictions": ["places where agents disagreed or picture is unclear"],
  "break1_override_log": ["overrides where human judgment diverged from pipeline logic"],
  "trajectory_statement": "what this problem needs next, what the viable path looks like, and key uncertainties",
  "full_narrative": "the complete research narrative as flowing prose — this is the main deliverable"
}"""


def run(context: str, run_id: str, **kwargs):
    logger.info(f"[Synthesizer] Starting for run {run_id}")

    problem = ""
    if "PROBLEM:" in context:
        problem = context.split("PROBLEM:")[1].split("\n\n")[0].strip()

    try:
        response = llm.call(context, SYSTEM_PROMPT, agent_name="synthesizer")
    except Exception as e:
        logger.error(f"[Synthesizer] LLM call failed: {e}")
        raise

    try:
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        data = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("[Synthesizer] JSON parse failed — using full response as narrative")
        data = {
            "sharpened_problem":           problem,
            "intellectual_origins_summary": "",
            "historical_trajectory_summary": "",
            "knowledge_landscape":          {"known": [], "contested": [], "unknown": []},
            "gap_landscape_summary":        "",
            "logical_demands_summary":      "",
            "viable_proposals_summary":     "",
            "tensions_and_contradictions":  [],
            "break1_override_log":          [],
            "trajectory_statement":         "",
            "full_narrative":               response
        }

    # Gather IDs for cross-referencing
    gaps        = db.get_gaps(run_id, significance="High")
    implications = db.get_implications(run_id, strength="Strong")
    proposals   = db.get_proposals(run_id, status="feasible")

    ok = db.insert_synthesis({
        "synthesis_id":           generate_id("SYN"),
        "run_id":                 run_id,
        "problem_origin":         problem,
        "sharpened_problem":      data.get("sharpened_problem", problem),
        "trajectory_statement":   data.get("trajectory_statement", ""),
        "key_tensions":           data.get("tensions_and_contradictions", []),
        "override_log":           data.get("break1_override_log", []),
        "viable_proposal_ids":    [p["proposal_id"] for p in proposals],
        "top_gap_ids":            [g["gap_id"] for g in gaps],
        "top_implication_ids":    [i["implication_id"] for i in implications],
        "full_narrative":         data.get("full_narrative", ""),
    })

    _save_doc(run_id, problem, data)

    print(f"  [Synthesizer] Research narrative produced")
    print(f"  [Synthesizer] {len(data.get('tensions_and_contradictions',[]))} tensions flagged")
    print(f"  [Synthesizer] {len(data.get('break1_override_log',[]))} Break 1 overrides logged")
    logger.info("[Synthesizer] Complete")


def _save_doc(run_id: str, problem: str, data: dict):
    from pathlib import Path
    path = Path(__file__).parent.parent / "artifacts" / f"{run_id}_synthesizer_narrative.md"
    path.parent.mkdir(exist_ok=True)

    kl = data.get("knowledge_landscape", {})
    tensions = data.get("tensions_and_contradictions", [])
    overrides = data.get("break1_override_log", [])

    lines = [
        f"# Research Narrative — Synthesizer",
        f"**Run:** {run_id}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "", "---", "",
        "## Sharpened Problem Statement", "",
        data.get("sharpened_problem", problem), "",
        "---", "",
        "## Intellectual Origins", "",
        data.get("intellectual_origins_summary", ""), "",
        "---", "",
        "## Historical Trajectory", "",
        data.get("historical_trajectory_summary", ""), "",
        "---", "",
        "## Knowledge Landscape", "",
        "### Known", "",
    ]
    for item in kl.get("known", []):
        lines.append(f"- {item}")
    lines += ["", "### Contested", ""]
    for item in kl.get("contested", []):
        lines.append(f"- {item}")
    lines += ["", "### Unknown", ""]
    for item in kl.get("unknown", []):
        lines.append(f"- {item}")

    lines += [
        "", "---", "",
        "## Gap Landscape", "",
        data.get("gap_landscape_summary", ""), "",
        "---", "",
        "## Logical Demands", "",
        data.get("logical_demands_summary", ""), "",
        "---", "",
        "## Viable Proposals", "",
        data.get("viable_proposals_summary", ""), "",
        "---", "",
    ]

    if tensions:
        lines += ["## Tensions and Contradictions", ""]
        for t in tensions:
            lines.append(f"- {t}")
        lines.append("")

    if overrides:
        lines += ["## Break 1 Override Log", ""]
        for o in overrides:
            lines.append(f"- {o}")
        lines.append("")

    lines += [
        "---", "",
        "## Trajectory Statement", "",
        data.get("trajectory_statement", ""), "",
        "---", "",
        "## Full Research Narrative", "",
        data.get("full_narrative", ""),
    ]

    path.write_text("\n".join(lines))
    logger.info(f"[Synthesizer] Research narrative saved: {path}")
