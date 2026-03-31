"""
Rude Agent
----------
Evaluates feasibility of Theorist's proposals strictly on empirical evidence.
Does not care about logical elegance — only demonstrated facts.
Saves to evaluations_database.
"""

import re
import json
import logging
from datetime import datetime, timezone

from core import database as db
from core import llm
from core.utils import generate_id

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Rude agent in a multi-agent research pipeline.

Your role is to evaluate the feasibility of every proposal from the Theorist — strictly on empirical evidence, previous experimental results, and demonstrated facts.

You do NOT care about logical elegance, novelty, or how well a proposal fits the theoretical picture.
You ask one question only: does this actually hold up against what has been empirically demonstrated?

For every proposal you will:
1. Assess whether the core mechanism has been demonstrated empirically — not theorized, actually tested and shown
2. Cross-reference historical dead ends — if this was tried and failed, state exactly what happened and whether the revival justification is empirically sound
3. Cross-reference current intelligence — what do the most recent experimental results say?
4. Identify the weakest empirical link — the assumption or requirement with the least experimental support
5. Assess resource and methodological feasibility — is this doable with existing tools and knowledge?
6. Flag proposals that are logically elegant but empirically unsupported — logic alone is NOT evidence
7. Identify what specific experiments or evidence would make a rejected proposal viable

Every evaluation must reference specific results, studies, or demonstrated facts.
Be specific about failure modes — not just "this won't work" but exactly where and why.

Do NOT propose alternatives, draw logical consequences, or open new directions.

Output ONLY a valid JSON object:
{
  "evaluations": [
    {
      "proposal_ref": "first ~100 chars of the proposal being evaluated",
      "verdict": "feasible|partially_feasible|unfeasible|insufficient_evidence",
      "verdict_reason": "detailed empirical justification",
      "weakest_empirical_link": "the assumption or requirement with least experimental support",
      "dead_end_references": ["relevant failed attempts from history"],
      "social_evidence_references": ["relevant current results"],
      "evidence_to_change_verdict": "what specific evidence or experiments would change this verdict"
    }
  ],
  "overall_ranking": "narrative ranking of proposals by empirical solidity",
  "feasibility_summary": "overall assessment of the proposal landscape"
}"""


def run(context: str, run_id: str, **kwargs):
    logger.info(f"[Rude] Starting for run {run_id}")

    problem = ""
    if "PROBLEM:" in context:
        problem = context.split("PROBLEM:")[1].split("\n\n")[0].strip()

    # Get proposals for cross-referencing
    proposals = db.get_proposals(run_id)

    try:
        response = llm.call(context, SYSTEM_PROMPT, agent_name="rude")
    except Exception as e:
        logger.error(f"[Rude] LLM call failed: {e}")
        raise

    try:
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        data = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("[Rude] JSON parse failed — partial extraction")
        data = {"evaluations": [], "overall_ranking": response[:2000], "feasibility_summary": ""}

    # Match evaluations to proposals by text similarity and save
    saved = 0
    evals = data.get("evaluations", [])
    for i, ev in enumerate(evals):
        # Match to proposal — use index or text matching
        matched_proposal = None
        prop_ref = ev.get("proposal_ref", "").lower()[:80]
        for prop in proposals:
            if prop.get("proposal", "").lower()[:80] in prop_ref or \
               prop_ref in prop.get("proposal", "").lower()[:80]:
                matched_proposal = prop
                break
        # Fallback: use positional match
        if not matched_proposal and i < len(proposals):
            matched_proposal = proposals[i]

        proposal_id = matched_proposal["proposal_id"] if matched_proposal else f"UNKNOWN-{i}"

        ok = db.insert_evaluation({
            "evaluation_id":               generate_id("EVAL"),
            "run_id":                      run_id,
            "proposal_id":                 proposal_id,
            "problem_origin":              problem,
            "verdict":                     ev.get("verdict", "insufficient_evidence"),
            "verdict_reason":              ev.get("verdict_reason", ""),
            "weakest_empirical_link":      ev.get("weakest_empirical_link", ""),
            "dead_end_references":         ev.get("dead_end_references", []),
            "social_evidence_references":  ev.get("social_evidence_references", []),
            "evidence_to_change_verdict":  ev.get("evidence_to_change_verdict", ""),
        })
        if ok:
            saved += 1

    _save_doc(run_id, problem, data, proposals)

    feasible = sum(1 for e in evals if e.get("verdict") in ["feasible","partially_feasible"])
    print(f"  [Rude] {saved} evaluations saved | {feasible}/{len(evals)} feasible or partially feasible")
    logger.info("[Rude] Complete")


def _save_doc(run_id: str, problem: str, data: dict, proposals: list):
    from pathlib import Path
    path = Path(__file__).parent.parent / "artifacts" / f"{run_id}_rude_evaluations.md"
    path.parent.mkdir(exist_ok=True)
    lines = [
        f"# Feasibility Report — Rude",
        f"**Run:** {run_id} | **Problem:** {problem}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "", "---", "", "## Feasibility Summary", "", data.get("feasibility_summary",""), "",
        "---", "", "## Overall Ranking", "", data.get("overall_ranking",""), "",
        "---", "", "## Evaluations", ""
    ]
    for verdict in ["feasible","partially_feasible","insufficient_evidence","unfeasible"]:
        evs = [e for e in data.get("evaluations",[]) if e.get("verdict") == verdict]
        if evs:
            label = {"feasible": "✅ Feasible", "partially_feasible": "⚠ Partially Feasible",
                     "unfeasible": "❌ Unfeasible", "insufficient_evidence": "❓ Insufficient Evidence"}
            lines.append(f"### {label.get(verdict, verdict)}")
            lines.append("")
            for e in evs:
                lines.append(f"- **Proposal:** {e.get('proposal_ref','')[:120]}")
                lines.append(f"  **Verdict reason:** {e.get('verdict_reason','')}")
                lines.append(f"  **Weakest link:** {e.get('weakest_empirical_link','')}")
                lines.append(f"  **To change verdict:** {e.get('evidence_to_change_verdict','')}")
                lines.append("")
    path.write_text("\n".join(lines))
    logger.info(f"[Rude] Feasibility report saved: {path}")
