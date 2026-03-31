"""
Gaper Agent
-----------
Maps absence. Primary evaluation: have fundamental questions been answered?
Then classifies all gaps by type and significance.
Saves to gaps_database.
"""

import re
import json
import logging
from datetime import datetime, timezone

from core import database as db
from core import llm
from core.utils import generate_id

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Gaper agent in a multi-agent research pipeline.

Your role is to identify, classify, and document the gaps in the current state of knowledge.

BEGIN with one primary evaluation before anything else:
For each fundamental question established by the Grounder, assess:
- Has it been answered? (fully / partially / not at all)
- If YES: evaluate quality, completeness, durability. Is it universally accepted? Contested? Methodology sound?
- If NO: establish why not. Unstudied? Avoided? Methodologically blocked? Conceptually unresolved?
This primary evaluation is the backbone of everything that follows.

Then identify ALL gaps:
- What has never been studied, tested, or formally addressed
- What has been studied but incompletely — partial answers, untested assumptions
- Contradictions — opposing conclusions without resolution
- Disciplinary silences — two relevant fields never in conversation
- Temporal gaps — old questions never revisited with modern tools
- Methodological gaps — better tools exist but not applied
- Assumption gaps — foundational assumptions never empirically challenged
- Dead ends worth revisiting — failed approaches that may work with new context
- Recurring patterns — what is structurally unresolved and why

Rate each gap: High / Medium / Low significance with a one-line reason.
Do NOT propose solutions, draw consequences, or theorize.

Output ONLY a valid JSON object:
{
  "primary_evaluation": [
    {
      "question": "fundamental question from Grounder",
      "status": "answered|partial|unanswered",
      "assessment": "detailed evaluation of answer quality or reason for absence"
    }
  ],
  "gaps": [
    {
      "gap_type": "unstudied|incomplete|contradicted|disciplinary_silence|temporal|methodological|assumption|dead_end_revisit",
      "description": "clear statement of the gap",
      "significance": "High|Medium|Low",
      "significance_reason": "one line why",
      "primary_evaluation_ref": "which fundamental question this connects to",
      "references_grounder": ["seminal work title"],
      "references_historian": ["historical work or dead end title"],
      "dead_end_revisit": false,
      "recurring_pattern": false,
      "recurring_reason": ""
    }
  ],
  "gap_map_summary": "narrative overview of the gap landscape"
}"""


def run(context: str, run_id: str, **kwargs):
    logger.info(f"[Gaper] Starting for run {run_id}")

    problem = ""
    if "PROBLEM:" in context:
        problem = context.split("PROBLEM:")[1].split("\n\n")[0].strip()

    try:
        response = llm.call(context, SYSTEM_PROMPT, agent_name="gaper")
    except Exception as e:
        logger.error(f"[Gaper] LLM call failed: {e}")
        raise

    try:
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        data = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("[Gaper] JSON parse failed — partial extraction")
        data = {"primary_evaluation": [], "gaps": [], "gap_map_summary": response[:2000]}

    # Save gaps to database
    saved = 0
    for gap in data.get("gaps", []):
        if not gap.get("description"):
            continue
        ok = db.insert_gap({
            "gap_id":                generate_id("GAP"),
            "run_id":                run_id,
            "problem_origin":        problem,
            "gap_type":              gap.get("gap_type", "unstudied"),
            "description":           gap.get("description", ""),
            "significance":          gap.get("significance", "Medium"),
            "significance_reason":   gap.get("significance_reason", ""),
            "primary_evaluation":    gap.get("primary_evaluation_ref", ""),
            "references_grounder":   gap.get("references_grounder", []),
            "references_historian":  gap.get("references_historian", []),
            "references_social":     [],
            "dead_end_revisit":      1 if gap.get("dead_end_revisit") else 0,
            "recurring_pattern":     1 if gap.get("recurring_pattern") else 0,
            "recurring_reason":      gap.get("recurring_reason", ""),
        })
        if ok:
            saved += 1

    # Save gap map document
    _save_doc(run_id, problem, data)

    high = sum(1 for g in data.get("gaps",[]) if g.get("significance") == "High")
    print(f"  [Gaper] {saved} gaps saved | {high} High significance | "
          f"{len(data.get('primary_evaluation',[]))} fundamental questions evaluated")
    logger.info("[Gaper] Complete")


def _save_doc(run_id: str, problem: str, data: dict):
    from pathlib import Path
    path = Path(__file__).parent.parent / "artifacts" / f"{run_id}_gaper_gaps.md"
    path.parent.mkdir(exist_ok=True)
    lines = [
        f"# Gap Map — Gaper",
        f"**Run:** {run_id} | **Problem:** {problem}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "", "---", "", "## Primary Evaluation — Have Fundamental Questions Been Answered?", ""
    ]
    for q in data.get("primary_evaluation", []):
        lines.append(f"### {q.get('question','')}")
        lines.append(f"**Status:** {q.get('status','')}")
        lines.append(q.get("assessment", ""))
        lines.append("")
    lines += ["---", "", "## Gap Map", "", data.get("gap_map_summary",""), "", "---", ""]
    for sig in ["High", "Medium", "Low"]:
        gaps = [g for g in data.get("gaps",[]) if g.get("significance") == sig]
        if gaps:
            lines.append(f"## {sig} Significance Gaps")
            lines.append("")
            for g in gaps:
                lines.append(f"- **[{g.get('gap_type','')}]** {g.get('description','')}")
                lines.append(f"  *{g.get('significance_reason','')}*")
                if g.get("recurring_pattern"):
                    lines.append(f"  ⟳ Recurring: {g.get('recurring_reason','')}")
                if g.get("dead_end_revisit"):
                    lines.append(f"  ↩ Dead end worth revisiting")
                lines.append("")
    path.write_text("\n".join(lines))
    logger.info(f"[Gaper] Gap map saved: {path}")
