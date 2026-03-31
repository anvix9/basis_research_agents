"""
Vision Agent
------------
Extracts logical consequences, implications, and inferences.
First inferential agent — works after Break 1.
Saves to implications_database.
"""

import re
import json
import logging
from datetime import datetime, timezone

from core import database as db
from core import llm
from core.utils import generate_id

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Vision agent in a multi-agent research pipeline.

Your role is to extract the logical consequences, implications, and inferences that follow from the accumulated knowledge of the problem.

You work from the Grounder's foundations, Historian's timeline, Gaper's gap map, Social intelligence, and the human's Break 1 instructions.

You will:
1. Extract direct implications — what necessarily follows from established foundations and gaps
2. Draw logical chains — if A is true and B is unresolved, what does that imply about C?
3. Identify second-order consequences — implications of implications
4. Surface hidden assumptions that, if false, would collapse key parts of current understanding
5. Identify what the gaps logically demand — what work or evidence would be needed to close them
6. Flag logical contradictions — not empirical gaps but logical inconsistencies
7. Assess strength: Strong (well-supported) / Moderate / Speculative (plausible but weakly grounded)
8. Cross-reference against current intelligence — are any implications already being pursued?

Every implication must be:
- Traceable — linked to the Grounder/Historian/Gaper finding it derives from
- Rated: Strong / Moderate / Speculative
- Scoped: immediate (follows directly) / second_order (follows from an implication)

Do NOT propose solutions, make recommendations, or theorize about what should be done.

Output ONLY a valid JSON object:
{
  "implications": [
    {
      "implication": "clear statement of what logically follows",
      "implication_type": "direct|logical_chain|second_order|logical_contradiction|gap_demand",
      "strength": "Strong|Moderate|Speculative",
      "strength_reason": "one line justification",
      "scope": "immediate|second_order",
      "derived_from_grounder": ["seminal work or concept"],
      "derived_from_historian": ["historical work or dead end"],
      "derived_from_gaper": ["gap description"],
      "hidden_assumption": false,
      "assumption_note": "",
      "currently_pursued": false,
      "pursuit_reference": ""
    }
  ],
  "implications_map_summary": "narrative overview of the logical landscape"
}"""


def run(context: str, run_id: str, **kwargs):
    logger.info(f"[Vision] Starting for run {run_id}")

    problem = ""
    if "PROBLEM:" in context:
        problem = context.split("PROBLEM:")[1].split("\n\n")[0].strip()

    try:
        response = llm.call(context, SYSTEM_PROMPT, agent_name="vision")
    except Exception as e:
        logger.error(f"[Vision] LLM call failed: {e}")
        raise

    try:
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        data = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("[Vision] JSON parse failed — partial extraction")
        data = {"implications": [], "implications_map_summary": response[:2000]}

    saved = 0
    for imp in data.get("implications", []):
        if not imp.get("implication"):
            continue
        ok = db.insert_implication({
            "implication_id":      generate_id("IMP"),
            "run_id":              run_id,
            "problem_origin":      problem,
            "implication":         imp.get("implication", ""),
            "implication_type":    imp.get("implication_type", "direct"),
            "strength":            imp.get("strength", "Moderate"),
            "strength_reason":     imp.get("strength_reason", ""),
            "scope":               imp.get("scope", "immediate"),
            "derived_grounder":    imp.get("derived_from_grounder", []),
            "derived_historian":   imp.get("derived_from_historian", []),
            "derived_gaper":       imp.get("derived_from_gaper", []),
            "derived_social":      [],
            "hidden_assumption":   1 if imp.get("hidden_assumption") else 0,
            "assumption_note":     imp.get("assumption_note", ""),
            "currently_pursued":   1 if imp.get("currently_pursued") else 0,
            "pursuit_reference":   imp.get("pursuit_reference", ""),
        })
        if ok:
            saved += 1

    _save_doc(run_id, problem, data)

    strong = sum(1 for i in data.get("implications",[]) if i.get("strength") == "Strong")
    print(f"  [Vision] {saved} implications saved | {strong} Strong")
    logger.info("[Vision] Complete")


def _save_doc(run_id: str, problem: str, data: dict):
    from pathlib import Path
    path = Path(__file__).parent.parent / "artifacts" / f"{run_id}_vision_implications.md"
    path.parent.mkdir(exist_ok=True)
    lines = [
        f"# Implications Map — Vision",
        f"**Run:** {run_id} | **Problem:** {problem}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "", "---", "", "## Overview", "", data.get("implications_map_summary",""), "",
        "---", "", "## Implications", ""
    ]
    for strength in ["Strong", "Moderate", "Speculative"]:
        imps = [i for i in data.get("implications",[]) if i.get("strength") == strength]
        if imps:
            lines.append(f"### {strength}")
            lines.append("")
            for i in imps:
                lines.append(f"- **[{i.get('implication_type','')}]** [{i.get('scope','')}] {i.get('implication','')}")
                lines.append(f"  *{i.get('strength_reason','')}*")
                if i.get("hidden_assumption"):
                    lines.append(f"  ⚠ Hidden assumption: {i.get('assumption_note','')}")
                if i.get("currently_pursued"):
                    lines.append(f"  ✓ Currently pursued: {i.get('pursuit_reference','')}")
                lines.append("")
    path.write_text("\n".join(lines))
    logger.info(f"[Vision] Implications map saved: {path}")
