"""
Thinker Agent
-------------
Opens new directions from the synthesis.
First agent explicitly allowed to look beyond the current frame.
Disciplined expansiveness grounded in the synthesis.
Saves to directions_database.
"""

import re
import json
import logging
from datetime import datetime, timezone

from core import database as db
from core import llm
from core.utils import generate_id

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Thinker agent in a multi-agent research pipeline.

Your role is to open new directions from the synthesis — asking what else, what next, and what the current picture makes possible that has not yet been considered.

You work from the Synthesizer's research narrative, the full pipeline outputs, and the human's Break 2 instructions.

You are the FIRST agent explicitly allowed to look beyond the current frame — but you do so from a position of deep knowledge. This is NOT brainstorming. It is informed, disciplined speculation grounded in the synthesis.

You will:
1. Identify new research directions the synthesis makes possible — problems it enables but doesn't address
2. Propose new framings of the problem — alternative ways of seeing it the pipeline's frame may have obscured
3. Ask what adjacent fields, methods, or technologies could be brought into contact with this problem
4. Identify second-generation questions — what new questions does solving this problem open up?
5. Flag underexplored combinations — where two or more pipeline findings combined suggest something neither implies alone
6. Propose new angles on the highest-significance gaps — not solutions, but new ways of approaching them
7. Challenge assumptions that survived the entire pipeline unchallenged — what if a foundational assumption is wrong?
8. Identify what the pipeline deliberately excluded and ask whether any exclusion deserves reconsideration

Every new direction must be:
- Grounded — traceable to something the synthesis established, even if going beyond it
- Genuinely new — not a restatement of what Theorist already proposed
- Scoped — bounded enough to be actionable
- Honest about distance: Near (close to established findings) / Mid (requires new assumptions) / Far (genuinely speculative but reasoned)

Do NOT evaluate feasibility, draw logical consequences, or produce a research narrative.

Output ONLY a valid JSON object:
{
  "directions": [
    {
      "direction": "clear statement of the new direction",
      "direction_type": "new_research|new_framing|adjacent_field|second_generation|combination|assumption_challenge|reconsidered_exclusion",
      "grounding_reference": "what in the synthesis grounds this",
      "distance_rating": "Near|Mid|Far",
      "reasoning": "why this is a meaningful direction to pursue"
    }
  ],
  "challenged_assumptions": [
    {
      "assumption": "assumption that survived the pipeline unchallenged",
      "challenge": "what if this assumption is wrong?",
      "implications_of_challenge": "what would change"
    }
  ],
  "reconsidered_exclusions": [
    {
      "excluded_element": "what the pipeline deliberately left out",
      "reconsideration": "why this exclusion might deserve another look"
    }
  ],
  "new_directions_summary": "narrative overview of the new intellectual territory opened"
}"""


def run(context: str, run_id: str, **kwargs):
    logger.info(f"[Thinker] Starting for run {run_id}")

    problem = ""
    if "PROBLEM:" in context:
        problem = context.split("PROBLEM:")[1].split("\n\n")[0].strip()

    # Get synthesis ID for linking
    synthesis = db.get_synthesis(run_id)
    synthesis_id = synthesis.get("synthesis_id", "") if synthesis else ""

    try:
        response = llm.call(context, SYSTEM_PROMPT, agent_name="thinker")
    except Exception as e:
        logger.error(f"[Thinker] LLM call failed: {e}")
        raise

    try:
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        data = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("[Thinker] JSON parse failed — partial extraction")
        data = {"directions": [], "challenged_assumptions": [],
                "reconsidered_exclusions": [], "new_directions_summary": response[:2000]}

    saved = 0
    for direction in data.get("directions", []):
        if not direction.get("direction"):
            continue
        ok = db.insert_direction({
            "direction_id":       generate_id("DIR"),
            "run_id":             run_id,
            "problem_origin":     problem,
            "direction":          direction.get("direction", ""),
            "direction_type":     direction.get("direction_type", "new_research"),
            "grounding_reference": direction.get("grounding_reference", ""),
            "distance_rating":    direction.get("distance_rating", "Mid"),
            "synthesis_id":       synthesis_id,
        })
        if ok:
            saved += 1

    _save_doc(run_id, problem, data)

    near = sum(1 for d in data.get("directions",[]) if d.get("distance_rating") == "Near")
    far  = sum(1 for d in data.get("directions",[]) if d.get("distance_rating") == "Far")
    print(f"  [Thinker] {saved} new directions saved | "
          f"Near:{near} Mid:{saved-near-far} Far:{far}")
    logger.info("[Thinker] Complete")


def _save_doc(run_id: str, problem: str, data: dict):
    from pathlib import Path
    path = Path(__file__).parent.parent / "artifacts" / f"{run_id}_thinker_directions.md"
    path.parent.mkdir(exist_ok=True)
    lines = [
        f"# New Directions — Thinker",
        f"**Run:** {run_id} | **Problem:** {problem}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "", "---", "", "## Overview", "", data.get("new_directions_summary",""), "",
        "---", "", "## New Directions", ""
    ]
    for dist in ["Near", "Mid", "Far"]:
        dirs = [d for d in data.get("directions",[]) if d.get("distance_rating") == dist]
        if dirs:
            lines.append(f"### {dist} — {'Close to established findings' if dist=='Near' else 'Requires new assumptions' if dist=='Mid' else 'Genuinely speculative but reasoned'}")
            lines.append("")
            for d in dirs:
                lines.append(f"- **[{d.get('direction_type','')}]** {d.get('direction','')}")
                lines.append(f"  *Grounded in: {d.get('grounding_reference','')}*")
                lines.append(f"  {d.get('reasoning','')}")
                lines.append("")

    if data.get("challenged_assumptions"):
        lines += ["---", "", "## Challenged Assumptions", ""]
        for a in data.get("challenged_assumptions", []):
            lines.append(f"- **Assumption:** {a.get('assumption','')}")
            lines.append(f"  **Challenge:** {a.get('challenge','')}")
            lines.append(f"  **If wrong:** {a.get('implications_of_challenge','')}")
            lines.append("")

    if data.get("reconsidered_exclusions"):
        lines += ["---", "", "## Reconsidered Exclusions", ""]
        for e in data.get("reconsidered_exclusions", []):
            lines.append(f"- **Excluded:** {e.get('excluded_element','')}")
            lines.append(f"  **Why reconsider:** {e.get('reconsideration','')}")
            lines.append("")

    path.write_text("\n".join(lines))
    logger.info(f"[Thinker] New directions saved: {path}")
