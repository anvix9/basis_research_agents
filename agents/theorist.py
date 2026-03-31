"""
Theorist Agent
--------------
Proposes concrete approaches, frameworks, and solutions.
First constructive agent — every proposal anchored in pipeline outputs.
Saves to proposals_database.

Two-pass design to survive output size limits:
  Pass 1: proposals_summary + index of proposals (always completes)
  Pass 2: full detail for each proposal individually
This prevents JSON truncation from silently killing all proposals.
"""

import re
import json
import logging
from datetime import datetime, timezone

from core import database as db
from core import llm
from core.utils import generate_id

logger = logging.getLogger(__name__)


OVERVIEW_SYSTEM = """You are the Theorist agent in a multi-agent research pipeline.

Based on the pipeline context (Grounder foundations, Historian timeline, Gaper gaps, Vision implications), produce an overview of your proposals.

Output ONLY valid JSON:
{
  "proposals_summary": "2-3 paragraph narrative overview of the proposal landscape",
  "proposals_index": [
    {
      "id": "P1",
      "proposal": "one sentence concrete statement of the approach",
      "proposal_type": "novel|extension|revival|hybrid",
      "promise_rating": "High|Medium|Low",
      "promise_reason": "one sentence",
      "addresses_gaps": ["gap title or short description"],
      "addresses_implications": ["implication short description"]
    }
  ]
}

Produce 4-8 proposals. Be concrete and specific — not generic."""


DETAIL_SYSTEM = """You are the Theorist agent in a multi-agent research pipeline.

Expand ONE proposal into full detail. Output ONLY valid JSON:
{
  "proposal": "full clear statement of the approach",
  "proposal_type": "novel|extension|revival|hybrid",
  "addresses_gaps": ["gap description"],
  "addresses_implications": ["implication statement"],
  "addresses_foundations": ["seminal work title"],
  "assumptions": ["what must be true for this to work"],
  "requirements": ["what is needed to execute this"],
  "predictions": ["what this predicts if successful"],
  "dead_end_reassessment": false,
  "dead_end_reference": "",
  "dead_end_reason": "",
  "interdependencies": ["other proposal it depends on or enables"],
  "promise_rating": "High|Medium|Low",
  "promise_reason": "one line justification",
  "scope": "what this addresses and what it deliberately leaves out"
}"""


def _parse_json(text: str, label: str) -> dict:
    """Robust JSON extraction — handles fences and partial truncation."""
    clean = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    start = clean.find("{")
    end   = clean.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(clean[start:end+1])
        except json.JSONDecodeError:
            pass
    logger.warning(f"[{label}] JSON parse failed")
    return {}


def run(context: str, run_id: str, **kwargs):
    logger.info(f"[Theorist] Starting for run {run_id}")

    problem = ""
    if "PROBLEM:" in context:
        problem = context.split("PROBLEM:")[1].split("\n\n")[0].strip()

    # Pass 1 — overview (small, always completes)
    print("  [Theorist] Pass 1 — generating proposal overview...")
    try:
        overview_resp = llm.call(context, OVERVIEW_SYSTEM, agent_name="theorist")
    except Exception as e:
        logger.error(f"[Theorist] Overview call failed: {e}")
        raise

    overview          = _parse_json(overview_resp, "Theorist/overview")
    proposals_summary = overview.get("proposals_summary", "")
    proposals_index   = overview.get("proposals_index", [])

    if not proposals_index:
        logger.warning("[Theorist] No proposals in overview")
        data = {"proposals": [], "proposals_summary": proposals_summary or overview_resp[:2000]}
        _save_doc(run_id, problem, data)
        print("  [Theorist] 0 proposals saved")
        return

    print(f"  [Theorist] {len(proposals_index)} proposals indexed — expanding each...")

    # Pass 2 — full detail per proposal
    full_proposals = []
    for idx, stub in enumerate(proposals_index):
        print(f"  [Theorist] Expanding {stub.get('id','P'+str(idx+1))} ({idx+1}/{len(proposals_index)})...")
        detail_prompt = f"""{context}

---
PROPOSAL TO EXPAND:
ID: {stub.get('id','')}
Statement: {stub.get('proposal','')}
Type: {stub.get('proposal_type','')}
Promise: {stub.get('promise_rating','')} — {stub.get('promise_reason','')}
Addresses gaps: {', '.join(stub.get('addresses_gaps',[]))}
Addresses implications: {', '.join(stub.get('addresses_implications',[]))}

Expand this into full detail. Be specific and concrete."""

        try:
            detail_resp = llm.call(detail_prompt, DETAIL_SYSTEM, agent_name="theorist")
            detail = _parse_json(detail_resp, f"Theorist/P{idx+1}")
            if detail.get("proposal"):
                full_proposals.append(detail)
            else:
                # Keep stub with basic fields
                full_proposals.append({
                    "proposal":               stub.get("proposal", ""),
                    "proposal_type":          stub.get("proposal_type", "novel"),
                    "addresses_gaps":         stub.get("addresses_gaps", []),
                    "addresses_implications": stub.get("addresses_implications", []),
                    "addresses_foundations":  [],
                    "assumptions":            [],
                    "requirements":           [],
                    "predictions":            [],
                    "dead_end_reassessment":  False,
                    "dead_end_reference":     "",
                    "dead_end_reason":        "",
                    "interdependencies":      [],
                    "promise_rating":         stub.get("promise_rating", "Medium"),
                    "promise_reason":         stub.get("promise_reason", ""),
                    "scope":                  "",
                })
        except Exception as e:
            logger.warning(f"[Theorist] Detail expansion failed for {stub.get('id','?')}: {e}")

    data = {"proposals": full_proposals, "proposals_summary": proposals_summary}

    # Save to database
    saved = 0
    for prop in data["proposals"]:
        if not prop.get("proposal"):
            continue
        ok = db.insert_proposal({
            "proposal_id":            generate_id("PROP"),
            "run_id":                 run_id,
            "problem_origin":         problem,
            "proposal":               prop.get("proposal", ""),
            "proposal_type":          prop.get("proposal_type", "novel"),
            "addresses_gaps":         prop.get("addresses_gaps", []),
            "addresses_implications": prop.get("addresses_implications", []),
            "addresses_foundations":  prop.get("addresses_foundations", []),
            "assumptions":            prop.get("assumptions", []),
            "requirements":           prop.get("requirements", []),
            "predictions":            prop.get("predictions", []),
            "dead_end_reassessment":  1 if prop.get("dead_end_reassessment") else 0,
            "dead_end_reference":     prop.get("dead_end_reference", ""),
            "dead_end_reason":        prop.get("dead_end_reason", ""),
            "interdependencies":      prop.get("interdependencies", []),
            "promise_rating":         prop.get("promise_rating", "Medium"),
            "promise_reason":         prop.get("promise_reason", ""),
            "novel_vs_extension":     prop.get("proposal_type", "novel"),
            "scope":                  prop.get("scope", ""),
        })
        if ok:
            saved += 1

    _save_doc(run_id, problem, data)
    high = sum(1 for p in data["proposals"] if p.get("promise_rating") == "High")
    print(f"  [Theorist] {saved} proposals saved | {high} High promise")
    logger.info("[Theorist] Complete")


def _save_doc(run_id: str, problem: str, data: dict):
    from pathlib import Path
    path = Path(__file__).parent.parent / "artifacts" / f"{run_id}_theorist_proposals.md"
    path.parent.mkdir(exist_ok=True)
    lines = [
        "# Proposals Document — Theorist",
        f"**Run:** {run_id} | **Problem:** {problem}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "", "---", "", "## Overview", "",
        data.get("proposals_summary", ""), "",
        "---", "", "## Proposals", ""
    ]
    for rating in ["High", "Medium", "Low"]:
        props = [p for p in data.get("proposals", []) if p.get("promise_rating") == rating]
        if not props:
            continue
        lines += [f"### {rating} Promise", ""]
        for p in props:
            lines.append(f"#### [{p.get('proposal_type','')}] {p.get('proposal','')}")
            lines.append(f"*{p.get('promise_reason','')}*")
            lines.append(f"**Scope:** {p.get('scope','')}")
            if p.get("assumptions"):
                lines.append(f"**Assumes:** {'; '.join(p['assumptions'])}")
            if p.get("requirements"):
                lines.append(f"**Requires:** {'; '.join(p['requirements'])}")
            if p.get("predictions"):
                lines.append(f"**Predicts:** {'; '.join(p['predictions'])}")
            if p.get("dead_end_reassessment"):
                lines.append(f"↩ **Revival:** {p.get('dead_end_reason','')}")
            if p.get("interdependencies"):
                lines.append(f"⟷ **Depends on:** {', '.join(p['interdependencies'])[:120]}")
            lines.append("")
    path.write_text("\n".join(lines))
    logger.info(f"[Theorist] Proposals document saved: {path}")
