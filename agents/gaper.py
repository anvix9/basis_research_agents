"""
Gaper Agent — Two-Pass Gap Mapping
------------------------------------
Maps absence. Two-pass design to see the full landscape before analyzing:

Pass 1 (SCAN): Sees all seminal + historical sources in full detail, plus a
  theme-clustered digest of ALL current sources (counts, year ranges, top
  authors per theme). Outputs lightweight gap SKETCHES — where gaps are,
  what type, which themes and seminal works anchor them.

FILTER (Python, no LLM): Takes relevant_themes from each gap sketch,
  queries DB to pull the 10 most relevant current sources per gap area.

Pass 2 (ANALYZE): Receives the gap sketches back plus the targeted current
  sources. Produces final structured gaps with references to all three
  source layers.

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


# ─── Pass 1: Scan ─────────────────────────────────────────────────────────────

PASS1_SYSTEM = """You are the Gaper agent (Pass 1 — SCAN) in a multi-agent research pipeline.

Your role is to SURVEY the entire landscape and identify WHERE gaps exist.
You have access to:
- All seminal works (Grounder) — full detail
- All historical sources (Historian) — full detail
- A DIGEST of all current sources (Social) — clustered by theme with counts,
  year ranges, and top authors. You cannot see individual papers here, only
  the aggregate shape of what has been studied.

Your job in this pass:
1. For each fundamental question from the Grounder, assess:
   - Has it been answered? (fully / partially / not at all)
   - Quality, completeness, durability of the answer if yes
   - Why not if no: unstudied? avoided? methodologically blocked?

2. Identify gap SKETCHES — lightweight descriptions of where gaps exist:
   - What the gap is about (1-2 sentences)
   - What type: unstudied / incomplete / contradicted / disciplinary_silence /
     temporal / methodological / assumption / dead_end_revisit
   - Which themes from the current literature are relevant (so we can pull
     targeted papers for Pass 2)
   - Which seminal/historical works anchor this gap
   - Estimated significance: High / Medium / Low

Do NOT write detailed gap descriptions. Do NOT propose solutions.
This is a SURVEY — you are mapping terrain, not analyzing it.

IMPORTANT: Output Strong gaps first (High significance), then Medium, then Low.

Output ONLY a valid JSON object:
{
  "primary_evaluation": [
    {
      "question": "fundamental question from Grounder",
      "status": "answered|partial|unanswered",
      "assessment": "evaluation of answer quality or reason for absence"
    }
  ],
  "gap_sketches": [
    {
      "sketch_id": "GS-1",
      "gap_type": "unstudied|incomplete|contradicted|disciplinary_silence|temporal|methodological|assumption|dead_end_revisit",
      "brief": "1-2 sentence description of where the gap is",
      "significance": "High|Medium|Low",
      "relevant_themes": ["theme_id_1", "theme_id_2"],
      "anchoring_seminal": ["title of seminal work"],
      "anchoring_historical": ["title of historical work"],
      "primary_eval_ref": "which fundamental question this connects to"
    }
  ],
  "landscape_summary": "2-3 sentence overview of the gap landscape"
}"""


# ─── Pass 2: Analyze ──────────────────────────────────────────────────────────

PASS2_SYSTEM = """You are the Gaper agent (Pass 2 — ANALYZE) in a multi-agent research pipeline.

You completed a scan pass and identified gap sketches. Now you have targeted
current sources for each gap area, pulled from the database based on your
theme selections.

Your job:
1. For each gap sketch, produce the FULL gap analysis using the targeted
   sources as evidence. Confirm, refine, or revise your initial assessment.
2. Add references to current sources that confirm or contradict the gap.
3. Rate significance with a specific reason.
4. Identify recurring patterns and dead ends worth revisiting.

Every gap must reference at least one source layer (seminal, historical, or current).

Output ONLY a valid JSON object:
{
  "gaps": [
    {
      "gap_type": "unstudied|incomplete|contradicted|disciplinary_silence|temporal|methodological|assumption|dead_end_revisit",
      "description": "clear, detailed statement of the gap",
      "significance": "High|Medium|Low",
      "significance_reason": "one line why",
      "primary_evaluation_ref": "which fundamental question this connects to",
      "references_grounder": ["seminal work title"],
      "references_historian": ["historical work or dead end title"],
      "references_current": ["current source title that confirms/relates to this gap"],
      "dead_end_revisit": false,
      "recurring_pattern": false,
      "recurring_reason": ""
    }
  ],
  "gap_map_summary": "narrative overview of the full gap landscape"
}"""


# ─── Context builders (internal to Gaper) ─────────────────────────────────────

def _build_pass1_context(run_id: str, problem: str) -> str:
    """
    Pass 1 context: full seminal + historical, theme-clustered digest of current.
    The digest shows the SHAPE of current literature without individual papers.
    """
    seminal    = db.get_sources_by_type("seminal",    run_id)
    historical = db.get_sources_by_type("historical", run_id)
    current    = db.get_sources_by_type("current",    run_id)

    ctx = f"PROBLEM:\n{problem}\n"

    # Seminal — full detail, all of them
    ctx += "\n=== SEMINAL WORKS (Grounder) — FULL DETAIL ===\n"
    for s in seminal:
        authors = json.loads(s.get("authors") or "[]") if s.get("authors") else []
        author_str = ", ".join(authors[:3])
        ctx += (
            f"\n- [{s.get('year','n.d.')}] {s.get('title','Untitled')}"
            f" ({author_str}) | {s.get('source_name','')}"
            f"\n  Reason: {s.get('seminal_reason','')}"
            f"\n  Abstract: {(s.get('abstract','') or '')[:300]}"
        )

    # Historical — full detail, all of them
    ctx += "\n\n=== HISTORICAL SOURCES (Historian) — FULL DETAIL ===\n"
    for s in sorted(historical, key=lambda x: x.get('year') or 9999):
        authors = json.loads(s.get("authors") or "[]") if s.get("authors") else []
        author_str = ", ".join(authors[:2])
        ctx += (
            f"\n- [{s.get('year','n.d.')}] {s.get('title','')}"
            f" ({author_str})"
            f"\n  {s.get('historical_reason','')}"
        )

    # Current — theme-clustered DIGEST (no individual papers)
    ctx += "\n\n=== CURRENT LITERATURE DIGEST (Social) — AGGREGATE VIEW ===\n"
    ctx += f"Total current sources: {len(current)}\n"

    theme_clusters: dict[str, list] = {}
    for s in current:
        tags_raw = s.get("theme_tags", "[]")
        try:
            tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
        except (json.JSONDecodeError, TypeError):
            tags = []
        for tag in (tags if isinstance(tags, list) else []):
            theme_clusters.setdefault(tag, []).append(s)

    for theme_id, sources in sorted(theme_clusters.items(), key=lambda x: -len(x[1])):
        years = [s.get("year") for s in sources if s.get("year")]
        year_range = f"{min(years)}-{max(years)}" if years else "?"

        all_authors = []
        for s in sources:
            try:
                authors = json.loads(s.get("authors") or "[]")
                all_authors.extend(authors[:2] if isinstance(authors, list) else [])
            except (json.JSONDecodeError, TypeError):
                pass
        author_counts: dict[str, int] = {}
        for a in all_authors:
            if a:
                author_counts[a] = author_counts.get(a, 0) + 1
        top_authors = sorted(author_counts.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{a} ({n})" for a, n in top_authors) or "various"

        high = sum(1 for s in sources if s.get("relevance_rating") == "High")
        med  = sum(1 for s in sources if s.get("relevance_rating") == "Medium")

        ctx += (
            f"\n  [{theme_id}]: {len(sources)} papers | {year_range} | "
            f"High: {high}, Medium: {med} | Top authors: {top_str}"
        )

    return ctx


def _build_pass2_context(
    problem: str,
    pass1_data: dict,
    targeted_sources: dict[str, list[dict]],
    run_id: str,
) -> str:
    """
    Pass 2 context: gap sketches from Pass 1 + targeted current sources per gap.
    """
    ctx = f"PROBLEM:\n{problem}\n"

    # Primary evaluation from Pass 1
    ctx += "\n=== PRIMARY EVALUATION (from your Pass 1 scan) ===\n"
    for pe in pass1_data.get("primary_evaluation", []):
        ctx += f"\n- [{pe.get('status','')}] {pe.get('question','')}"
        ctx += f"\n  {pe.get('assessment','')}"

    # Gap sketches with their targeted sources
    ctx += "\n\n=== GAP SKETCHES + TARGETED CURRENT SOURCES ===\n"
    for gs in pass1_data.get("gap_sketches", []):
        sketch_id = gs.get("sketch_id", "?")
        ctx += (
            f"\n--- {sketch_id}: [{gs.get('gap_type','')}] [{gs.get('significance','')}] ---"
            f"\n  {gs.get('brief','')}"
            f"\n  Anchored in: {', '.join(gs.get('anchoring_seminal',[])) or 'none'}"
            f"\n  Historical: {', '.join(gs.get('anchoring_historical',[])) or 'none'}"
        )

        gap_sources = targeted_sources.get(sketch_id, [])
        if gap_sources:
            ctx += f"\n  Targeted current sources ({len(gap_sources)}):"
            for s in gap_sources:
                authors = json.loads(s.get("authors") or "[]") if s.get("authors") else []
                author_str = ", ".join(authors[:2]) or "?"
                ctx += (
                    f"\n    - [{s.get('year','?')}] {s.get('title','')[:100]}"
                    f" ({author_str}) [{s.get('relevance_rating','')}]"
                    f"\n      {(s.get('abstract','') or '')[:200]}"
                )
        else:
            ctx += "\n  No targeted sources found for this gap area."

    # Seminal works reference list
    seminal = db.get_sources_by_type("seminal", run_id)
    ctx += "\n\n=== SEMINAL WORKS (reference) ===\n"
    for s in seminal:
        ctx += f"\n- [{s.get('year','?')}] {s.get('title','')[:80]}"

    return ctx


# ─── Filter: gap sketches → targeted DB queries ──────────────────────────────

def _fetch_targeted_sources(
    run_id: str,
    gap_sketches: list[dict],
    per_gap_limit: int = 10,
) -> dict[str, list[dict]]:
    """
    For each gap sketch, pull the most relevant current sources from the DB
    based on the relevant_themes the LLM identified. Pure Python — no LLM.
    """
    import sqlite3
    from pathlib import Path

    db_path = Path(__file__).parent.parent / "db" / "pipeline.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    targeted: dict[str, list[dict]] = {}

    for gs in gap_sketches:
        sketch_id = gs.get("sketch_id", "?")
        themes = gs.get("relevant_themes", [])
        if not themes:
            targeted[sketch_id] = []
            continue

        results = []
        for theme in themes:
            query = """
                SELECT * FROM sources
                WHERE run_id = ? AND type = 'current'
                  AND theme_tags LIKE ?
                ORDER BY
                  CASE relevance_rating
                    WHEN 'High' THEN 1
                    WHEN 'Medium' THEN 2
                    ELSE 3
                  END,
                  year DESC
                LIMIT ?
            """
            rows = conn.execute(query, (run_id, f'%"{theme}"%', per_gap_limit)).fetchall()
            for row in rows:
                results.append(dict(row))

        # Deduplicate by source_id
        seen = set()
        deduped = []
        for r in results:
            sid = r.get("source_id", "")
            if sid not in seen:
                seen.add(sid)
                deduped.append(r)
            if len(deduped) >= per_gap_limit:
                break

        targeted[sketch_id] = deduped

    conn.close()
    return targeted


# ─── Main run ─────────────────────────────────────────────────────────────────

def run(context: str, run_id: str, **kwargs):
    logger.info(f"[Gaper] Starting for run {run_id}")

    problem = ""
    if "PROBLEM:" in context:
        problem = context.split("PROBLEM:")[1].split("\n\n")[0].strip()

    # ── Pass 1: Scan ──────────────────────────────────────────────────────
    print("  [Gaper] Pass 1 — scanning full landscape...")
    pass1_ctx = _build_pass1_context(run_id, problem)

    try:
        pass1_response = llm.call(pass1_ctx, PASS1_SYSTEM, agent_name="gaper")
    except Exception as e:
        logger.error(f"[Gaper] Pass 1 LLM call failed: {e}")
        raise

    try:
        clean = re.sub(r"```(?:json)?|```", "", pass1_response).strip()
        pass1_data = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("[Gaper] Pass 1 JSON parse failed — using partial data")
        pass1_data = {
            "primary_evaluation": [],
            "gap_sketches": [],
            "landscape_summary": pass1_response[:2000],
        }

    sketches = pass1_data.get("gap_sketches", [])
    pe_count = len(pass1_data.get("primary_evaluation", []))
    print(f"  [Gaper] Pass 1 complete — {len(sketches)} gap sketches, "
          f"{pe_count} fundamental questions evaluated")

    # ── Filter: deterministic DB query ────────────────────────────────────
    print(f"  [Gaper] Fetching targeted sources for {len(sketches)} gap areas...")
    targeted = _fetch_targeted_sources(run_id, sketches, per_gap_limit=10)
    total_targeted = sum(len(v) for v in targeted.values())
    print(f"  [Gaper] {total_targeted} targeted sources pulled from DB")

    # ── Pass 2: Analyze ───────────────────────────────────────────────────
    print("  [Gaper] Pass 2 — analyzing gaps with targeted evidence...")
    pass2_ctx = _build_pass2_context(problem, pass1_data, targeted, run_id)

    try:
        pass2_response = llm.call(pass2_ctx, PASS2_SYSTEM, agent_name="gaper")
    except Exception as e:
        logger.error(f"[Gaper] Pass 2 LLM call failed: {e}")
        raise

    try:
        clean2 = re.sub(r"```(?:json)?|```", "", pass2_response).strip()
        pass2_data = json.loads(clean2)
    except json.JSONDecodeError:
        logger.warning("[Gaper] Pass 2 JSON parse failed — salvaging")
        pass2_data = {"gaps": [], "gap_map_summary": pass2_response[:2000]}

    # ── Save to database ──────────────────────────────────────────────────
    saved = 0
    for gap in pass2_data.get("gaps", []):
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
            "references_social":     gap.get("references_current", []),
            "dead_end_revisit":      1 if gap.get("dead_end_revisit") else 0,
            "recurring_pattern":     1 if gap.get("recurring_pattern") else 0,
            "recurring_reason":      gap.get("recurring_reason", ""),
        })
        if ok:
            saved += 1

    # ── Save artifact ─────────────────────────────────────────────────────
    _save_doc(run_id, problem, pass1_data, pass2_data)

    high = sum(1 for g in pass2_data.get("gaps", []) if g.get("significance") == "High")
    print(f"  [Gaper] {saved} gaps saved | {high} High significance | "
          f"{pe_count} fundamental questions evaluated")
    logger.info("[Gaper] Complete")


# ─── Artifact writer ──────────────────────────────────────────────────────────

def _save_doc(run_id: str, problem: str, pass1_data: dict, pass2_data: dict):
    from pathlib import Path
    path = Path(__file__).parent.parent / "artifacts" / f"{run_id}_gaper_gaps.md"
    path.parent.mkdir(exist_ok=True)

    lines = [
        f"# Gap Map — Gaper (Two-Pass)",
        f"**Run:** {run_id} | **Problem:** {problem}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "", "---", "",
        "## Primary Evaluation — Have Fundamental Questions Been Answered?", "",
    ]

    for q in pass1_data.get("primary_evaluation", []):
        lines.append(f"### {q.get('question', '')}")
        lines.append(f"**Status:** {q.get('status', '')}")
        lines.append(q.get("assessment", ""))
        lines.append("")

    lines += [
        "---", "",
        "## Landscape Summary", "",
        pass1_data.get("landscape_summary", ""),
        "", "---", "",
        "## Gap Map", "",
        pass2_data.get("gap_map_summary", ""),
        "", "---", "",
    ]

    for sig in ["High", "Medium", "Low"]:
        gaps = [g for g in pass2_data.get("gaps", []) if g.get("significance") == sig]
        if gaps:
            lines.append(f"## {sig} Significance Gaps")
            lines.append("")
            for g in gaps:
                lines.append(f"- **[{g.get('gap_type', '')}]** {g.get('description', '')}")
                lines.append(f"  *{g.get('significance_reason', '')}*")

                refs = []
                for r in g.get("references_grounder", []):
                    refs.append(f"Grounder: {r}")
                for r in g.get("references_historian", []):
                    refs.append(f"Historian: {r}")
                for r in g.get("references_current", []):
                    refs.append(f"Current: {r}")
                if refs:
                    lines.append(f"  Sources: {'; '.join(refs[:5])}")

                if g.get("recurring_pattern"):
                    lines.append(f"  ⟳ Recurring: {g.get('recurring_reason', '')}")
                if g.get("dead_end_revisit"):
                    lines.append(f"  ↩ Dead end worth revisiting")
                lines.append("")

    path.write_text("\n".join(lines))
    logger.info(f"[Gaper] Gap map saved: {path}")
