"""
Gaper Agent — Tree-Native Gap Mapping
---------------------------------------
Maps absence using the argument tree as its primary analytical structure.

Three-step process:

Step 1 — STRUCTURAL GAPS (deterministic, from tree):
  Uses tree.find_gaps() to identify:
    - Unanswered questions (question nodes with no claims)
    - Unsupported claims (claim nodes with no evidence)
    - Weak claims (low confidence, single evidence source)
  Uses tree.find_bridge_needs() to identify temporal gaps.
  These are REAL gaps — the tree proves they exist.

Step 2 — ANALYTICAL GAPS (LLM, two-pass):
  Pass 1 (SCAN): LLM sees the full tree + structural gaps from Step 1.
    Identifies gaps the tree structure CAN'T detect mechanistically:
    - Disciplinary silences (adjacent fields that should connect but don't)
    - Methodological blind spots (everyone uses the same approach)
    - Assumption gaps (hidden premises no one has questioned)
    - Dead-end revisit opportunities (abandoned approaches worth another look)
    Outputs gap sketches with relevant themes.

  FILTER: Deterministic DB query pulls targeted current sources per gap area.

  Pass 2 (ANALYZE): LLM sees sketches + targeted sources + structural gaps.
    Produces final gap analysis with references to all three source layers.

Step 3 — MERGE: Combines structural gaps (from tree) with analytical gaps
  (from LLM) into a single gap map. Structural gaps get priority status
  because they are proven, not inferred.

The tree's gaps are the FOUNDATION. The LLM adds depth and context.
The LLM cannot override or dismiss a structural gap.
"""

import re
import json
import logging
from datetime import datetime, timezone

from core import database as db
from core import llm
from core.utils import generate_id

logger = logging.getLogger(__name__)


# ─── LLM Prompts ──────────────────────────────────────────────────────────────

PASS1_SYSTEM = """You are the Gaper agent (Pass 1 — ANALYTICAL SCAN) in a multi-agent research pipeline.

You have received:
1. The ARGUMENT TREE — a structured map of all claims and evidence gathered so far
2. STRUCTURAL GAPS — gaps that the tree itself proves exist (unanswered questions,
   unsupported claims, weak claims, temporal gaps). These are FACTS, not suggestions.
   You CANNOT dismiss or downgrade them.
3. A theme-clustered digest of current literature (aggregate counts, not individual papers)

Your job: identify gaps that the tree structure CANNOT detect mechanistically:
- Disciplinary silences: adjacent fields that SHOULD connect but have zero bridging evidence
- Methodological blind spots: all evidence uses the same methodology — alternatives untried
- Assumption gaps: hidden premises underlying multiple claims that no one has questioned
- Contradictions that haven't been surfaced: claims that logically conflict but aren't marked
- Dead-end revisit opportunities: approaches abandoned for reasons that may no longer hold
- Temporal silences: periods where no research exists (beyond what bridge_needs already found)
- Paradigm gaps: dominant framework may be masking alternative interpretations

IMPORTANT:
- Do NOT repeat the structural gaps — they are already identified. Add to them.
- Every analytical gap must explain WHY the tree didn't catch it (what makes it invisible to structure alone)
- Tag which themes from current literature would be relevant (for targeted source pull in Pass 2)

Output ONLY a valid JSON object:
{
  "analytical_gaps": [
    {
      "sketch_id": "AG-1",
      "gap_type": "disciplinary_silence|methodological|assumption|contradiction|dead_end_revisit|temporal_silence|paradigm",
      "brief": "1-2 sentence description",
      "significance": "High|Medium|Low",
      "why_tree_missed": "why structural analysis couldn't detect this",
      "relevant_themes": ["theme_id_1", "theme_id_2"],
      "anchoring_nodes": ["node_id or claim text that this gap relates to"],
      "connects_to_structural": "which structural gap this extends or 'independent'"
    }
  ],
  "tree_observations": "2-3 sentence assessment of the tree's overall health — where it's strong, where it's fragile"
}"""


PASS2_SYSTEM = """You are the Gaper agent (Pass 2 — FULL ANALYSIS) in a multi-agent research pipeline.

You have:
1. STRUCTURAL GAPS from the argument tree (proven — cannot be dismissed)
2. Your analytical gap sketches from Pass 1
3. Targeted current sources pulled from the database for each gap area

Your job: produce the FINAL gap analysis. For each gap (both structural and analytical):
1. Write a clear, detailed description grounded in the evidence
2. Rate significance with a specific reason
3. Reference sources from all three layers (seminal, historical, current)
4. For structural gaps: explain what they mean for the research (not just "this question has no claims")
5. For analytical gaps: confirm, refine, or revise based on targeted sources

CRITICAL RULES:
- Structural gaps from the tree are MANDATORY — include all of them, enhanced with your analysis
- You are ADDING context and depth to structural gaps, not replacing them
- Every gap must reference at least one specific work or tree node
- Do not invent gaps that have no basis in either the tree structure or the evidence

Output ONLY a valid JSON object:
{
  "gaps": [
    {
      "gap_origin": "structural|analytical",
      "gap_type": "unanswered_question|unsupported_claim|weak_claim|temporal|disciplinary_silence|methodological|assumption|contradiction|dead_end_revisit|paradigm",
      "description": "clear, detailed statement of the gap",
      "significance": "High|Medium|Low",
      "significance_reason": "one line why",
      "tree_node_ref": "node_id this gap connects to (if any)",
      "references_grounder": ["seminal work title"],
      "references_historian": ["historical work or dead end"],
      "references_current": ["current source that confirms/relates to gap"],
      "dead_end_revisit": false,
      "recurring_pattern": false,
      "recurring_reason": ""
    }
  ],
  "gap_map_summary": "narrative overview of the full gap landscape"
}"""


# ─── Step 1: Structural gaps from tree ─────────────────────────────────────────

def _get_structural_gaps(run_id: str) -> dict:
    """Extract structural gaps directly from the argument tree."""
    from core.argument_tree import TreeBuilder

    tree = TreeBuilder(run_id)
    stats = tree.get_stats()

    if stats.get("total_nodes", 0) == 0:
        tree.close()
        return {"gaps": [], "bridge_needs": [], "tree_context": "", "stats": stats}

    gaps = tree.find_gaps()
    bridge_needs = tree.find_bridge_needs(min_gap_years=15)
    tree_ctx = tree.to_context(max_depth=3, include_evidence=True)
    tree.close()

    return {
        "gaps": gaps,
        "bridge_needs": bridge_needs,
        "tree_context": tree_ctx,
        "stats": stats,
    }


# ─── Step 2: Context builders ─────────────────────────────────────────────────

def _build_pass1_context(run_id: str, problem: str, structural: dict) -> str:
    """Pass 1: tree + structural gaps + current literature digest."""
    current = db.get_sources_by_type("current", run_id)

    ctx = f"PROBLEM:\n{problem}\n"

    # Argument tree
    ctx += f"\n=== ARGUMENT TREE ===\n{structural['tree_context']}\n"

    # Structural gaps (from tree — these are proven facts)
    ctx += "\n=== STRUCTURAL GAPS (proven by tree — CANNOT be dismissed) ===\n"
    for g in structural["gaps"]:
        ctx += f"\n  [{g['gap_type']}] {g['content'][:200]}"
        if g.get("confidence"):
            ctx += f" (confidence: {g['confidence']:.0%})"

    if structural["bridge_needs"]:
        ctx += "\n\n  TEMPORAL GAPS (need bridge papers):\n"
        for b in structural["bridge_needs"]:
            ctx += f"\n  {b['earlier_year']} → {b['later_year']} ({b['gap_years']}yr gap): {b['question'][:80]}"

    # Current literature digest (aggregate)
    ctx += f"\n\n=== CURRENT LITERATURE DIGEST ({len(current)} sources) ===\n"
    theme_clusters: dict[str, list] = {}
    for s in current:
        try:
            tags = json.loads(s.get("theme_tags", "[]")) if isinstance(s.get("theme_tags"), str) else s.get("theme_tags", [])
        except:
            tags = []
        for tag in (tags if isinstance(tags, list) else []):
            theme_clusters.setdefault(tag, []).append(s)

    for theme_id, sources in sorted(theme_clusters.items(), key=lambda x: -len(x[1])):
        years = [s.get("year") for s in sources if s.get("year")]
        year_range = f"{min(years)}-{max(years)}" if years else "?"
        high = sum(1 for s in sources if s.get("relevance_rating") == "High")
        ctx += f"\n  [{theme_id}]: {len(sources)} papers | {year_range} | High: {high}"

    return ctx


def _build_pass2_context(
    problem: str, structural: dict, analytical_gaps: list,
    targeted_sources: dict[str, list[dict]], run_id: str
) -> str:
    """Pass 2: structural gaps + analytical sketches + targeted sources."""
    ctx = f"PROBLEM:\n{problem}\n"

    # Structural gaps — mandatory, enhanced
    ctx += "\n=== STRUCTURAL GAPS (from tree — MUST be included in output) ===\n"
    for i, g in enumerate(structural["gaps"]):
        ctx += f"\n  SG-{i+1} [{g['gap_type']}]: {g['content'][:200]}"

    for i, b in enumerate(structural["bridge_needs"]):
        ctx += f"\n  SG-BRIDGE-{i+1}: {b['earlier_year']}→{b['later_year']} gap in '{b['question'][:60]}'"

    # Analytical gaps with targeted sources
    ctx += "\n\n=== ANALYTICAL GAP SKETCHES + TARGETED SOURCES ===\n"
    for ag in analytical_gaps:
        sketch_id = ag.get("sketch_id", "?")
        ctx += (
            f"\n--- {sketch_id}: [{ag.get('gap_type','')}] [{ag.get('significance','')}] ---"
            f"\n  {ag.get('brief','')}"
            f"\n  Why tree missed: {ag.get('why_tree_missed','')}"
        )

        gap_sources = targeted_sources.get(sketch_id, [])
        if gap_sources:
            ctx += f"\n  Targeted sources ({len(gap_sources)}):"
            for s in gap_sources:
                authors = json.loads(s.get("authors") or "[]") if s.get("authors") else []
                ctx += (
                    f"\n    - [{s.get('year','?')}] {s.get('title','')[:80]}"
                    f" ({', '.join(authors[:2])})"
                    f"\n      {(s.get('abstract','') or '')[:150]}"
                )

    # Seminal works reference
    seminal = db.get_sources_by_type("seminal", run_id)
    ctx += "\n\n=== SEMINAL WORKS (reference) ===\n"
    for s in seminal:
        ctx += f"\n- [{s.get('year','?')}] {s.get('title','')[:80]}"

    return ctx


# ─── Filter: fetch targeted sources per gap ────────────────────────────────────

def _fetch_targeted_sources(
    run_id: str, gap_sketches: list[dict], per_gap_limit: int = 10
) -> dict[str, list[dict]]:
    """Pull relevant current sources per analytical gap from DB."""
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
            rows = conn.execute(
                """SELECT * FROM sources
                   WHERE run_id = ? AND type = 'current' AND theme_tags LIKE ?
                   ORDER BY CASE relevance_rating WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END, year DESC
                   LIMIT ?""",
                (run_id, f'%"{theme}"%', per_gap_limit)
            ).fetchall()
            results.extend(dict(r) for r in rows)

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


# ─── Main run ──────────────────────────────────────────────────────────────────

def run(context: str, run_id: str, **kwargs):
    logger.info(f"[Gaper] Starting for run {run_id}")

    problem = ""
    if "PROBLEM:" in context:
        problem = context.split("PROBLEM:")[1].split("\n\n")[0].strip()

    # ── Step 1: Structural gaps from tree ─────────────────────────────────
    print("  [Gaper] Step 1 — extracting structural gaps from argument tree...")
    structural = _get_structural_gaps(run_id)

    n_struct = len(structural["gaps"])
    n_bridge = len(structural["bridge_needs"])
    print(f"  [Gaper] Tree has {structural['stats'].get('total_nodes',0)} nodes")
    print(f"  [Gaper] Structural gaps: {n_struct} (unanswered/unsupported/weak)")
    print(f"  [Gaper] Bridge needs: {n_bridge} temporal gaps")

    for g in structural["gaps"]:
        print(f"    [{g['gap_type']}] {g['content'][:70]}...")

    # ── Step 2a: Pass 1 — LLM analytical scan ────────────────────────────
    print("  [Gaper] Step 2a — LLM analytical scan (gaps tree can't detect)...")
    pass1_ctx = _build_pass1_context(run_id, problem, structural)

    try:
        pass1_response = llm.call(pass1_ctx, PASS1_SYSTEM, agent_name="gaper")
    except Exception as e:
        logger.error(f"[Gaper] Pass 1 failed: {e}")
        raise

    try:
        clean = re.sub(r"```(?:json)?|```", "", pass1_response).strip()
        pass1_data = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("[Gaper] Pass 1 JSON parse failed")
        pass1_data = {"analytical_gaps": [], "tree_observations": pass1_response[:1000]}

    analytical = pass1_data.get("analytical_gaps", [])
    print(f"  [Gaper] Analytical gaps found: {len(analytical)}")
    print(f"  [Gaper] Tree health: {pass1_data.get('tree_observations','')[:100]}")

    # ── Step 2b: Filter — fetch targeted sources ──────────────────────────
    if analytical:
        print(f"  [Gaper] Fetching targeted sources for {len(analytical)} analytical gaps...")
        targeted = _fetch_targeted_sources(run_id, analytical, per_gap_limit=10)
        total_targeted = sum(len(v) for v in targeted.values())
        print(f"  [Gaper] {total_targeted} targeted sources pulled")
    else:
        targeted = {}

    # ── Step 2c: Pass 2 — full analysis ───────────────────────────────────
    print("  [Gaper] Step 2c — full gap analysis with targeted evidence...")
    pass2_ctx = _build_pass2_context(problem, structural, analytical, targeted, run_id)

    try:
        pass2_response = llm.call(pass2_ctx, PASS2_SYSTEM, agent_name="gaper")
    except Exception as e:
        logger.error(f"[Gaper] Pass 2 failed: {e}")
        raise

    try:
        clean2 = re.sub(r"```(?:json)?|```", "", pass2_response).strip()
        pass2_data = json.loads(clean2)
    except json.JSONDecodeError:
        logger.warning("[Gaper] Pass 2 JSON parse failed")
        pass2_data = {"gaps": [], "gap_map_summary": pass2_response[:2000]}

    # ── Step 3: Merge + validate ──────────────────────────────────────────
    # Ensure ALL structural gaps appear in final output
    final_gaps = pass2_data.get("gaps", [])
    structural_types = {g["content"][:50] for g in structural["gaps"]}

    # Check: did the LLM include all structural gaps?
    llm_structural = [g for g in final_gaps if g.get("gap_origin") == "structural"]
    if len(llm_structural) < n_struct:
        logger.warning(
            f"[Gaper] LLM only included {len(llm_structural)}/{n_struct} structural gaps — "
            f"injecting missing ones"
        )
        # Inject any missing structural gaps
        for sg in structural["gaps"]:
            found = any(sg["content"][:40] in g.get("description", "") for g in final_gaps)
            if not found:
                final_gaps.append({
                    "gap_origin":           "structural",
                    "gap_type":             sg["gap_type"],
                    "description":          sg["content"],
                    "significance":         "High",  # structural gaps are always high priority
                    "significance_reason":  "Proven by argument tree structure — not inferred",
                    "tree_node_ref":        sg.get("node_id", ""),
                    "references_grounder":  [],
                    "references_historian": [],
                    "references_current":   [],
                    "dead_end_revisit":     False,
                    "recurring_pattern":    False,
                    "recurring_reason":     "",
                })

    # Also inject bridge needs as gaps
    for bn in structural["bridge_needs"]:
        final_gaps.append({
            "gap_origin":           "structural",
            "gap_type":             "temporal",
            "description":          f"Temporal gap: {bn['earlier_year']}-{bn['later_year']} "
                                    f"({bn['gap_years']} years) in '{bn['question'][:80]}'",
            "significance":         "Medium",
            "significance_reason":  "Bridge papers needed to connect historical to contemporary evidence",
            "tree_node_ref":        bn.get("question_id", ""),
            "references_grounder":  [],
            "references_historian": [],
            "references_current":   [],
            "dead_end_revisit":     False,
            "recurring_pattern":    False,
            "recurring_reason":     "",
        })

    pass2_data["gaps"] = final_gaps

    # ── Save to database AND tree ──────────────────────────────────────────
    # Every gap gets written into the argument tree so downstream agents see
    # a single consistent structure. Analytical gaps (found outside the tree)
    # get the same treatment as structural gaps — claim + evidence nodes.
    from core.argument_tree import TreeBuilder
    tree = TreeBuilder(run_id)

    # Find the root node to attach gap claims to
    root_nodes = tree.get_nodes_by_type("root")
    # Create a dedicated "gaps" question node under root
    gap_parent = tree.add_question(
        root_nodes[0]["node_id"] if root_nodes else None,
        "What are the identified gaps in the research landscape?",
        question_level="structural",
        agent="gaper",
    )

    saved = 0
    tree_nodes_added = 0
    for gap in final_gaps:
        if not gap.get("description"):
            continue

        gap_id = generate_id("GAP")

        # 1. Save to gaps table (existing logic)
        ok = db.insert_gap({
            "gap_id":                gap_id,
            "run_id":                run_id,
            "problem_origin":        problem,
            "gap_type":              gap.get("gap_type", "unstudied"),
            "description":           gap.get("description", ""),
            "significance":          gap.get("significance", "Medium"),
            "significance_reason":   gap.get("significance_reason", ""),
            "primary_evaluation":    gap.get("tree_node_ref", ""),
            "references_grounder":   gap.get("references_grounder", []),
            "references_historian":  gap.get("references_historian", []),
            "references_social":     gap.get("references_current", []),
            "dead_end_revisit":      1 if gap.get("dead_end_revisit") else 0,
            "recurring_pattern":     1 if gap.get("recurring_pattern") else 0,
            "recurring_reason":      gap.get("recurring_reason", ""),
        })
        if ok:
            saved += 1

        # 2. Write into argument tree as claim + evidence nodes
        origin = gap.get("gap_origin", "analytical")
        gap_type = gap.get("gap_type", "unstudied")
        significance = gap.get("significance", "Medium")

        # Confidence based on origin: structural gaps are proven, analytical are inferred
        confidence = 0.9 if origin == "structural" else 0.6

        # Create a claim node for this gap
        claim_id = tree.add_claim(
            gap_parent,
            f"[{gap_type}] [{significance}] {gap.get('description', '')[:300]}",
            confidence=confidence,
            source_ids=[],
            agent="gaper",
        )
        tree_nodes_added += 1

        # Collect all referenced source titles
        all_refs = (
            gap.get("references_grounder", []) +
            gap.get("references_historian", []) +
            gap.get("references_current", [])
        )

        # For each referenced source, try to find its source_id in the DB
        # and add an evidence node linking the gap claim to the source
        for ref_title in all_refs:
            if not ref_title:
                continue
            # Search sources table for this title
            import sqlite3 as _sql
            _conn = _sql.connect(str(db.DB_PATH))
            _conn.row_factory = _sql.Row
            _row = _conn.execute(
                "SELECT source_id, type FROM sources WHERE run_id = ? AND title LIKE ? LIMIT 1",
                (run_id, f"%{ref_title[:50]}%")
            ).fetchone()
            _conn.close()

            if _row:
                source_id = _row["source_id"]
                source_type = _row["type"] or "paper"
                # Map source type to evidence type
                ev_type = {
                    "seminal": "paper", "historical": "paper",
                    "current": "paper",
                }.get(source_type, "paper")

                tree.add_evidence(
                    claim_id, source_id,
                    evidence_type=ev_type,
                    relationship="confirms_gap",
                    snippet=f"Referenced in {gap_type} gap: {ref_title[:100]}",
                    agent="gaper",
                    metadata={
                        "gap_id": gap_id,
                        "gap_origin": origin,
                        "gap_significance": significance,
                    },
                )
                tree_nodes_added += 1
            else:
                # Source not in DB — still record as evidence with title only
                tree.add_evidence(
                    claim_id, "",
                    evidence_type="other",
                    relationship="confirms_gap",
                    snippet=f"Referenced but not in DB: {ref_title[:150]}",
                    agent="gaper",
                    metadata={
                        "gap_id": gap_id,
                        "gap_origin": origin,
                        "unresolved_ref": ref_title,
                    },
                )
                tree_nodes_added += 1

        # If structural gap has a tree_node_ref, mark relationship
        if origin == "structural" and gap.get("tree_node_ref"):
            tree.update_status(gap["tree_node_ref"], "unsupported")

    tree.close()

    # ── Save artifact ─────────────────────────────────────────────────────
    _save_doc(run_id, problem, structural, pass1_data, pass2_data)

    structural_count = sum(1 for g in final_gaps if g.get("gap_origin") == "structural")
    analytical_count = sum(1 for g in final_gaps if g.get("gap_origin") == "analytical")
    high = sum(1 for g in final_gaps if g.get("significance") == "High")

    print(f"  [Gaper] {saved} gaps saved | {structural_count} structural + "
          f"{analytical_count} analytical | {high} High significance | "
          f"{tree_nodes_added} tree nodes added")
    logger.info("[Gaper] Complete")


# ─── Artifact writer ──────────────────────────────────────────────────────────

def _save_doc(run_id: str, problem: str, structural: dict,
              pass1_data: dict, pass2_data: dict):
    from pathlib import Path
    path = Path(__file__).parent.parent / "artifacts" / f"{run_id}_gaper_gaps.md"
    path.parent.mkdir(exist_ok=True)

    final_gaps = pass2_data.get("gaps", [])

    lines = [
        f"# Gap Map — Gaper (Tree-Native)",
        f"**Run:** {run_id} | **Problem:** {problem}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Tree nodes:** {structural['stats'].get('total_nodes',0)} | "
        f"**Structural gaps:** {len(structural['gaps'])} | "
        f"**Bridge needs:** {len(structural['bridge_needs'])}",
        "", "---", "",
        "## Tree Health Assessment", "",
        pass1_data.get("tree_observations", ""),
        "", "---", "",
        "## Gap Map Summary", "",
        pass2_data.get("gap_map_summary", ""),
        "", "---", "",
    ]

    # Structural gaps first (proven)
    struct_gaps = [g for g in final_gaps if g.get("gap_origin") == "structural"]
    if struct_gaps:
        lines += ["## Structural Gaps (proven by argument tree)", ""]
        for g in struct_gaps:
            icon = {"High": "🔴", "Medium": "🟡", "Low": "⚪"}.get(g.get("significance"), "")
            lines.append(f"- {icon} **[{g.get('gap_type','')}]** {g.get('description','')}")
            lines.append(f"  *{g.get('significance_reason','')}")
            if g.get("tree_node_ref"):
                lines.append(f"  Tree node: `{g['tree_node_ref']}`")
            refs = (g.get("references_grounder",[]) + g.get("references_historian",[])
                    + g.get("references_current",[]))
            if refs:
                lines.append(f"  Sources: {'; '.join(refs[:5])}")
            lines.append("")

    # Analytical gaps (inferred)
    anal_gaps = [g for g in final_gaps if g.get("gap_origin") == "analytical"]
    if anal_gaps:
        lines += ["## Analytical Gaps (identified by LLM analysis)", ""]
        for sig in ["High", "Medium", "Low"]:
            sig_gaps = [g for g in anal_gaps if g.get("significance") == sig]
            if sig_gaps:
                for g in sig_gaps:
                    lines.append(f"- **[{g.get('gap_type','')}]** {g.get('description','')}")
                    lines.append(f"  *{g.get('significance_reason','')}")
                    refs = (g.get("references_grounder",[]) + g.get("references_historian",[])
                            + g.get("references_current",[]))
                    if refs:
                        lines.append(f"  Sources: {'; '.join(refs[:5])}")
                    if g.get("recurring_pattern"):
                        lines.append(f"  ⟳ Recurring: {g.get('recurring_reason','')}")
                    if g.get("dead_end_revisit"):
                        lines.append(f"  ↩ Dead end worth revisiting")
                    lines.append("")

    path.write_text("\n".join(lines))
    logger.info(f"[Gaper] Gap map saved: {path}")
