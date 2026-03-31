"""
Pipeline Runner
---------------
Main entry point. Orchestrates the full pipeline:

  Social feed → Break 0 → Grounder → Historian → Gaper
  → Break 1 → Vision → Theorist → Rude → Synthesizer
  → Break 2 → Thinker → Scribe

Usage:
  python3 main.py run  --problem "Your research problem here"
  python3 main.py run  --problem "..." --run-id RUN-20260330-XXXX  (resume)
  python3 main.py collect                                            (Social passive scan)
  python3 main.py recheck                                            (link health check)
  python3 main.py status --run-id RUN-20260330-XXXX                 (check run status)
  python3 main.py bank                                               (show seminal bank proposals)
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Ensure pipeline root is in path
sys.path.insert(0, str(Path(__file__).parent))

# Load .env FIRST — before any module that reads os.environ at import time
from core.keys import _load_env
_load_env()

from core.utils     import setup_logging, generate_run_id, load_config
from core           import database as db
from core           import breaks
from core.context   import (
    for_grounder, for_historian, for_gaper,
    for_vision, for_theorist, for_rude,
    for_synthesizer, for_thinker, for_scribe
)
from agents.social  import feed as social_feed, collect as social_collect
from agents.social  import produce_intelligence_package, recheck_links
from core.concept_mapper import expand as concept_expand, print_expansion_report


# ---------------------------------------------------------------------------
# Agent imports — each agent exposes a run(context, run_id) function
# ---------------------------------------------------------------------------

def _import_agents():
    """Lazy import agents to give clear error if one is missing."""
    from agents.grounder    import run as run_grounder
    from agents.historian   import run as run_historian
    from agents.gaper       import run as run_gaper
    from agents.vision      import run as run_vision
    from agents.theorist    import run as run_theorist
    from agents.rude        import run as run_rude
    from agents.synthesizer import run as run_synthesizer
    from agents.thinker     import run as run_thinker
    from agents.scribe      import run as run_scribe
    return {
        "grounder":    run_grounder,
        "historian":   run_historian,
        "gaper":       run_gaper,
        "vision":      run_vision,
        "theorist":    run_theorist,
        "rude":        run_rude,
        "synthesizer": run_synthesizer,
        "thinker":     run_thinker,
        "scribe":      run_scribe,
    }


# ---------------------------------------------------------------------------
# Pipeline step runner
# ---------------------------------------------------------------------------

def _run_step(
    step_name: str,
    agent_fn,
    context: str,
    run_id: str,
    extra: dict = None
) -> bool:
    """
    Run a single pipeline step with error handling.
    Returns True on success, False on failure.
    """
    logger = logging.getLogger("pipeline")
    print(f"\n{'─'*60}")
    print(f"  ▶  {step_name.upper()}")
    print(f"{'─'*60}")
    logger.info(f"Starting step: {step_name}")

    try:
        if extra:
            agent_fn(context, run_id, **extra)
        else:
            agent_fn(context, run_id)
        logger.info(f"Step complete: {step_name}")
        print(f"  ✓  {step_name} complete")
        return True
    except Exception as e:
        logger.error(f"Step failed: {step_name} — {e}", exc_info=True)
        print(f"  ✗  {step_name} FAILED: {e}")
        return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(problem: str, run_id: str = None, resume: bool = False):
    logger = logging.getLogger("pipeline")

    # Setup
    if run_id is None:
        run_id = generate_run_id()
    log = setup_logging(run_id)
    db.init_db()
    config = load_config()

    print(f"\n{'='*60}")
    print(f"  MULTI-AGENT RESEARCH PIPELINE")
    print(f"{'='*60}")
    print(f"  Run ID:  {run_id}")
    print(f"  Problem: {problem[:70]}{'...' if len(problem) > 70 else ''}")
    print(f"  Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}\n")

    # Create or retrieve run
    existing_run = db.get_run(run_id)
    if existing_run and not resume:
        print(f"Run {run_id} already exists. Use --resume to continue.")
        return
    if not existing_run:
        db.create_run(run_id, problem)
        logger.info(f"New run created: {run_id}")

    run = db.get_run(run_id)

    # -----------------------------------------------------------------------
    # SOCIAL FEED + BREAK 0
    # -----------------------------------------------------------------------

    if not run.get("break0_done"):
        # Concept mapper — translate problem into conceptual territory
        print("\n▶  CONCEPT MAPPER — Semantic expansion...")
        try:
            expansion = concept_expand(problem, run_id, config)
            print_expansion_report(expansion)
            # Override selected themes with concept mapper results
            activated_theme_ids = expansion["final_themes"]
            selected_themes = [t for t in config.get("themes", [])
                               if t["theme_id"] in activated_theme_ids]
            excluded_themes = [{"theme_id": t["theme_id"], "label": t.get("label",""),
                                 "reason": "Not activated by concept mapper"}
                                for t in config.get("themes", [])
                                if t["theme_id"] not in activated_theme_ids]
            logger.info(f"Concept mapper activated {len(selected_themes)} themes")
        except Exception as e:
            logger.warning(f"Concept mapper failed ({e}) — falling back to all themes")
            selected_themes = config.get("themes", [])
            excluded_themes = []
            expansion = None

        print("\n▶  SOCIAL — Targeted intelligence pull...")
        try:
            selected_themes, excluded_themes = social_feed(problem, run_id, config,
                                                           selected_themes=selected_themes)
            intel_pkg = produce_intelligence_package(run_id, selected_themes, problem)
            logger.info(f"Social feed complete — {len(selected_themes)} themes selected")
            print(f"  ✓  Social feed complete — {len(selected_themes)} themes selected")
        except Exception as e:
            logger.error(f"Social feed failed: {e}", exc_info=True)
            print(f"  ✗  Social feed failed: {e}")
            selected_themes, excluded_themes = [], []
            intel_pkg = f"Problem: {problem}\n\nNo Social intelligence available."

        # Break 0
        break0_instructions = breaks.break0(
            run_id, problem, selected_themes, excluded_themes
        )
        logger.info(f"Break 0 instructions received: {len(break0_instructions)} chars")
    else:
        print("  ↩  Break 0 already completed — resuming")
        break0_instructions = "CONFIRMED"
        selected_themes = config.get("themes", [])

    # -----------------------------------------------------------------------
    # GROUNDER
    # -----------------------------------------------------------------------

    agents = _import_agents()

    grounder_ctx = for_grounder(run_id, problem, db.get_sources_by_type("current", run_id))
    if not _run_step("Grounder", agents["grounder"], grounder_ctx, run_id):
        _abort(run_id, "Grounder")
        return

    # -----------------------------------------------------------------------
    # HISTORIAN
    # -----------------------------------------------------------------------

    historian_ctx = for_historian(run_id, problem)
    if not _run_step("Historian", agents["historian"], historian_ctx, run_id):
        _abort(run_id, "Historian")
        return

    # -----------------------------------------------------------------------
    # GAPER
    # -----------------------------------------------------------------------

    gaper_ctx = for_gaper(run_id, problem)
    if not _run_step("Gaper", agents["gaper"], gaper_ctx, run_id):
        _abort(run_id, "Gaper")
        return

    # -----------------------------------------------------------------------
    # BREAK 1
    # -----------------------------------------------------------------------

    if not run.get("break1_done"):
        break1_instructions = breaks.break1(run_id, problem)
        logger.info(f"Break 1 instructions received: {len(break1_instructions)} chars")
    else:
        print("  ↩  Break 1 already completed — resuming")
        break1_instructions = "CONFIRMED"

    # -----------------------------------------------------------------------
    # VISION
    # -----------------------------------------------------------------------

    vision_ctx = for_vision(run_id, problem, break1_instructions)
    if not _run_step("Vision", agents["vision"], vision_ctx, run_id):
        _abort(run_id, "Vision")
        return

    # -----------------------------------------------------------------------
    # THEORIST
    # -----------------------------------------------------------------------

    theorist_ctx = for_theorist(run_id, problem, break1_instructions)
    if not _run_step("Theorist", agents["theorist"], theorist_ctx, run_id):
        _abort(run_id, "Theorist")
        return

    # -----------------------------------------------------------------------
    # RUDE
    # -----------------------------------------------------------------------

    rude_ctx = for_rude(run_id, problem, break1_instructions)
    if not _run_step("Rude", agents["rude"], rude_ctx, run_id):
        _abort(run_id, "Rude")
        return

    # -----------------------------------------------------------------------
    # SYNTHESIZER
    # -----------------------------------------------------------------------

    synthesizer_ctx = for_synthesizer(run_id, problem, break1_instructions)
    if not _run_step("Synthesizer", agents["synthesizer"], synthesizer_ctx, run_id):
        _abort(run_id, "Synthesizer")
        return

    # -----------------------------------------------------------------------
    # BREAK 2
    # -----------------------------------------------------------------------

    if not run.get("break2_done"):
        break2_instructions = breaks.break2(run_id, problem)
        logger.info(f"Break 2 instructions received: {len(break2_instructions)} chars")
    else:
        print("  ↩  Break 2 already completed — resuming")
        break2_instructions = "CONFIRMED\nSCRIBE OUTPUT: research_brief | audience: researcher"

    # Parse Scribe output requests
    scribe_requests = breaks.parse_scribe_requests(break2_instructions)

    # -----------------------------------------------------------------------
    # THINKER
    # -----------------------------------------------------------------------

    thinker_ctx = for_thinker(run_id, problem, break2_instructions)
    if not _run_step("Thinker", agents["thinker"], thinker_ctx, run_id):
        _abort(run_id, "Thinker")
        return

    # -----------------------------------------------------------------------
    # SCRIBE — one artifact per requested output type
    # -----------------------------------------------------------------------

    for req in scribe_requests:
        output_type = req["output_type"]
        audience    = req["audience"]
        scribe_ctx  = for_scribe(run_id, problem, output_type, audience, break2_instructions)
        if not _run_step(
            f"Scribe [{output_type}]",
            agents["scribe"],
            scribe_ctx,
            run_id,
            extra={"output_type": output_type, "audience": audience}
        ):
            logger.warning(f"Scribe failed for output type: {output_type}")

    # -----------------------------------------------------------------------
    # COMPLETE
    # -----------------------------------------------------------------------

    db.update_run_status(run_id, "completed")

    artifacts = db.get_artifacts(run_id)
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Run ID:    {run_id}")
    print(f"  Artifacts: {len(artifacts)} produced")
    for art in artifacts:
        print(f"    → [{art.get('output_type')}] {art.get('file_path','')}")
    print(f"  Database:  {db.DB_PATH}")
    print(f"  Logs:      logs/{run_id}.log")
    print(f"{'='*60}\n")
    logger.info(f"Pipeline complete: {run_id}")


def _abort(run_id: str, step: str):
    db.update_run_status(run_id, f"failed:{step}")
    logging.getLogger("pipeline").error(f"Pipeline aborted at: {step}")
    print(f"\n  Pipeline aborted at step: {step}")
    print(f"  Run ID saved: {run_id}")
    print(f"  You can resume with: python3 main.py run --problem '...' --run-id {run_id} --resume")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_run(args):
    # Check ANTHROPIC_API_KEY is available before starting
    import os
    from core import llm as llm_module
    client = llm_module.get_client()
    if client.anthropic_client is None:
        print("\n  ⚠️  WARNING: ANTHROPIC_API_KEY is not set or not loaded.")
        print("     The pipeline will use Ollama (local) for all LLM calls.")
        print("     To use Claude API: add ANTHROPIC_API_KEY to your .env file.")
        print("     Continuing in 5 seconds...\n")
        import time; time.sleep(5)
    else:
        print("\n  ✅  Claude API key loaded — using Claude for all agents.")
    run_pipeline(
        problem=args.problem,
        run_id=args.run_id,
        resume=args.resume
    )


def cmd_collect(args):
    """Run Social passive collection."""
    setup_logging("social-collect")
    db.init_db()
    config = load_config()
    print("\n▶  Social passive collection starting...")
    summary = social_collect(config)
    print(f"\n  ✓  Collection complete:")
    print(f"     Themes scanned:    {summary['themes_scanned']}")
    print(f"     Sources collected: {summary['sources_collected']}")
    print(f"     Dead links:        {summary['dead_links']}")


def cmd_recheck(args):
    """Run link health check."""
    setup_logging("link-recheck")
    db.init_db()
    print("\n▶  Link health recheck starting...")
    summary = recheck_links()
    print(f"\n  ✓  Recheck complete:")
    print(f"     Checked:     {summary['checked']}")
    print(f"     Active:      {summary['active']}")
    print(f"     Redirected:  {summary['redirected']}")
    print(f"     Dead:        {summary['dead']}")
    print(f"     Flagged:     {summary['flagged']} (seminal — manual review needed)")


def cmd_status(args):
    """Show run status."""
    db.init_db()
    run = db.get_run(args.run_id)
    if not run:
        print(f"Run not found: {args.run_id}")
        return
    print(f"\n{'='*60}")
    print(f"  Run Status: {args.run_id}")
    print(f"{'='*60}")
    print(f"  Problem:     {run['problem'][:70]}")
    print(f"  Status:      {run['status']}")
    print(f"  Created:     {run['created_at']}")
    print(f"  Break 0:     {'✓' if run['break0_done'] else '✗'}")
    print(f"  Break 1:     {'✓' if run['break1_done'] else '✗'}")
    print(f"  Break 2:     {'✓' if run['break2_done'] else '✗'}")
    print(f"  Completed:   {run.get('completed_at', '—')}")

    # Counts
    print(f"\n  Database entries:")
    for table, label in [
        ("sources",      "Sources"),
        ("gaps",         "Gaps"),
        ("implications", "Implications"),
        ("proposals",    "Proposals"),
        ("evaluations",  "Evaluations"),
        ("syntheses",    "Syntheses"),
        ("directions",   "Directions"),
        ("artifacts",    "Artifacts"),
    ]:
        n = db.count(table, {"run_id": args.run_id})
        print(f"    {label:<16} {n}")
    print()


def cmd_bank(args):
    """Show seminal bank proposals."""
    db.init_db()
    proposals = db.get_seminal_bank("pending_review")
    if not proposals:
        print("\n  No pending proposals in seminal bank.")
        return
    print(f"\n{'='*60}")
    print(f"  Seminal Bank — Pending Review ({len(proposals)} proposals)")
    print(f"{'='*60}")
    for p in proposals:
        print(f"\n  [{p['bank_id']}] {p['proposed_theme']}")
        print(f"  Reason:  {p.get('reason','')}")
        print(f"  Problem: {p.get('problem_origin','')[:60]}")
        print(f"  Date:    {p.get('date_proposed','')}")
    print()


def cmd_runs(args):
    """List recent runs."""
    db.init_db()
    runs = db.fetch("runs")
    runs.sort(key=lambda r: r.get("created_at",""), reverse=True)
    if not runs:
        print("\n  No runs found.")
        return
    print(f"\n{'='*60}")
    print(f"  Recent Runs")
    print(f"{'='*60}")
    for r in runs[:20]:
        breaks_done = sum([r.get("break0_done",0), r.get("break1_done",0), r.get("break2_done",0)])
        print(f"  {r['run_id']}  [{r['status']}]  breaks:{breaks_done}/3")
        print(f"    {r['problem'][:65]}")
    print()


def cmd_keys(args):
    """Show API key status."""
    from core.keys import print_key_status
    print_key_status()


    """List recent runs."""
    db.init_db()
    runs = db.fetch("runs")
    runs.sort(key=lambda r: r.get("created_at",""), reverse=True)
    if not runs:
        print("\n  No runs found.")
        return
    print(f"\n{'='*60}")
    print(f"  Recent Runs")
    print(f"{'='*60}")
    for r in runs[:20]:
        breaks_done = sum([r.get("break0_done",0), r.get("break1_done",0), r.get("break2_done",0)])
        print(f"  {r['run_id']}  [{r['status']}]  breaks:{breaks_done}/3")
        print(f"    {r['problem'][:60]}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  run       Run the pipeline on a problem
  collect   Run Social passive collection (twice weekly)
  recheck   Run link health check
  status    Show status of a run
  bank      Show seminal bank proposals pending review
  runs      List recent runs
        """
    )
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run the pipeline")
    p_run.add_argument("--problem", required=True, help="Research problem statement")
    p_run.add_argument("--run-id",  default=None,  help="Resume an existing run")
    p_run.add_argument("--resume",  action="store_true", help="Resume from last completed step")
    p_run.set_defaults(func=cmd_run)

    # collect
    p_collect = sub.add_parser("collect", help="Social passive collection")
    p_collect.set_defaults(func=cmd_collect)

    # recheck
    p_recheck = sub.add_parser("recheck", help="Link health check")
    p_recheck.set_defaults(func=cmd_recheck)

    # status
    p_status = sub.add_parser("status", help="Show run status")
    p_status.add_argument("--run-id", required=True, help="Run ID to check")
    p_status.set_defaults(func=cmd_status)

    # bank
    p_bank = sub.add_parser("bank", help="Show seminal bank proposals")
    p_bank.set_defaults(func=cmd_bank)

    # keys
    p_keys = sub.add_parser("keys", help="Show API key status")
    p_keys.set_defaults(func=cmd_keys)

    # runs
    p_runs = sub.add_parser("runs", help="List recent runs")
    p_runs.set_defaults(func=cmd_runs)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
