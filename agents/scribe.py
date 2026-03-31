"""
Scribe Agent
------------
Formats pipeline outputs into clean, audience-ready artifacts.
Multiple artifacts per run — each in the correct format.
  .md  → blog post, research brief, internal memo
  .tex → paper section, literature review, grant background
Saves to artifacts_database and writes actual files.
"""

import re
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from core import database as db
from core import llm
from core.utils import generate_id

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LaTeX template
# ---------------------------------------------------------------------------

LATEX_PREAMBLE = r"""\documentclass[12pt, a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{setspace}
\onehalfspacing
\usepackage{natbib}
\bibliographystyle{apalike}
\usepackage{hyperref}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{booktabs}

"""

# ---------------------------------------------------------------------------
# System prompts per output type
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {

"blog_post": """You are the Scribe agent producing a BLOG POST.

Format: Markdown (.md)
Audience: {audience}
Tone: Accessible, engaging, written for an informed but non-specialist reader.

Structure:
1. Hook — why this problem matters right now
2. Background — key origins and why the question is hard
3. What we know and don't know — gap landscape in plain language
4. Promising directions — most viable proposals from Rude + new directions from Thinker
5. What to watch — closing with the most important open question

Rules:
- No jargon without explanation
- Concrete examples and analogies
- 800-1200 words
- Use markdown headers, bold for emphasis
- Do NOT invent facts — only use what the pipeline established
- Cite key thinkers and works naturally in prose

Output ONLY the markdown content. No preamble.""",

"research_brief": r"""You are the Scribe agent producing a RESEARCH BRIEF.

Format: Markdown (.md)
Audience: {audience}
Tone: Dense, sharp, every sentence earns its place.

Structure:
1. Problem Statement (2-3 sentences)
2. Current State (what is known, contested, unknown)
3. Key Gaps (top 3-5, ranked by significance)
4. Viable Approaches (proposals that passed Rude's evaluation)
5. Recommended Trajectory (trajectory statement from Synthesizer)
6. Key Uncertainties (tensions flagged by Synthesizer)

Rules:
- No unnecessary elaboration
- Bullet points and short paragraphs
- 400-600 words
- Every claim traceable to pipeline output

Output ONLY the markdown content. No preamble.""",

"internal_memo": """You are the Scribe agent producing an INTERNAL RESEARCH MEMO.

Format: Markdown (.md)
Audience: {audience}
Tone: Complete and honest — written for yourself or a close collaborator.

Structure:
1. Problem and run context
2. What the pipeline established (full picture)
3. Gap landscape — complete, including low significance
4. Proposals — all proposals with verdicts including rejected ones
5. Tensions and contradictions — do not smooth over
6. Break 1 overrides — where your judgment diverged
7. New directions from Thinker
8. Next steps

Rules:
- Include everything, even what was rejected and why
- Note where the pipeline was uncertain
- 800-1500 words
- This is for your own research record — be thorough

Output ONLY the markdown content. No preamble.""",

"literature_review": r"""You are the Scribe agent producing a LITERATURE REVIEW SECTION.

Format: LaTeX (.tex) — body only, no preamble, ready to \input into a larger document
Audience: {audience}
Tone: Formal, precise, specialist audience.

Structure:
- Organized by themes, NOT chronology
- Every claim cited — use \cite{{key}} placeholders where you would cite
- Identify gaps explicitly as part of the review
- Follow academic literature review conventions

Rules:
- Use proper LaTeX sectioning (\subsection, \paragraph)
- Citations as \cite{{AuthorYear}} placeholders
- 600-1000 words
- No invented facts — only what pipeline established

Output ONLY the LaTeX body content. No \begin{{document}}.""",

"paper_section": r"""You are the Scribe agent producing a PAPER SECTION.

Format: LaTeX (.tex) — body only, ready to \input into a larger paper
Audience: {audience}
Tone: Formal academic, specialist audience.

Produce whichever section is most appropriate given the pipeline outputs:
- Introduction (if problem framing is the main output)
- Related Work (if literature mapping is the main output)
- Motivation/Background (if gap analysis is the main output)

Rules:
- Proper LaTeX sectioning
- Citations as \cite{{AuthorYear}} placeholders
- 500-900 words
- Integrate seamlessly into a larger paper

Output ONLY the LaTeX body content. No \begin{{document}}.""",

"grant_background": """You are the Scribe agent producing a GRANT/PROPOSAL BACKGROUND SECTION.

Format: LaTeX (.tex) — body only
Audience: {audience}
Tone: Persuasive and precise — makes the case for why this problem matters and why now.

Structure:
1. Significance — why this problem matters
2. Current state of knowledge — what is established
3. Critical gap — the specific gap this work addresses
4. Novelty — why this approach is new and why now

Rules:
- Persuasive framing while remaining accurate
- Citations as \\cite{{AuthorYear}} placeholders
- 400-700 words
- Every claim grounded in pipeline outputs

Output ONLY the LaTeX body content. No \\begin{{document}}.""",
}

# Format mapping
FORMAT_MAP = {
    "blog_post":        "md",
    "research_brief":   "md",
    "internal_memo":    "md",
    "literature_review": "tex",
    "paper_section":    "tex",
    "grant_background": "tex",
}


def run(context: str, run_id: str, output_type: str = "research_brief",
        audience: str = "researcher", **kwargs):
    logger.info(f"[Scribe] Starting for run {run_id} — output: {output_type}")

    problem = ""
    if "PROBLEM:" in context:
        problem = context.split("PROBLEM:")[1].split("\n\n")[0].strip()

    # Get system prompt for this output type
    system_template = SYSTEM_PROMPTS.get(output_type, SYSTEM_PROMPTS["research_brief"])
    system_prompt = system_template.format(audience=audience)

    try:
        response = llm.call(context, system_prompt, agent_name="scribe")
    except Exception as e:
        logger.error(f"[Scribe] LLM call failed: {e}")
        raise

    # Determine format
    fmt = FORMAT_MAP.get(output_type, "md")

    # Clean response
    content = response.strip()
    # Strip markdown fences if LLM added them
    content = re.sub(r"^```(?:markdown|latex|tex|md)?\n?", "", content)
    content = re.sub(r"\n?```$", "", content)
    content = content.strip()

    # For LaTeX outputs, wrap in full document
    if fmt == "tex":
        title = _make_title(problem, output_type)
        full_content = (
            LATEX_PREAMBLE +
            f"\\title{{{title}}}\n"
            f"\\author{{Pipeline Run: {run_id}}}\n"
            f"\\date{{{datetime.now(timezone.utc).strftime('%B %Y')}}}\n\n"
            f"\\begin{{document}}\n"
            f"\\maketitle\n\n"
            f"{content}\n\n"
            f"\\end{{document}}\n"
        )
    else:
        full_content = content

    # Save file
    filename = f"{run_id}_{output_type}.{fmt}"
    file_path = ARTIFACTS_DIR / filename
    file_path.write_text(full_content, encoding="utf-8")
    logger.info(f"[Scribe] Artifact written: {file_path}")

    # Save to database
    synthesis = db.get_synthesis(run_id)
    directions = db.get_directions(run_id)

    db.insert_artifact({
        "artifact_id":      generate_id("ART"),
        "run_id":           run_id,
        "problem_origin":   problem,
        "output_type":      output_type,
        "format":           fmt,
        "title":            _make_title(problem, output_type),
        "audience":         audience,
        "synthesis_id":     synthesis.get("synthesis_id", "") if synthesis else "",
        "directions_used":  [d["direction_id"] for d in directions],
        "file_path":        str(file_path),
        "word_count":       len(content.split()),
    })

    print(f"  [Scribe] [{output_type}] artifact saved → {file_path.name}")
    print(f"  [Scribe] Format: .{fmt} | Words: ~{len(content.split())}")
    logger.info(f"[Scribe] Complete — {output_type}")


def _make_title(problem: str, output_type: str) -> str:
    """Generate a short title from the problem."""
    prefix = {
        "blog_post":        "Blog Post",
        "research_brief":   "Research Brief",
        "internal_memo":    "Internal Memo",
        "literature_review": "Literature Review",
        "paper_section":    "Paper Section",
        "grant_background": "Grant Background",
    }.get(output_type, "Research Output")
    short_problem = problem[:60] + ("..." if len(problem) > 60 else "")
    return f"{prefix}: {short_problem}"
