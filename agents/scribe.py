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
from core import references
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
"understanding_map": """You are the Scribe agent producing a RESEARCH UNDERSTANDING MAP.

This is the core mandatory output — generated for every pipeline run regardless of what the researcher requested.
Its purpose is NOT to summarise the findings. Its purpose is to actively guide the researcher through the intellectual territory so they can achieve genuine comprehension, not just awareness.

Format: Markdown (.md)
Audience: The researcher conducting this investigation — someone who has run this pipeline and now needs to deeply understand the field before proposing their own contribution.

---

CITATION REQUIREMENT (CRITICAL):

You will be given a CITABLE SOURCES MANIFEST at the end of the context. Each entry has a short citation key in square brackets, e.g. `[Freire1970]` or `[Nabulsi2009]`.

Every substantive assertion in your Understanding Map — every claim about the field's history, its debates, its foundational works, its unresolved questions — must end with one or more `[CiteKey]` markers drawn from the manifest. The citation marks which source from the pipeline's retrieval supports the claim.

Rules:
- Use ONLY cite keys that appear in the manifest. Do not invent keys.
- Do not cite a source you could not plausibly defend as supporting the claim.
- If a claim has no supporting source in the manifest, either drop the claim or mark it explicitly with `[no source in run]` and rephrase it as an open question.
- Place citations at the end of the sentence or clause they support, in square brackets: `...the field divided over the question of intentionality [Nabulsi2009, Wind2024].`
- In the Reading Curriculum (section 3), each listed work MUST use its manifest cite key in the title line, formatted as: `**[CiteKey]** Title — Author, Year`.
- Do NOT add a References section yourself — one will be appended automatically after your output.

---

STRUCTURE (follow this exactly):

## 1. The Territory at a Glance
A single dense paragraph (150-200 words) that frames the intellectual landscape. Not a summary — a map legend. What are the 2-3 central tensions that organise this entire field? What is the one question that, if answered, would unlock everything else? What kind of field is this — one with empirical consensus but conceptual confusion, or one with competing frameworks and no shared method?

## 2. The Intellectual Genealogy — How We Got Here
A narrative (300-400 words) tracing the intellectual lineage from foundational ideas to the present. Written as a story of ideas, not a list of papers. Show causality: who was responding to whom, what broke what framework, what vindicated what dismissed approach. The researcher must feel the trajectory, not just know the milestones.

## 3. The Reading Curriculum
Organise the seminal works into THREE tiers. For each work:
  - **[CiteKey]** Title — Author, Year
  - Why you read it at this tier (not what it says — why the ORDER matters)
  - What to look for while reading (2-3 specific active reading prompts)
  - What it connects to (which other works it responds to or anticipates, by cite key)

**Tier 1 — Foundations (read first):** Works that establish the basic vocabulary and frame the problem. Without these, later works are opaque.

**Tier 2 — The Main Debates (read second):** Works where the central tensions crystallised. These are the works where the field divided — understanding each position here is essential before engaging with current literature.

**Tier 3 — The Current Frontier (read third):** Recent work that represents where the field is now. Read these AFTER the foundations — their significance is only visible against the historical background.

## 4. The Conceptual Map — How the Ideas Connect
A structured prose section (250-350 words) describing the relationships between key concepts, NOT between papers. Which concepts are contested? Which definitions are doing hidden theoretical work? Where does an apparent consensus actually rest on an unresolved disagreement one level deeper? This section should be readable as a standalone guide to the intellectual architecture. Cite sources at claim level.

## 5. The Unresolved Core
Identify the single most important unresolved question in the field — the one that the pipeline found unanswered and that the researcher's own work could engage with. Explain:
  - Why this question remains open (is it empirically underdetermined? philosophically contested? methodologically blocked?) — with citations for prior positions
  - What a genuine contribution to this question would require
  - What distinguishes a superficial engagement with this question from a deep one

## 6. Self-Assessment Questions — Test Your Understanding
Generate exactly 8 questions. These are NOT factual recall questions. They are Socratic questions that test whether the researcher has understood the STRUCTURE of the field — the tensions, the assumptions, the logical dependencies.

For each question:
  - State the question clearly
  - Provide the answer (2-4 sentences), with citations to manifest sources
  - Explain why this question matters for the researcher's own work

Questions should test:
  - Whether the researcher understands WHY a debate is unresolvable (not just what the positions are)
  - Whether the researcher can identify hidden assumptions in dominant frameworks
  - Whether the researcher understands what a genuine contribution would require
  - Whether the researcher can distinguish empirical gaps from conceptual ones
  - Whether the researcher understands the historical reasons why certain approaches were abandoned

---

RULES:
- Every reading recommendation must come from the manifest — do not invent sources
- The active reading prompts must be specific to each work — not generic "take notes as you read"
- The assessment questions must have correct, substantive answers grounded in the pipeline outputs
- Do not summarise the pipeline outputs — transform them into intellectual guidance
- The tone is that of a rigorous academic supervisor preparing a graduate student for their first major reading
- Every substantive claim must carry a `[CiteKey]` drawn from the manifest

Output ONLY the markdown content. No preamble. No References section (that will be appended automatically).""",
}
FORMAT_MAP = {
    "blog_post":         "md",
    "research_brief":    "md",
    "internal_memo":     "md",
    "literature_review": "tex",
    "paper_section":     "tex",
    "grant_background":  "tex",
    "understanding_map": "md",
}


def run(context: str, run_id: str, output_type: str = "research_brief",
        audience: str = "researcher", **kwargs):
    logger.info(f"[Scribe] Starting for run {run_id} — output: {output_type}")

    problem = ""
    if "PROBLEM:" in context:
        problem = context.split("PROBLEM:")[1].split("\n\n")[0].strip()

    # Understanding Map has a dedicated path with citation machinery
    if output_type == "understanding_map":
        return _run_understanding_map(context, run_id, problem, audience,
                                       verify_online=kwargs.get("verify_online", True))

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
        "blog_post":         "Blog Post",
        "research_brief":    "Research Brief",
        "internal_memo":     "Internal Memo",
        "literature_review": "Literature Review",
        "paper_section":     "Paper Section",
        "grant_background":  "Grant Background",
        "understanding_map": "Understanding Map",
    }.get(output_type, "Research Output")
    short_problem = problem[:60] + ("..." if len(problem) > 60 else "")
    return f"{prefix}: {short_problem}"


# ---------------------------------------------------------------------------
# Understanding Map — dedicated path with citation machinery
# ---------------------------------------------------------------------------

def _run_understanding_map(context: str, run_id: str, problem: str,
                           audience: str, verify_online: bool = True) -> None:
    """
    Dedicated Scribe path for the Understanding Map.

    Stages:
      1. Build citable manifest from this run's sources.
      2. Generate Understanding Map with [CiteKey] markers.
      3. Redact any CiteKeys that are not in the manifest (LLM hallucinations).
      4. Semantically validate each claim-citation pair; drop weak ones.
      5. Online-verify the cited sources (Crossref/OpenAlex/URL HEAD).
      6. Render References section + companion .tex bibitems file.
      7. Write artifacts and persist.
    """
    # --- Stage 1: manifest ---------------------------------------------------
    manifest = references.build_manifest(run_id)
    if not manifest:
        logger.warning(f"[Scribe] No sources in run {run_id} — Understanding Map "
                       f"will be generated without citations.")
    manifest_by_key = {s.cite_key: s for s in manifest}

    # --- Stage 2: generate ---------------------------------------------------
    system_prompt = SYSTEM_PROMPTS["understanding_map"].format(audience=audience)
    manifest_text = references.format_manifest_for_prompt(manifest)
    enriched_context = (
        f"{context}\n\n"
        f"---\n"
        f"CITABLE SOURCES MANIFEST ({len(manifest)} entries):\n"
        f"Format: [CiteKey] Authors (Year). Title  Abstract: ...\n"
        f"Cite using the [CiteKey] markers below. Do not invent keys.\n\n"
        f"{manifest_text}\n"
    )
    try:
        response = llm.call(enriched_context, system_prompt, agent_name="scribe")
    except Exception as e:
        logger.error(f"[Scribe] LLM call failed: {e}")
        raise

    content = response.strip()
    content = re.sub(r"^```(?:markdown|md)?\n?", "", content)
    content = re.sub(r"\n?```$", "", content)
    content = content.strip()

    # --- Stage 3: redact hallucinated keys ----------------------------------
    valid_keys = set(manifest_by_key.keys())
    unknown = references.find_unknown_cite_keys(content, valid_keys)
    if unknown:
        logger.warning(f"[Scribe] Redacting {len(unknown)} unknown cite keys: {unknown}")
        content = _redact_unknown_keys(content, valid_keys, unknown)

    cited_keys = references.extract_cite_keys(content, valid_keys)
    logger.info(f"[Scribe] Understanding Map cites {len(cited_keys)} sources")

    # --- Stage 4: semantic validation ---------------------------------------
    # Extract claim-citation pairs: for each sentence containing [CiteKey],
    # treat the sentence as the claim.
    claims_and_cites = _extract_claims_for_validation(content, valid_keys)
    if claims_and_cites:
        verdicts = references.validate_citation_claims(claims_and_cites, manifest_by_key)
        weak_pairs = []  # (claim_text, cite_key)
        for entry in verdicts:
            for v in entry["verdicts"]:
                if v.get("plausible") is False and v.get("confidence") in ("high", "medium"):
                    weak_pairs.append((entry["claim"], v["cite_key"], v.get("reason", "")))
        if weak_pairs:
            logger.warning(f"[Scribe] {len(weak_pairs)} citations flagged as "
                           f"semantically weak; marking in document.")
            # Conservatively: annotate rather than silently drop. Let the researcher see.
            for claim, key, reason in weak_pairs:
                # Replace [key] with [key?] to flag in the rendered doc
                # (only first occurrence in that claim, best effort)
                content = content.replace(f"[{key}]", f"[{key}?]", 1)

    # Recompute cited keys after redactions (strip '?' suffix for manifest lookup)
    cited_keys_final = references.extract_cite_keys(
        re.sub(r"\[(\w+)\?\]", r"[\1]", content),
        valid_keys
    )
    cited_sources = [manifest_by_key[k] for k in cited_keys_final if k in manifest_by_key]

    # --- Stage 5: online verification ---------------------------------------
    if verify_online and cited_sources:
        logger.info(f"[Scribe] Verifying {len(cited_sources)} cited sources online...")
        try:
            references.verify_online(cited_sources, use_cache=True)
        except Exception as e:
            logger.warning(f"[Scribe] Online verification failed: {e}")

    # --- Stage 6: render References section + .tex companion ----------------
    refs_md = references.render_references_markdown(cited_sources)
    refs_tex = references.render_references_tex(cited_sources)

    full_md = content + "\n\n---\n\n" + refs_md

    # --- Stage 7: write artifacts + persist ---------------------------------
    md_path = ARTIFACTS_DIR / f"{run_id}_understanding_map.md"
    md_path.write_text(full_md, encoding="utf-8")
    logger.info(f"[Scribe] Understanding Map written: {md_path}")

    tex_path = ARTIFACTS_DIR / f"{run_id}_understanding_map_refs.tex"
    tex_path.write_text(refs_tex, encoding="utf-8")
    logger.info(f"[Scribe] Reference .tex written: {tex_path}")

    # Save to DB — one artifact record per file for traceability
    synthesis = db.get_synthesis(run_id)
    directions = db.get_directions(run_id)
    base_meta = {
        "run_id":          run_id,
        "problem_origin":  problem,
        "output_type":     "understanding_map",
        "audience":        audience,
        "synthesis_id":    synthesis.get("synthesis_id", "") if synthesis else "",
        "directions_used": [d["direction_id"] for d in directions],
    }
    db.insert_artifact({
        **base_meta,
        "artifact_id": generate_id("ART"),
        "format":      "md",
        "title":       _make_title(problem, "understanding_map"),
        "file_path":   str(md_path),
        "word_count":  len(full_md.split()),
    })
    db.insert_artifact({
        **base_meta,
        "artifact_id": generate_id("ART"),
        "format":      "tex",
        "title":       _make_title(problem, "understanding_map") + " — References",
        "file_path":   str(tex_path),
        "word_count":  len(refs_tex.split()),
    })

    # Summary to stdout
    ok_cnt = sum(1 for s in cited_sources if s.exists_online)
    dead_cnt = sum(1 for s in cited_sources if s.exists_online is False)
    print(f"  [Scribe] Understanding Map → {md_path.name}")
    print(f"  [Scribe] Reference .tex    → {tex_path.name}")
    print(f"  [Scribe] Citations: {len(cited_sources)} sources "
          f"({ok_cnt} verified online, {dead_cnt} unverified)")


# ---------------------------------------------------------------------------
# Claim extraction for semantic validation
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _extract_claims_for_validation(text: str, valid_keys: set[str]) -> list[dict]:
    """
    Walk the rendered text, find sentences that carry one or more valid
    [CiteKey] markers, and return [{"claim": sentence, "cite_keys": [...]}].
    """
    claims = []
    # Skip the Reading Curriculum bullets where [CiteKey] is a label, not a claim.
    # We detect that by checking if the citation is followed immediately by an em-dash
    # or by the start of a section; in that case skip.
    for para in text.split("\n\n"):
        # Skip headings
        if para.strip().startswith("#"):
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(para):
            sentence = sentence.strip()
            if not sentence:
                continue
            keys = [m.group(1) for m in references.CITE_KEY_PATTERN.finditer(sentence)
                    if m.group(1) in valid_keys]
            if not keys:
                continue
            # Heuristic: if the sentence starts with "**[Key]**" it's a reading-list entry,
            # not a claim — skip.
            if re.match(r"^\s*\*{0,2}\[\w+\]\*{0,2}\s*[—-]", sentence):
                continue
            claims.append({"claim": sentence, "cite_keys": keys})
    return claims


def _redact_unknown_keys(text: str, valid_keys: set, unknown_keys: list) -> str:
    """
    Surgically remove hallucinated keys while preserving valid ones in the
    same bracket. [Good, Bad] -> [Good]. [Bad] alone -> [unsupported].
    """
    unknown_set = set(unknown_keys)

    def _fix_bracket(match):
        inner = match.group(1)
        keys = [k.strip() for k in inner.split(",") if k.strip()]
        kept = [k for k in keys if k in valid_keys and k not in unknown_set]
        if kept:
            return f"[{', '.join(kept)}]"
        return "[unsupported]"

    return references.CITE_KEY_PATTERN.sub(_fix_bracket, text)
