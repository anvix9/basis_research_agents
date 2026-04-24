# SEEKER

**A multi-agent research intelligence pipeline for deep, traceable academic work.**

Give it a research question. It excavates the intellectual history, maps the gaps, proposes approaches, stress-tests their feasibility, synthesises a narrative, and delivers two final documents: a Research Brief and an Understanding Map. You stay in control through three mandatory human review breaks.

**Current version: v10.4**

---

## What it does

Most research tools search. This one thinks.

Ten specialised agents run in sequence, each building on the last. They pull from live academic APIs — not LLM training data — so every finding is traceable to a real, current source. The pipeline distinguishes between what a field has established, what it has tried and abandoned, where the genuine gaps are, and what is worth proposing next. At three points, it stops and hands control back to you.

---

## Pipeline at a glance

```
    ┌─────────────────────────────────────────────────────────────┐
    │  Social → [Break 0] → Grounder → Historian → Gaper →        │
    │  [Break 1] → Vision → Theorist → Rude → Synthesizer →       │
    │  [Break 2] → Thinker → Scribe                               │
    └─────────────────────────────────────────────────────────────┘
```

| Agent | Role |
|---|---|
| **Social** | Passive collection across live data sources; seeds the run database |
| **Grounder** | Identifies seminal works and intellectual origins of the question |
| **Historian** | Traces chronology, dead ends, abandoned paradigms (dead-end doctrine) |
| **Gaper** | Maps genuine gaps — what the field has *not* answered |
| **Vision** | Extracts strong implications from the accumulated foundation |
| **Theorist** | Proposes approaches; two-pass design with elevated token ceiling (16k) |
| **Rude** | Stress-tests each proposal; finds weakest links |
| **Synthesizer** | Narrates coherence; sharpens the question; maps tensions |
| **Thinker** | Opens genuinely new directions beyond the existing proposals |
| **Scribe** | Produces final outputs: Research Brief + Understanding Map |

---

## The three breaks — human-in-the-loop supervision

Unlike autonomous agentic systems, SEEKER has three forced stops where the pipeline waits for your instructions before continuing.

| Break | What you receive | What you decide |
|---|---|---|
| **Break 0** | Social collection summary | Confirm or redirect the source pool |
| **Break 1** | Foundation review (seminal + historical + gaps) | Validate trajectory; upload instructions for Phase 2 |
| **Break 2** | Analysis review (implications + proposals + evaluations + synthesis) | Approve coherence; specify final artefact format |

Resume after any break with `--resume`; the pipeline infers completion from data presence in each agent's output table and skips work that is already done.

---

## What's new in v10.4

### Consensus integration via MCP OAuth
Consensus semantic search (200M+ peer-reviewed papers) is now available to agents via the official MCP Python SDK with OAuth 2.1 Authorization Code + PKCE. No API key required — you log in once with your Consensus account, and tokens are persisted in `db/consensus_tokens.json` and auto-refreshed. Valid search parameters: `query`, `year_min`, `year_max`, `study_types`, `sjr_max`, `human`, `sample_size_min`. Full OAuth flow documented in `OAuth_MCP_Auth_Diagram.docx`.

### Per-agent source control
New `agent_sources` block in `config.json` lets you control which sources each agent can query. Social and Grounder read their allowed sources from config, not from hardcoded lists. Consensus is disabled by default for token-cost reasons; enable it per-agent as needed.

### Resume from any break
`python3 main.py run --problem "..." --run-id RUN-XXXXXXXX --resume` picks up where a crash or cancellation left off. The `_agent_done()` helper checks for data presence in each agent's output table and skips completed work. Break 2 instructions are recovered from `_break2_review.md` so downstream agents get your real guidance, not a generic fallback.

### Understanding Map — mandatory Scribe output
Every run now produces a second artefact alongside the Research Brief: a six-section **Understanding Map** designed for the researcher to actually learn the field, not just receive a report.

1. **Territory at a Glance** — what the field looks like
2. **Intellectual Genealogy** — who built on whom
3. **Reading Curriculum** — three tiers (Foundational / Developmental / Contemporary) with active reading prompts for each paper
4. **Conceptual Map** — how the key concepts relate
5. **Unresolved Core** — what the field has not settled
6. **Self-Assessment** — 8 Socratic questions with answers

### Vision robustness
Vision's token ceiling raised to 12,000. A new `_salvage_truncated_json()` brace-counting parser recovers complete implications if the model is cut off mid-output. Strong implications are now instructed to stream first so the most important content always survives truncation.

### Break 2 truncation
Field-length caps applied per field (narrative 1500, trajectory 800, tensions 600, verdict reason 400, weakest link 200, proposal 300 chars) with `...[truncated — full text in DB]` markers. Full text is always preserved in SQLite.

### Evaluation and publishing tools
- `eval_references.py` — scores reference quality across a run
- `eval_claims.py` — audits claim-to-source attribution
- `export_seminal.py` — exports seminal papers grouped by category for blog publication. Outputs JSON (Jekyll `_data/` compatible), CSV, and optional per-paper Jekyll Markdown posts. Supports `--list-runs` and `--run` flags.

---

## Live data sources

Agents retrieve from live APIs, not LLM training data. This is a core architectural principle.

| Source | Coverage | Auth |
|---|---|---|
| OpenAlex | 250M+ works across all disciplines | API key (free) |
| Semantic Scholar | 200M+ papers with citation graph | API key recommended |
| Consensus | 200M+ peer-reviewed papers, semantic search | MCP OAuth (one-time browser login) |
| arXiv | Physics, CS, math, quantitative bio | None |
| CORE | 300M+ open-access papers | API key (free) |
| PhilPapers | Philosophy | None |
| NCBI / PubMed | Biomedical literature | Email (courtesy) |
| Google Books | Book-length works | API key |
| Open Library | Book metadata | None |

---

## LLM routing

Primary: **Claude** (Haiku and Sonnet via Anthropic API).
Fallback: **Ollama** (local; qwen2.5 and similar).

The router selects per-agent based on reasoning depth needed. Elevated token ceilings: Theorist and Synthesizer 16,000, Vision 12,000, remaining agents default.

---

## Persistence

SQLite at `db/pipeline.db`, 11-table schema. Every source, implication, proposal, evaluation, synthesis field, direction, and artefact is traceable to its originating run and agent. This is what makes `--resume` possible and what makes contradictions detectable across agents.

---

## Installation

```bash
git clone https://github.com/anvix9/basis_research_agents
cd basis_research_agents
pip install -r requirements.txt
cp .env.example .env
# Edit .env — add required keys
python3 main.py keys      # verify configuration
python3 main.py run --problem "Your research question here"
```

### Required environment variables

| Variable | Required | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | For Claude Haiku/Sonnet |
| `OPENALEX_API_KEY` | Yes | Free; required since Feb 2026 |
| `NCBI_EMAIL` | Recommended | Courtesy identification |
| `CORE_API_KEY` | Optional | Enables CORE retrieval |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional | Higher rate limits |
| `GOOGLE_BOOKS_API_KEY` | Optional | Enables book-length retrieval |

**Consensus does not use an API key.** On first run you'll be prompted to log in once via browser; tokens persist automatically.

---

## Common commands

```bash
# Full run
python3 main.py run --problem "Your research question"

# Resume a crashed or cancelled run
python3 main.py run --problem "..." --run-id RUN-XXXXXXXX --resume

# List all runs with status
python3 main.py runs

# Verify all API keys and Consensus MCP auth status
python3 main.py keys

# Passive collection (suitable for cron)
python3 main.py collect

# Evaluation tools
python3 eval_references.py --run RUN-XXXXXXXX
python3 eval_claims.py --run RUN-XXXXXXXX

# Export seminal papers for blog
python3 export_seminal.py --list-runs
python3 export_seminal.py --run RUN-XXXXXXXX --jekyll
```

---

## Known limitations — v10.4

- **Token cost on Consensus.** Consensus returns verbose plain text requiring regex parsing; it is gated behind `agent_sources` config and off by default for most agents.
- **No automated cross-run synthesis.** Each run is self-contained; comparing insights across runs is manual.
- **Narrative contamination risk.** A "Noter" agent concept (deterministic briefing layer) has been validated as a real need but not yet implemented — implementing it as a reasoning agent risks narrative bleed between agents.
- **Single-language output.** Final artefacts inherit the language of the problem statement; no automated translation.

---

## Licence

MIT — see `LICENCE`.

Data source licences: OpenAlex (CC0), arXiv metadata (CC0), PubMed metadata (public domain), Open Library metadata (CC0 / CC BY), PhilArchive (CC BY-SA). Consensus, Semantic Scholar, CORE, and Google Books are subject to their respective terms of service.

Built with Claude (Anthropic) and Ollama.
