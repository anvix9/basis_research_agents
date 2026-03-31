# Multi-Agent Research Intelligence Pipeline

A locally-running multi-agent system for deep, interdisciplinary research. Give it a research question — it excavates the intellectual history, maps the gaps, proposes approaches, evaluates their feasibility, synthesises a research narrative, and produces a final document. You stay in control through three mandatory review breaks.

---

## What it does

Most research tools search. This one thinks.

The pipeline runs 10 specialised agents in sequence, each building on the last. It pulls from academic databases, book catalogs, and web search simultaneously. It distinguishes between what the field has established, what it has tried and abandoned, where the real gaps are, and what is worth proposing next. A human-in-the-loop design means you review and redirect at three points before the pipeline continues.

```
Social → [Break 0] → Grounder → Historian → Gaper
       → [Break 1] → Vision → Theorist → Rude → Synthesizer
       → [Break 2] → Thinker → Scribe
```

---

## Agents

| Agent | Role |
|---|---|
| **Social** | Collects current papers and books from 8 academic sources |
| **Grounder** | Decomposes the problem into sub-questions, excavates intellectual origins, finds seminal works and books |
| **Historian** | Builds a chronological map of how the field developed — including dead ends |
| **Gaper** | Identifies and classifies all meaningful gaps (empirical, conceptual, methodological, theoretical) |
| **Vision** | Draws logical implications from everything established so far |
| **Theorist** | Proposes concrete, scoped, falsifiable approaches anchored in the gaps and implications |
| **Rude** | Evaluates every proposal with empirical rigour — identifies the weakest link in each |
| **Synthesizer** | Produces the unified research narrative and sharpens the original problem |
| **Thinker** | Opens genuinely new research directions beyond existing proposals |
| **Scribe** | Writes the final document in the format you request (blog post, research brief, literature review, paper section, grant background) |

---

## Human-in-the-loop breaks

Three mandatory review breaks where the pipeline stops, produces a structured summary document, and waits for your instructions before continuing.

- **Break 0** — confirm which themes and sources to search
- **Break 1** — review foundations, timeline, and gaps; direct the proposal phase
- **Break 2** — review proposals and synthesis; specify the output format

---

## Sources

| Source | Type | Key needed |
|---|---|---|
| OpenAlex | Academic papers | Required (free) — [openalex.org/settings/api](https://openalex.org/settings/api) |
| arXiv | Preprints | None — uses official `arxiv` library |
| PubMed | Biomedical | Optional (free) — 3x rate boost with key |
| Semantic Scholar | Academic papers | Optional (free) |
| CORE | Open access aggregator | Optional (free) |
| PhilPapers | Philosophy index | Optional (free) — skipped gracefully without key |
| PhilArchive | Open access philosophy | None — OAI-PMH |
| PhilSci-Archive | Philosophy of science | None — OAI-PMH |
| **Google Books** | Books and monographs | Optional (free) — Grounder only |
| **Open Library** | Books and monographs | None — Grounder only |
| **Web search** | Broad coverage | Via Anthropic API — Grounder only |

---

## Semantic concept expansion

Before searching, the pipeline translates your research question into its full conceptual territory using a local ConceptNet database (184MB, 2.29 million English edges). A three-layer process — term extraction, ConceptNet neighbourhood, LLM synthesis — determines which of the 23 configured research themes to activate.

This means a question like *"What is the place of AI in human life?"* automatically activates philosophy of mind, anthropology, sociology, ethics, cognitive science, history, and law — without you having to specify them.

---

## LLM routing

The pipeline uses Claude as the primary model with automatic fallback:

1. Claude Sonnet 4.5 (primary — all heavy reasoning agents)
2. Claude Haiku 4.5 (lighter agents and fallback)
3. Ollama `deepseek-r1:8b` (local fallback if Claude API unavailable)
4. Ollama `llama3.2:3b` (final local fallback)

Per-agent token limits prevent truncation: Grounder and Theorist get 16,000 tokens; Historian and Synthesizer get 10,000.

---

## Output formats

Specified in your Break 2 instruction file:

```
SCRIBE OUTPUT: blog_post | audience: general public
SCRIBE OUTPUT: literature_review | audience: academic peers
SCRIBE OUTPUT: research_brief | audience: policy makers
SCRIBE OUTPUT: paper_section | audience: journal reviewers
SCRIBE OUTPUT: grant_background | audience: funding committee
SCRIBE OUTPUT: internal_memo | audience: research team
```

Prose formats (blog post, brief, memo) output `.md`. Academic formats (literature review, paper section, grant background) output `.tex`.

---

## Installation

```bash
git clone https://github.com/yourusername/pipeline
cd pipeline
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your keys:

```bash
python3 main.py keys    # see what is set and what is missing
```

---

## Required keys

| Key | Status | Where |
|---|---|---|
| `ANTHROPIC_API_KEY` | Required | [console.anthropic.com](https://console.anthropic.com) |
| `OPENALEX_API_KEY` | Required (free) | [openalex.org/settings/api](https://openalex.org/settings/api) |
| `NCBI_EMAIL` | Required | Your email — NCBI terms of service |
| `NCBI_API_KEY` | Optional (free) | [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/) |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional (free) | [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api) |
| `CORE_API_KEY` | Optional (free) | [core.ac.uk/services/api](https://core.ac.uk/services/api) |
| `PHILPAPERS_API_ID` + `API_KEY` | Optional (free) | [philpapers.org/utils/create_api_user.html](https://philpapers.org/utils/create_api_user.html) |
| `GOOGLE_BOOKS_API_KEY` | Optional (free) | Google Cloud Console → Books API |

---

## Run

```bash
python3 main.py run --problem "What is the place of AI in human life?"
```

Resume an interrupted run:

```bash
python3 main.py run --problem "..." --run-id RUN-20260330-203603-3283 --resume
```

Other commands:

```bash
python3 main.py collect          # passive scan of all themes (run twice weekly)
python3 main.py recheck          # check all saved links are still alive
python3 main.py status --run-id RUN-XXX
python3 main.py runs             # list recent runs
python3 main.py bank             # review Grounder's proposed new themes
python3 main.py keys             # check API key status
```

---

## ConceptNet local database

The concept expansion layer works without ConceptNet (falls back to LLM-only) but is significantly richer with it. One-time setup:

```bash
# Download the raw dump (~1.5GB compressed)
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz

# Dry run — test filters on first 2 million lines without writing
python3 tools/import_conceptnet.py --input conceptnet-assertions-5.7.0.csv.gz --dry-run

# Full import (~30-60 min depending on disk speed)
python3 tools/import_conceptnet.py --input conceptnet-assertions-5.7.0.csv.gz

# Verify
python3 tools/import_conceptnet.py --stats
```

Output: `db/conceptnet.db` — 184MB, 2.29M English edges, indexed for fast lookup.

---

## Passive collection (optional cron)

Run Social twice weekly to build up the source database before any specific problem is submitted:

```bash
# Add to crontab
0 6 * * 1,4 cd /path/to/pipeline && python3 main.py collect
```

---

## Database

All pipeline data is stored in `db/pipeline.db` (SQLite). 11 tables:

`runs` · `sources` · `gaps` · `implications` · `proposals` · `evaluations` · `syntheses` · `directions` · `artifacts` · `seminal_bank` · `dead_links`

Full schema documented in [TECHNICAL.md](./TECHNICAL.md).

---

## File structure

```
pipeline/
├── main.py                    # CLI entry point
├── config.json                # 23 themes, 14 sources — edit to extend
├── concept_map.json           # 27 disciplinary clusters, 500+ trigger concepts
├── .env.example               # Key instructions
├── requirements.txt
├── core/
│   ├── llm.py                 # LLM router — Claude primary, Ollama fallback
│   ├── database.py            # SQLite — all tables and CRUD
│   ├── context.py             # Context assembly per agent
│   ├── breaks.py              # Hard stop mechanics
│   ├── rate_limiter.py        # Per-source delays, backoff, progress display
│   ├── keys.py                # .env loader, typed key accessors
│   ├── concept_mapper.py      # 3-layer semantic expansion
│   └── utils.py               # Logging, ID generation, config loading
├── agents/
│   ├── social.py              # Multi-source collector
│   ├── grounder.py            # Intellectual origins
│   ├── historian.py           # Chronological map
│   ├── gaper.py               # Gap identification
│   ├── vision.py              # Logical implications
│   ├── theorist.py            # Proposals (two-pass)
│   ├── rude.py                # Feasibility evaluation
│   ├── synthesizer.py         # Research narrative
│   ├── thinker.py             # New directions
│   └── scribe.py              # Final artifact output
├── tools/
│   └── import_conceptnet.py   # One-time ConceptNet CSV → SQLite
├── db/
│   ├── pipeline.db            # Main database (created on first run)
│   └── conceptnet.db          # ConceptNet local graph (import separately)
└── artifacts/                 # All agent outputs + final documents
```

---

## Requirements

- Python 3.12+
- `pip install -r requirements.txt` — `anthropic`, `requests`, `arxiv`
- Anthropic API key with credits
- OpenAlex API key (free, required since February 2026)
- Optional: Ollama running locally for API-free fallback

---

## Licence

MIT — see [LICENCE](./LICENCE).

Data sources: OpenAlex (CC0), ConceptNet (CC BY-SA 4.0), arXiv (metadata CC0), PubMed (public domain metadata), Open Library (CC0), PhilArchive (CC BY-SA).
