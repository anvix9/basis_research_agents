# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

**One-time setup**: the ConceptNet DB ships compressed and must be unpacked before the first run.

```bash
cd db && gunzip conceptnet.db.gz && cd ..
cp .env.example .env   # then add ANTHROPIC_API_KEY (or leave blank for Ollama-only)
pip install -r requirements.txt
```

**Pipeline**:

```bash
python3 main.py run --problem "Your research question"
python3 main.py run --problem "..." --run-id RUN-YYYYMMDD-XXXX --resume
```

**Other CLI subcommands** (all in [main.py](main.py)):

| Command | Purpose |
|---|---|
| `run` | Full pipeline (10 agents + 3 breaks) |
| `collect` | Social passive theme scan, no run_id |
| `recheck` | Link health check across `sources` table |
| `status --run-id RUN-...` | Show run progress + per-table counts |
| `bank` | List Grounder's pending seminal-bank theme proposals |
| `runs` | List recent runs |
| `keys` | Show API key status |
| `test --source <name> --query "..."` | Debug a single source handler |

**Tests** — both patterns work; tests are pytest-compatible AND runnable as standalone scripts:

```bash
python3 -m pytest tests/ -v
python3 tests/test_grounder_tree.py     # standalone-script style also works
```

**Evaluation tools** (in [tools/](tools/)): `eval_references.py`, `eval_claims.py`, `generate_references.py`, `export_seminal.py`, `import_conceptnet.py`.

## Architecture

### Pipeline orchestration

[main.py](main.py) sequences 10 agents around 3 human breaks (Concept Mapper → Break 0 → Grounder → Social → Historian → Gaper → Break 1 → Vision → Theorist → Rude → Synthesizer → Break 2 → Thinker → Scribe). Each agent exposes `run(context, run_id)` and writes its results to its own DB table. **There is no direct data passing between agents** — downstream agents read what upstream agents wrote by querying the DB through [core/context.py](core/context.py) builders.

### The argument tree is the spine

[core/argument_tree.py](core/argument_tree.py) defines `TreeBuilder`, a SQLite-backed tree with 9 node types: `root`, `question`, `claim`, `evidence`, `bridge`, `counter`, `historical`, `external`, `audit_note`. Grounder **creates** the tree; every agent after Grounder may extend it (claims, counters, historical context, audit notes) and reads from it via the `for_<agent>()` context builders in [core/context.py](core/context.py).

Evidence-specific fields (evidence type, relationship to the claim) live in a JSON `metadata` column, **not** as schema columns. When adding evidence, pass `evidence_type` and `relationship` as parameters; they're serialized into metadata. Read them back through `to_context()` formatting rather than querying metadata directly.

### `--resume` is table-presence-based, not step-flag-based

[`_agent_done()`](main.py:389) infers completion by checking whether the agent's output table has rows for that `run_id`:

```
grounder    → seminal sources
social      → current sources
historian   → historical sources
gaper       → gaps
vision      → implications
theorist    → proposals
rude        → evaluations
synthesizer → synthesis row
thinker     → directions
scribe      → artifacts
```

**Implication**: if an agent partially fails (e.g., wrote 3 of 10 expected rows), `--resume` will **skip** it. To retry, manually `DELETE FROM <table> WHERE run_id = ...` first.

Only the three breaks are tracked explicitly: `break0_done`, `break1_done`, `break2_done` in the `runs` table.

### Two `breaks.py` files — don't conflate them

- [core/breaks.py](core/breaks.py) — the **break flow**. Writes `artifacts/{run_id}_break{N}_review.md`, blocks on `input()`, parses instructions out of the markdown (under the `**Your instructions:**` marker), calls `db.mark_break_done()`, returns instructions string for the next agent's context.
- [agents/breaks.py](agents/breaks.py) — **agent-side helpers** for interpreting break instructions inside agents.

### LLM router ([core/llm.py](core/llm.py)) — four backends

Four backends are supported. Three of them (`anthropic`, `deepseek`, `omlx`) share the same code path: an `Anthropic()` SDK client constructed with backend-specific `api_key=` and `base_url=`. Only `ollama` uses its own native HTTP transport.

| Backend | Heavy / Light models | API key | Base URL |
|---|---|---|---|
| `anthropic` | `claude-sonnet-4-5` / `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` | SDK default |
| `deepseek` | `deepseek-v4-pro` / `deepseek-v4-flash` | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/anthropic` |
| `omlx` | `Qwen3.6-35B-A3B-4bit` / `gemma-4-e4b-it-8bit` | `OMLX_API_KEY` | `http://127.0.0.1:8000` |
| `ollama` | `deepseek-r1:8b` / `llama3.2:3b` | none | `http://localhost:11434` |

**Important**: even though DeepSeek's docs say the `x-api-key` header is read from `ANTHROPIC_API_KEY`, the router pulls the key from `DEEPSEEK_API_KEY` and passes it to the SDK explicitly. Never expose `$ANTHROPIC_API_KEY` to non-Anthropic backends.

**Backend selection (highest precedence first)**:
1. `--backend` CLI flag on `main.py run`
2. `LLM_BACKEND` env var
3. `config.json` → `llm.backend`
4. Legacy auto-detect: `anthropic` if `ANTHROPIC_API_KEY` is set, else `ollama`

**Per-agent tier**: agents are tagged `heavy` or `light` in `AGENT_TIER`. Heavy agents call the heavy model and within-backend-fall-back to the light model on failure. Light agents go straight to the light model with no within-backend retry (it's already the cheap model). Per-agent token caps remain in `AGENT_MAX_TOKENS`.

**Cross-backend fallback is opt-in**. Set `config.llm.fallback_backend` or `LLM_FALLBACK_BACKEND` to chain backends; default is `null`, meaning failure on the chosen backend raises rather than silently routing elsewhere. This is a deliberate behavior change from the older "always fall back to Ollama" path.

The router is a lazily-initialized module-level singleton; tests can call `llm.reset_client()` after mutating env to rebuild it.

### Database ([core/database.py](core/database.py)) — no migration system

12 tables, schema bootstrapped once via `executescript(SCHEMA)` in `init_db()`. Adding or changing a column requires editing the `SCHEMA` constant **and** manually `ALTER TABLE`-ing any existing `db/pipeline.db`. There is no Alembic, no versioned migrations, no auto-upgrade.

Many fields are JSON strings inside `TEXT` columns (`authors`, `theme_tags`, `addresses_gaps`, `assumptions`, etc.). Use the `_json()` / `_from_json()` helpers. `upsert_source()` uses `INSERT OR REPLACE` for idempotency.

### Concept Mapper + Break 0

[core/concept_mapper.py](core/concept_mapper.py) reads `concept_map.json` (37 semantic clusters) and activates themes from `config.json`. The activation is shown to the user at Break 0 for confirmation/editing **before any source search runs** — this is the only chance to scope the literature search.

### Prompts are hardcoded in agent files

Each agent in [agents/](agents/) defines its system prompt and user-prompt template as Python string literals. There is no separate prompts directory and no prompt versioning — editing a prompt is editing code. Output JSON parsing in every agent must include a fallback parser; LLM outputs (especially under Ollama) are best-effort.

## Configuration & Environment

- `config.json` — **theme registry only**: themes, their seed keywords, and the `agent_sources` map per agent. Not feature flags. Adding a theme = adding an entry here plus, if needed, a discipline mapping in `core/concept_mapper.py`'s `EXPLICIT_MAP`.
- `concept_map.json` — 37 semantic clusters that translate problem statements into theme activations.
- Required env: `ANTHROPIC_API_KEY` (or leave blank to run Ollama-only at `localhost:11434`).
- Optional env: `OPENALEX_EMAIL`, `S2_API_KEY`, `SCOPUS_API_KEY` + `SCOPUS_INST_TOKEN`, `CORE_API_KEY`, `NCBI_API_KEY` + `NCBI_EMAIL`, `PHILPAPERS_API_ID` + `PHILPAPERS_API_KEY`, `OLLAMA_HOST`. See [.env.example](.env.example).
- [main.py](main.py) calls `_load_env()` at the very top, **before** other imports. Any module that reads `os.environ` at import time must be imported after that call.

## Conventions & Gotchas

- **New source handler**: subclass `SourceHandler` in [agents/social.py](agents/social.py), set `SOURCE_ID`, implement `search(query, keywords, limit, run_id)` returning dicts with `title`, `authors`, `year`, `abstract`, `doi`, etc., register in the `SOURCE_HANDLERS` dict, then add the source name to the relevant theme's `sources` list (or to `agent_sources`) in `config.json`.
- **Tree integrity**: every claim should have at least one evidence node. Orphan claims surface as "weak" or "unsupported" in Historian's audit pass.
- `run_id` format is `RUN-YYYYMMDD-XXXX` and acts as the foreign key across all tables — never reuse one across distinct problems.
- **Test isolation pattern** (no conftest, no fixtures): tests patch `at.DB_PATH` and `database.DB_PATH` to a `/tmp/test_*.db` path before importing the modules under test, then build minimal schema via `conn.executescript()`. Duplicate this pattern when adding tests.
- The `pyproject.toml` script entry is named `aranea` — that's the legacy project name; the repo and code use `SEEKER`.

## Critical Files

- [main.py](main.py) — pipeline orchestration, CLI, `_agent_done()` resume logic
- [core/argument_tree.py](core/argument_tree.py) — `TreeBuilder`
- [core/database.py](core/database.py) — `SCHEMA` and DB helpers
- [core/context.py](core/context.py) — per-agent context builders, tree injection
- [core/breaks.py](core/breaks.py) — human-in-the-loop flow
- [core/llm.py](core/llm.py) — Claude/Ollama routing, per-agent model selection
- [core/concept_mapper.py](core/concept_mapper.py) — feeds Break 0
- [config.json](config.json), [concept_map.json](concept_map.json) — runtime config
- [.env.example](.env.example) — full env-var reference
