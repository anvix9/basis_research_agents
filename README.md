# SEEKER — Multi-Agent Deep Research Intelligence Pipeline

**SEEKER** is a 10-agent research pipeline that conducts deep, traceable academic research by orchestrating specialized agents across live data sources. It builds a persistent **argument tree** where every claim traces to verifiable evidence — papers, books, legal documents, news articles, court decisions, testimony, or archival records.

The system is designed to preserve the researcher's intellectual contribution. It does not write your paper. It maps the intellectual territory so you can navigate it yourself.

## How It Works

```
Concept Mapper → Break 0 (you confirm themes)
  → Grounder (decomposes problem → searches → builds argument tree)
  → Social (contemporary + bridge papers → extends tree)
  → Historian (audits tree → historical search → external factors)
  → Gaper (structural + analytical gap mapping from tree)
  → Break 1 (you review foundations and gaps)
  → Vision (logical implications)
  → Theorist (research proposals)
  → Rude (feasibility evaluation — adversarial)
  → Synthesizer (research narrative)
  → Break 2 (you set trajectory)
  → Thinker (new directions)
  → Scribe (understanding map + outputs)
```

**Three human-in-the-loop breaks** where you read, learn, and decide the trajectory. The pipeline stops and waits for your input.

**The Argument Tree** is the backbone. Every agent reads and extends it. Claims without evidence are flagged. Temporal gaps are detected and bridge papers are searched automatically. The Historian audits the tree for solidity before downstream agents use it.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/basis_research_agents.git
cd basis_research_agents
pip install -r requirements.txt
```

### 2. Set up the ConceptNet database

The ConceptNet database provides semantic concept expansion. It ships compressed:

```bash
cd db/
gunzip conceptnet.db.gz
cd ..
```

This produces `db/conceptnet.db` (~184 MB). If you skip this step, the concept mapper will use a simpler keyword-only mode.

### 3. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

**That's the only required key.** All academic sources (OpenAlex, Semantic Scholar, arXiv, CORE) are free and keyless.

### 4. Run your first pipeline

```bash
python3 main.py run --problem "How does identity influence intelligence through the lens of identity theory?"
```

The pipeline will:
1. Expand your problem into conceptual territory
2. Stop at **Break 0** — you confirm which academic themes to search
3. Search 10+ academic sources, build the argument tree
4. Stop at **Break 1** — you review the foundations and gaps
5. Generate implications, proposals, evaluations, synthesis
6. Stop at **Break 2** — you set the trajectory and choose outputs
7. Produce an Understanding Map, research brief, and whatever else you requested

### 5. Resume a run

If the pipeline is interrupted or you want to re-run from a break point:

```bash
python3 main.py run --problem "your problem" --run-id RUN-20260407-022355-242D --resume
```

The pipeline detects which agents already completed (by checking the database) and skips them.

## Using Ollama (Free, Local, No API Key)

SEEKER works with local models via [Ollama](https://ollama.com/) as a fallback. No cloud API needed.

### Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (14B+ recommended for research quality)
ollama pull qwen2.5:14b

# Or a smaller model for testing
ollama pull qwen2.5:7b
```

### Configuration

Leave `ANTHROPIC_API_KEY` empty in your `.env` file. The pipeline auto-detects Ollama on `localhost:11434` and uses it for all agents.

```
# .env — leave blank for Ollama-only mode
ANTHROPIC_API_KEY=
```

### What to expect

Local models (7B–14B) produce usable results but with lower quality than Claude:
- Sub-question decomposition may be shallower
- JSON parsing failures are more frequent (the pipeline has fallback parsers)
- Synthesis quality depends heavily on model size
- 14B+ is recommended; 7B works but produces thin outputs

The pipeline is designed to degrade gracefully — every JSON parser has a fallback, every agent has error handling, and the argument tree preserves whatever was successfully parsed.

## Configuration

### `config.json`

Controls which academic themes are available, their search keywords, and which sources each agent uses:

```json
{
  "agent_sources": {
    "social":   ["openalex", "arxiv", "pubmed", "semantic_scholar", "core"],
    "grounder": ["openalex", "semantic_scholar"]
  }
}
```

Add or remove sources per agent without touching code.

### `concept_map.json`

37 semantic concept clusters that translate natural language research problems into academic themes. If SEEKER doesn't recognize your field, add a cluster here (see CONTRIBUTING.md).

## Evaluation Tools

```bash
# Check if references are real (Semantic Scholar + OpenAlex + Claude web search)
python3 tools/eval_references.py --run-id RUN-... --types seminal --limit 10

# Check if claims about references are accurate
python3 tools/eval_claims.py --run-id RUN-... --types seminal --limit 10

# Generate a formatted reference section (APA/Chicago)
python3 tools/generate_references.py --run-id RUN-... --format apa
```

## Academic Sources

SEEKER searches 10+ live academic sources. No source uses model training data.

| Source | Key Required | Coverage |
|--------|-------------|----------|
| OpenAlex | No (email for faster access) | 250M+ works, full metadata |
| Semantic Scholar | No (key for higher limits) | 200M+ papers, citation graphs |
| arXiv | No | Preprints (CS, physics, math, etc.) |
| PubMed/NCBI | No | Biomedical literature |
| CORE | No | 200M+ open access papers |
| PhilPapers | No | Philosophy-specific |
| PhilArchive | No | Open philosophy preprints |
| PhilSci Archive | No | Philosophy of science |
| Scopus | API key (institutional) | Elsevier's comprehensive index |
| Consensus | MCP OAuth (Pro plan) | Semantic search, 200M+ papers |

## Project Structure

```
basis_research_agents/
├── main.py              # Pipeline runner and CLI
├── config.json          # Themes, sources, agent configuration
├── concept_map.json     # 37 semantic concept clusters
│
├── agents/              # The 10 pipeline agents
│   ├── grounder.py      # Problem decomposition + argument tree building
│   ├── social.py        # Contemporary + bridge paper discovery
│   ├── historian.py     # Tree audit + historical context + external factors
│   ├── gaper.py         # Tree-native structural + analytical gap mapping
│   ├── vision.py        # Logical implications extraction
│   ├── theorist.py      # Research proposal generation (two-pass)
│   ├── rude.py          # Adversarial feasibility evaluation
│   ├── synthesizer.py   # Research narrative synthesis
│   ├── thinker.py       # New direction exploration
│   └── scribe.py        # Understanding Map + output generation
│
├── core/                # Infrastructure
│   ├── argument_tree.py # Persistent argument tree (TreeBuilder class)
│   ├── llm.py           # LLM router (Claude primary, Ollama fallback)
│   ├── database.py      # SQLite schema (12 tables)
│   ├── context.py       # Context assembly with tree injection
│   ├── concept_mapper.py# Problem → theme activation (130 disciplines)
│   ├── breaks.py        # Human-in-the-loop break points
│   ├── references.py    # Reference formatting for Scribe
│   ├── consensus_mcp.py # Consensus OAuth 2.1 MCP client
│   ├── rate_limiter.py  # Per-source API rate limiting
│   ├── keys.py          # Environment variable management
│   └── utils.py         # Shared utilities
│
├── tools/               # Evaluation and export tools
│   ├── eval_references.py    # Reference existence verification
│   ├── eval_claims.py        # Claim accuracy verification
│   ├── generate_references.py# APA/Chicago reference section generator
│   ├── export_seminal.py     # Seminal works → Jekyll blog export
│   └── import_conceptnet.py  # Build ConceptNet DB from CSV dump
│
├── tests/               # Unit tests
│   ├── test_grounder_tree.py # Argument tree building
│   ├── test_social_bridge.py # Bridge gap detection
│   ├── test_historian_tree.py# Tree audit + extension
│   ├── test_context_tree.py  # Tree context injection
│   ├── test_gaper_tree.py    # Tree-native gap mapping
│   └── test_references.py    # Reference section generation
│
├── db/                  # Databases
│   ├── conceptnet.db.gz # ConceptNet (compressed — gunzip before first run)
│   ├── pipeline.db      # Created automatically on first run
│   └── consensus_tokens.json  # OAuth tokens (auto-managed)
│
├── artifacts/           # Pipeline outputs (per run)
├── logs/                # Run logs
├── .env.example         # API key template
├── CONTRIBUTING.md      # Contribution guide
├── LICENSE              # MIT
└── requirements.txt     # Python dependencies
```

## The Argument Tree

The argument tree is the single source of truth for the entire pipeline. Every agent reads and extends it.

```
ROOT: "How does identity influence intelligence?"
├── Q1: What is identity?
│   ├── CLAIM [solid] (90%): Identity is socially constructed
│   │   ├── EVIDENCE [book]: Mead 1934 — Mind, Self, and Society
│   │   ├── EVIDENCE [paper]: Tajfel 1979 — Social identity
│   │   └── COUNTER: Essentialists argue biological basis
│   └── BRIDGE [temporal]: Stryker 1980 connects 1934→1995
├── Q2: What is intelligence?
│   └── CLAIM [weak] (40%): Intelligence is g-factor — single evidence
├── Q3: How are they related? [UNANSWERED — structural gap]
├── HISTORICAL: Locke 1690 — personal identity tied to memory
├── EXTERNAL [war]: WWII disrupted European psychology departments
└── AUDIT: Tree health — 2 solid, 1 weak, 1 unanswered
```

Node types: `root`, `question`, `claim`, `evidence`, `bridge`, `counter`, `historical`, `external`, `audit_note`

Evidence types: `paper`, `book`, `report`, `legal_document`, `court_decision`, `news_article`, `archival`, `testimony`, `resolution`, `dataset`

## Tests

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run individually
python3 tests/test_grounder_tree.py
python3 tests/test_social_bridge.py
python3 tests/test_historian_tree.py
python3 tests/test_context_tree.py
python3 tests/test_gaper_tree.py
python3 tests/test_references.py
```

## CLI Reference

```bash
# Full pipeline run
python3 main.py run --problem "Your research question"

# Resume from where you left off
python3 main.py run --problem "..." --run-id RUN-XXXXXXXX --resume

# Test a single source handler
python3 main.py test --source openalex --query "neural networks"

# Passive collection (Social only, no pipeline)
python3 main.py collect

# Check run status
python3 main.py status --run-id RUN-XXXXXXXX

# View seminal bank proposals
python3 main.py bank
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Thanks to Academic data from OpenAlex, Semantic Scholar, arXiv, PubMed, CORE, PhilPapers, and Consensus, and Claude (Anthropic), Ollama local model fallback. 
