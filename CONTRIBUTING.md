# Contributing to SEEKER

Thank you for your interest in contributing. SEEKER is an open research tool and welcomes contributions from researchers, developers, and anyone interested in improving academic research workflows.

## Getting Started

1. **Fork and clone** the repository
2. **Set up the environment** (see README.md)
3. **Run the tests** to make sure everything works:
   ```bash
   python3 -m pytest tests/ -v
   ```

## Project Structure

```
basis_research_agents/
├── main.py              # Pipeline runner and CLI
├── config.json           # Themes, sources, agent_sources configuration
├── concept_map.json      # Semantic concept clusters (37 clusters)
├── agents/               # The 10 pipeline agents
│   ├── grounder.py       # Decomposes problem, builds argument tree
│   ├── social.py         # Contemporary + bridge papers, extends tree
│   ├── historian.py      # Audits tree, historical context, external factors
│   ├── gaper.py          # Tree-native gap mapping
│   ├── vision.py         # Logical implications
│   ├── theorist.py       # Research proposals
│   ├── rude.py           # Feasibility evaluation
│   ├── synthesizer.py    # Research narrative
│   ├── thinker.py        # New directions
│   └── scribe.py         # Output generation
├── core/                 # Infrastructure
│   ├── argument_tree.py  # Persistent argument tree (TreeBuilder)
│   ├── context.py        # Context assembly per agent
│   ├── database.py       # SQLite schema and operations
│   ├── llm.py            # LLM router (Claude + Ollama)
│   ├── concept_mapper.py # Problem → theme activation
│   ├── breaks.py         # Human-in-the-loop break points
│   ├── rate_limiter.py   # API rate limiting
│   ├── keys.py           # Environment variable management
│   ├── references.py     # Reference formatting for Scribe
│   ├── consensus_mcp.py  # Consensus MCP OAuth client
│   └── utils.py          # Shared utilities
├── tools/                # Standalone evaluation and export tools
├── tests/                # Unit tests
├── db/                   # Databases (created on first run)
├── artifacts/            # Pipeline outputs (per run)
└── docs/                 # Documentation
```

## What Can You Contribute?

### New Source Handlers

The easiest way to contribute. Each source handler in `agents/social.py` follows a simple pattern:

```python
class YourSourceHandler(SourceHandler):
    SOURCE_ID = "your_source"
    
    def search(self, query, keywords, limit=10, run_id=""):
        # Call your API
        # Return list of dicts with: title, authors, year, abstract, doi, etc.
        return results
```

Add your handler to `SOURCE_HANDLERS` dict and add the source to `config.json` themes.

### New Concept Map Clusters

If SEEKER doesn't recognize your research domain, add a cluster to `concept_map.json`:

```json
{
  "cluster_id": "your_domain",
  "label": "Your Research Domain",
  "trigger_concepts": ["term1", "term2", "term3"],
  "disciplines": ["political_science", "sociology"],
  "bridge_concepts": ["connecting_idea_1"]
}
```

Then add the discipline → theme_id mapping in `core/concept_mapper.py`'s `EXPLICIT_MAP`.

### New Themes

Add themes to `config.json` with keywords and source lists:

```json
{
  "theme_id": "your_theme",
  "label": "Your Theme Label",
  "keywords": [{"seed": "keyword1"}, {"seed": "keyword2"}],
  "sources": ["openalex", "semantic_scholar"]
}
```

### Bug Fixes and Improvements

- Run the test suite before and after your changes
- Keep the argument tree as the single source of truth
- Every claim must be traceable to an evidence node with a real source

## Code Style

- Python 3.11+
- No strict formatter enforced — match the existing style
- Docstrings on all public functions
- Type hints encouraged but not required
- JSON output from agents must be parseable — always include fallback parsing

## Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run a specific test
python3 tests/test_grounder_tree.py

# Test a source handler
python3 main.py test --source openalex --query "neural networks"
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Run the test suite
4. Submit a PR with a clear description of what you changed and why

## Questions?

Open an issue. We're happy to help.
