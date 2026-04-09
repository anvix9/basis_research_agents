"""
API Key Manager
---------------
Loads API keys from .env file and environment variables.
Provides keys to all source handlers.
Warns clearly when a required key is missing.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_PATH = Path(__file__).parent.parent / ".env"


def _load_env():
    """Load .env file into os.environ if not already set."""
    if not ENV_PATH.exists():
        logger.debug(f"[Keys] No .env file found at {ENV_PATH} — using environment only")
        return
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key   = key.strip()
                value = value.strip()
                # Only set if not already in environment
                if key and value and key not in os.environ:
                    os.environ[key] = value


# Load on import
_load_env()


def get(key: str, required: bool = False, source_name: str = "") -> str:
    """
    Get an API key by environment variable name.
    If required=True and key is missing, logs a clear warning.
    Returns empty string if not set.
    """
    value = os.environ.get(key, "").strip()
    if not value and required:
        logger.warning(
            f"[Keys] {key} is not set — {source_name} will not work correctly.\n"
            f"  Add it to your .env file or environment.\n"
            f"  See .env.example for instructions."
        )
    return value


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def openalex() -> str:
    return get("OPENALEX_API_KEY", required=True, source_name="OpenAlex (required since Feb 2026)")

def ncbi_api_key() -> str:
    return get("NCBI_API_KEY", required=False, source_name="PubMed/NCBI (optional — 3x rate limit boost)")

def ncbi_email() -> str:
    return get("NCBI_EMAIL", required=True, source_name="PubMed/NCBI (email required by ToS)")

def semantic_scholar() -> str:
    return get("SEMANTIC_SCHOLAR_API_KEY", required=False, source_name="Semantic Scholar (optional)")

def core() -> str:
    return get("CORE_API_KEY", required=False, source_name="CORE (optional — higher rate limits)")

def philpapers_id() -> str:
    return get("PHILPAPERS_API_ID", required=False, source_name="PhilPapers (optional — OAI-PMH used as fallback)")

def philpapers_key() -> str:
    return get("PHILPAPERS_API_KEY", required=False, source_name="PhilPapers (optional — OAI-PMH used as fallback)")

def anthropic() -> str:
    return get("ANTHROPIC_API_KEY", required=True, source_name="Anthropic Claude API")

def google_books() -> str:
    return get("GOOGLE_BOOKS_API_KEY", required=False, source_name="Google Books (optional — higher quota)")

def scopus_api_key() -> str:
    return get("SCOPUS_API_KEY", required=False, source_name="Scopus (optional — needs institutional IP/VPN)")

def scopus_inst_token() -> str:
    return get("SCOPUS_INST_TOKEN", required=False, source_name="Scopus institutional token (optional — email datasupport@elsevier.com)")

def consensus_mcp_status() -> str:
    """Consensus uses MCP OAuth — check db/consensus_tokens.json for token status."""
    from pathlib import Path
    token_file = Path(__file__).parent.parent / "db" / "consensus_tokens.json"
    return "authenticated" if token_file.exists() else "not authenticated (run pipeline once to trigger OAuth)"


def print_key_status():
    """Print a clear table of which keys are set and which are missing."""
    checks = [
        ("OPENALEX_API_KEY",        "OpenAlex",         True,  "openalex.org/settings/api"),
        ("NCBI_API_KEY",            "PubMed (NCBI)",    False, "ncbi.nlm.nih.gov/account"),
        ("NCBI_EMAIL",              "PubMed email",     True,  "any valid email"),
        ("SEMANTIC_SCHOLAR_API_KEY","Semantic Scholar", False, "semanticscholar.org/product/api"),
        ("CORE_API_KEY",            "CORE",             False, "core.ac.uk/services/api"),
        ("PHILPAPERS_API_ID",       "PhilPapers ID",    False, "philpapers.org/utils/create_api_user.html"),
        ("PHILPAPERS_API_KEY",      "PhilPapers Key",   False, "philpapers.org/utils/create_api_user.html"),
        ("SCOPUS_API_KEY",          "Scopus",           False, "dev.elsevier.com → Create API Key"),
        ("SCOPUS_INST_TOKEN",        "Scopus Inst Token",False, "email datasupport@elsevier.com"),
        ("GOOGLE_BOOKS_API_KEY",    "Google Books",     False, "console.cloud.google.com → Books API"),
        ("ANTHROPIC_API_KEY",       "Anthropic Claude", True,  "console.anthropic.com"),
    ]
    print(f"\n  {'─'*60}")
    print(f"  API Key Status")
    print(f"  {'─'*60}")
    all_required_ok = True
    for env_var, name, required, url in checks:
        value = os.environ.get(env_var, "").strip()
        if value:
            masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
            status = f"✅ set ({masked})"
        elif required:
            status = f"❌ MISSING (required) → {url}"
            all_required_ok = False
        else:
            status = f"⚠️  not set (optional) → {url}"
        req_str = "[required]" if required else "[optional]"
        print(f"  {name:<22} {req_str:<12} {status}")
    print(f"  {'─'*60}")
    if not all_required_ok:
        print(f"  ⚠️  Some required keys are missing. Edit your .env file.")
        print(f"     Copy .env.example → .env and fill in the values.")
    else:
        print(f"  ✅ All required keys are set.")
    print(f"  {'─'*60}\n")
