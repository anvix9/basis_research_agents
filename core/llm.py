"""
LLM Router
----------
Primary:  Claude API (Anthropic)
Fallback: Ollama (local models)

Every agent calls llm.call(prompt, system, agent_name)
The router handles retries, fallback, and logging transparently.
"""

import os
import json
import time
import logging
import requests
from typing import Optional
from anthropic import Anthropic, APIStatusError, APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Claude models — primary
CLAUDE_PRIMARY_MODEL   = "claude-sonnet-4-5"
CLAUDE_FALLBACK_MODEL  = "claude-haiku-4-5-20251001"  # cheaper fallback within API

# Ollama — local fallback
OLLAMA_BASE_URL        = "http://localhost:11434"
OLLAMA_PRIMARY_MODEL   = "deepseek-r1:8b"
OLLAMA_LIGHT_MODEL     = "llama3.2:3b"

# Retry settings
MAX_API_RETRIES        = 3
RETRY_DELAY_SECONDS    = 5
MAX_TOKENS             = 8192

# Per-agent token limits — agents with large structured JSON outputs need more
AGENT_MAX_TOKENS = {
    "grounder":    16000,  # decomposition + multi-source search + large JSON
    "theorist":    16000,  # multiple proposals each with full fields
    "historian":   10000,  # chronological map can be long
    "synthesizer": 10000,  # narrative + full bibliography
    "gaper":        8192,
    "vision":       8192,
    "rude":         8192,
    "thinker":      8192,
    "social":       4096,
    "scribe":       8192,
}


# ---------------------------------------------------------------------------
# Agent → model mapping
# Heavier reasoning agents get the primary model
# Lighter formatting agents get the cheaper/faster model
# ---------------------------------------------------------------------------

AGENT_MODEL_MAP = {
    # Heavy reasoning — always use primary Claude
    "grounder":    {"claude": CLAUDE_PRIMARY_MODEL, "ollama": OLLAMA_PRIMARY_MODEL},
    "historian":   {"claude": CLAUDE_PRIMARY_MODEL, "ollama": OLLAMA_PRIMARY_MODEL},
    "gaper":       {"claude": CLAUDE_PRIMARY_MODEL, "ollama": OLLAMA_PRIMARY_MODEL},
    "vision":      {"claude": CLAUDE_PRIMARY_MODEL, "ollama": OLLAMA_PRIMARY_MODEL},
    "theorist":    {"claude": CLAUDE_PRIMARY_MODEL, "ollama": OLLAMA_PRIMARY_MODEL},
    "rude":        {"claude": CLAUDE_PRIMARY_MODEL, "ollama": OLLAMA_PRIMARY_MODEL},
    "synthesizer": {"claude": CLAUDE_PRIMARY_MODEL, "ollama": OLLAMA_PRIMARY_MODEL},
    "thinker":     {"claude": CLAUDE_PRIMARY_MODEL, "ollama": OLLAMA_PRIMARY_MODEL},
    # Lighter tasks — can use cheaper/faster model
    "social":      {"claude": CLAUDE_FALLBACK_MODEL, "ollama": OLLAMA_LIGHT_MODEL},
    "scribe":      {"claude": CLAUDE_FALLBACK_MODEL, "ollama": OLLAMA_LIGHT_MODEL},
}

DEFAULT_MODEL_ENTRY = {"claude": CLAUDE_PRIMARY_MODEL, "ollama": OLLAMA_PRIMARY_MODEL}


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self):
        self.anthropic_client = None
        self._load_env_first()
        self._init_anthropic()

    def _load_env_first(self):
        """Load .env before initializing clients — keys.py may not have run yet."""
        from pathlib import Path
        env_path = Path(__file__).parent.parent / ".env"
        if not env_path.exists():
            return
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key   = key.strip()
                    value = value.strip()
                    if key and value and key not in os.environ:
                        os.environ[key] = value

    def _init_anthropic(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set — will fall back to Ollama immediately")
            return
        try:
            self.anthropic_client = Anthropic(api_key=api_key)
            logger.info("Anthropic client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic client: {e}")
            self.anthropic_client = None

    def _ensure_anthropic(self):
        """Re-check key in case it was set after initial import — lazy init."""
        if self.anthropic_client is None:
            self._load_env_first()
            self._init_anthropic()

    def _get_models(self, agent_name: str) -> dict:
        return AGENT_MODEL_MAP.get(agent_name.lower(), DEFAULT_MODEL_ENTRY)

    # -----------------------------------------------------------------------
    # Claude API call
    # -----------------------------------------------------------------------

    def _call_claude(
        self,
        prompt: str,
        system: str,
        model: str,
        agent_name: str
    ) -> Optional[str]:
        if not self.anthropic_client:
            return None

        for attempt in range(1, MAX_API_RETRIES + 1):
            try:
                logger.info(f"[{agent_name}] Claude API call — model: {model} — attempt {attempt}/{MAX_API_RETRIES}")
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=AGENT_MAX_TOKENS.get(agent_name.lower(), MAX_TOKENS),
                    system=system,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text
                logger.info(f"[{agent_name}] Claude API success — {len(text)} chars returned")
                return text

            except RateLimitError as e:
                logger.warning(f"[{agent_name}] Rate limited — attempt {attempt}: {e}")
                if attempt < MAX_API_RETRIES:
                    wait = RETRY_DELAY_SECONDS * attempt
                    logger.info(f"[{agent_name}] Waiting {wait}s before retry...")
                    time.sleep(wait)

            except APIConnectionError as e:
                logger.warning(f"[{agent_name}] Connection error — attempt {attempt}: {e}")
                if attempt < MAX_API_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)

            except APIStatusError as e:
                logger.error(f"[{agent_name}] API status error {e.status_code}: {e.message}")
                # 5xx errors are retryable, 4xx are not
                if e.status_code >= 500 and attempt < MAX_API_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    return None

            except Exception as e:
                logger.error(f"[{agent_name}] Unexpected Claude error: {e}")
                return None

        logger.error(f"[{agent_name}] Claude API exhausted all retries")
        return None

    # -----------------------------------------------------------------------
    # Ollama local call
    # -----------------------------------------------------------------------

    def _call_ollama(
        self,
        prompt: str,
        system: str,
        model: str,
        agent_name: str
    ) -> Optional[str]:
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt}
            ],
            "stream": False,
            "options": {
                "num_predict": MAX_TOKENS,
                "temperature": 0.7
            }
        }

        try:
            logger.info(f"[{agent_name}] Ollama call — model: {model}")
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            text = data["message"]["content"]
            logger.info(f"[{agent_name}] Ollama success — {len(text)} chars returned")
            return text

        except requests.exceptions.ConnectionError:
            logger.error(f"[{agent_name}] Ollama not reachable at {OLLAMA_BASE_URL} — is it running?")
            return None

        except requests.exceptions.Timeout:
            logger.error(f"[{agent_name}] Ollama timed out after 300s")
            return None

        except Exception as e:
            logger.error(f"[{agent_name}] Ollama error: {e}")
            return None

    def _check_ollama_model(self, model: str) -> bool:
        """Check if a model is available in Ollama."""
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check exact or prefix match
            return any(m == model or m.startswith(model.split(":")[0]) for m in models)
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # Main public interface
    # -----------------------------------------------------------------------

    def call(
        self,
        prompt: str,
        system: str = "You are a helpful research assistant.",
        agent_name: str = "unknown",
        force_local: bool = False
    ) -> str:
        """
        Route LLM call with automatic fallback.
        
        Priority:
          1. Claude API (primary model for agent)
          2. Claude API (haiku fallback within API)
          3. Ollama primary model
          4. Ollama light model
          5. Raise exception — nothing available
        """
        models = self._get_models(agent_name)

        # Ensure client is initialized — handles case where .env was not loaded at import time
        if not force_local:
            self._ensure_anthropic()

        # Step 1: Try Claude API
        if not force_local and self.anthropic_client:
            result = self._call_claude(prompt, system, models["claude"], agent_name)
            if result:
                return result

            # Step 2: Try cheaper Claude model if primary failed
            if models["claude"] != CLAUDE_FALLBACK_MODEL:
                logger.info(f"[{agent_name}] Trying Claude fallback model: {CLAUDE_FALLBACK_MODEL}")
                result = self._call_claude(prompt, system, CLAUDE_FALLBACK_MODEL, agent_name)
                if result:
                    return result

        # Step 3: Try Ollama primary model
        logger.info(f"[{agent_name}] Falling back to Ollama — model: {models['ollama']}")
        if self._check_ollama_model(models["ollama"]):
            result = self._call_ollama(prompt, system, models["ollama"], agent_name)
            if result:
                return result
        else:
            logger.warning(f"[{agent_name}] Ollama model {models['ollama']} not available — trying light model")

        # Step 4: Try Ollama light model
        if models["ollama"] != OLLAMA_LIGHT_MODEL:
            logger.info(f"[{agent_name}] Trying Ollama light model: {OLLAMA_LIGHT_MODEL}")
            if self._check_ollama_model(OLLAMA_LIGHT_MODEL):
                result = self._call_ollama(prompt, system, OLLAMA_LIGHT_MODEL, agent_name)
                if result:
                    return result

        # Step 5: Nothing worked
        raise RuntimeError(
            f"[{agent_name}] All LLM backends failed. "
            f"Check ANTHROPIC_API_KEY and Ollama availability."
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: Optional[LLMClient] = None

def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client

def call(
    prompt: str,
    system: str = "You are a helpful research assistant.",
    agent_name: str = "unknown",
    force_local: bool = False
) -> str:
    """Convenience function — use this in agents."""
    return get_client().call(prompt, system, agent_name, force_local)
