"""
Rate Limiter
------------
Per-source rate limiting with progress display for all Social agent API calls.

Each source has:
  - min_delay: seconds to wait between calls (from official docs)
  - daily_limit: maximum calls per day (None = unlimited)
  - Exponential backoff on 429 / 5xx

Progress bar shows:
  - Current source being queried
  - Calls made / remaining for this run
  - Wait countdown when throttling
"""

import time
import logging
from datetime import datetime, timezone
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Official rate limits per source (from documentation)
# ---------------------------------------------------------------------------

SOURCE_LIMITS = {
    # source_id: (min_delay_seconds, daily_limit, notes)
    "openalex":         (0.15,  100_000, "10 req/sec free; needs API key Feb 2026+"),
    "arxiv":            (3.0,   None,    "Official ToS: 1 req per 3 seconds strictly"),
    "pubmed":           (0.4,   None,    "NCBI: 3 req/sec without API key, 10 with key"),
    "semantic_scholar": (3.5,   None,    "100 req/5 min unauthenticated"),
    "core":             (0.6,   None,    "~2 req/sec on free tier"),
    "philpapers":       (2.0,   None,    "No official docs — conservative"),
    "philarchive":      (2.0,   None,    "PhilPapers open archive — OAI-PMH"),
    "philsci":          (2.0,   None,    "PhilSci-Archive Pittsburgh — OAI-PMH"),
    "scopus":           (1.0,   None,    "Elsevier Scopus — needs institutional IP or VPN"),
    "consensus":        (0.5,   None,    "Consensus semantic search — 200M+ papers"),
    "jstor":            (2.0,   None,    "Conservative — limited API access"),
    "ssrn":             (2.0,   None,    "Conservative"),
    "base":             (1.0,   None,    "BASE search API"),
    "hal":              (0.5,   None,    "HAL open API"),
    "eric":             (0.5,   None,    "IES ERIC API"),
    "nber":             (1.0,   None,    "NBER working papers"),
    "persee":           (1.0,   None,    "Persée French archive"),
    "crossref":         (1.0,   None,    "Crossref polite pool"),
    "default":          (1.0,   None,    "Unknown source — safe default"),
}

# Backoff settings
MAX_RETRIES      = 3
BACKOFF_BASE     = 2.0   # seconds — doubles each retry
BACKOFF_MAX      = 30.0  # cap at 30 seconds


# ---------------------------------------------------------------------------
# Rate limiter class
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Thread-safe per-source rate limiter with progress tracking.
    One instance per pipeline run — tracks all calls made.
    """

    def __init__(self, run_id: str = ""):
        self.run_id = run_id
        self._locks: dict[str, Lock] = {}
        self._last_call: dict[str, float] = {}
        self._call_counts: dict[str, int] = {}
        self._total_calls = 0
        self._start_time = time.time()

    def _get_lock(self, source_id: str) -> Lock:
        if source_id not in self._locks:
            self._locks[source_id] = Lock()
        return self._locks[source_id]

    def _get_config(self, source_id: str) -> tuple[float, Optional[int]]:
        config = SOURCE_LIMITS.get(source_id, SOURCE_LIMITS["default"])
        return config[0], config[1]  # (min_delay, daily_limit)

    def wait(self, source_id: str, call_label: str = ""):
        """
        Block until it is safe to make a call to source_id.
        Respects min_delay between calls.
        Prints a countdown if waiting more than 0.5s.
        """
        min_delay, daily_limit = self._get_config(source_id)
        lock = self._get_lock(source_id)

        with lock:
            now = time.time()
            last = self._last_call.get(source_id, 0)
            elapsed = now - last
            remaining = min_delay - elapsed

            if remaining > 0:
                if remaining > 0.5:
                    self._print_wait(source_id, remaining, call_label)
                time.sleep(remaining)

            # Check daily limit
            count = self._call_counts.get(source_id, 0)
            if daily_limit and count >= daily_limit:
                logger.warning(f"[RateLimit] Daily limit reached for {source_id}: {daily_limit}")
                return

            # Record call
            self._last_call[source_id] = time.time()
            self._call_counts[source_id] = count + 1
            self._total_calls += 1

    def backoff(self, source_id: str, attempt: int, status_code: int = 0):
        """
        Exponential backoff after a failed request.
        Call this when you get a 429 or 5xx response.
        """
        wait = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
        if status_code == 429:
            wait = max(wait, 10.0)  # 429 always wait at least 10s
            logger.warning(f"[RateLimit] 429 on {source_id} — backing off {wait:.0f}s")
        else:
            logger.warning(f"[RateLimit] {status_code} on {source_id} — backing off {wait:.0f}s")
        self._print_wait(source_id, wait, f"backoff after {status_code}")
        time.sleep(wait)

    def _print_wait(self, source_id: str, seconds: float, label: str = ""):
        """Print a visible wait notice with countdown."""
        label_str = f" ({label})" if label else ""
        if seconds >= 2.0:
            # Countdown for long waits
            print(f"\r  ⏳ [{source_id}]{label_str} — waiting {seconds:.1f}s", end="", flush=True)
            start = time.time()
            while True:
                elapsed = time.time() - start
                left = seconds - elapsed
                if left <= 0:
                    break
                print(f"\r  ⏳ [{source_id}]{label_str} — waiting {left:.1f}s  ", end="", flush=True)
                time.sleep(0.2)
            print(f"\r  ✓ [{source_id}] ready{' '*30}", flush=True)
        else:
            # Short wait — just show a brief message
            print(f"\r  ⏳ [{source_id}] {seconds:.2f}s... ", end="", flush=True)

    def print_progress(
        self,
        source_id: str,
        current: int,
        total: int,
        label: str = ""
    ):
        """Print a progress line for the current source."""
        bar_width = 20
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = int(100 * current / total) if total > 0 else 0
        elapsed = time.time() - self._start_time
        total_made = self._call_counts.get(source_id, 0)
        label_str = f" {label}" if label else ""
        print(
            f"\r  [{source_id}] [{bar}] {pct}% ({current}/{total}){label_str}"
            f" | total calls: {self._total_calls} | {elapsed:.0f}s",
            end="", flush=True
        )

    def print_source_start(self, source_id: str, theme_id: str, query: str):
        """Print a header when starting a new source query."""
        delay, limit = self._get_config(source_id)
        count = self._call_counts.get(source_id, 0)
        limit_str = f"/{limit}" if limit else ""
        print(
            f"\n  → [{source_id}] theme={theme_id} | "
            f"calls: {count}{limit_str} | "
            f"delay: {delay}s | "
            f"query: {query[:60]}"
        )

    def print_source_done(self, source_id: str, results: int):
        """Print completion line for a source."""
        print(f"\r  ✓ [{source_id}] {results} results returned{' '*40}")

    def print_run_summary(self):
        """Print summary of all API calls made in this run."""
        elapsed = time.time() - self._start_time
        print(f"\n  {'─'*50}")
        print(f"  API Call Summary — run {self.run_id}")
        print(f"  {'─'*50}")
        print(f"  Total calls:  {self._total_calls}")
        print(f"  Time elapsed: {elapsed:.0f}s")
        print(f"\n  By source:")
        for src, count in sorted(self._call_counts.items(), key=lambda x: -x[1]):
            delay, limit = self._get_config(src)
            limit_str = f"/{limit}" if limit else "/∞"
            print(f"    {src:<20} {count:>5} calls{limit_str}")
        print(f"  {'─'*50}\n")


# ---------------------------------------------------------------------------
# Module-level singleton per run
# ---------------------------------------------------------------------------

_limiter: Optional[RateLimiter] = None

def get_limiter(run_id: str = "") -> RateLimiter:
    global _limiter
    if _limiter is None or (run_id and _limiter.run_id != run_id):
        _limiter = RateLimiter(run_id)
    return _limiter

def reset_limiter(run_id: str = ""):
    global _limiter
    _limiter = RateLimiter(run_id)
    return _limiter
