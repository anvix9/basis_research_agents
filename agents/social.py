"""
Social Agent
------------
Two modes:
  1. collect(config)     — passive twice-weekly scan of all themes
  2. feed(problem, run_id, config) — targeted pull for a specific problem

Sources: OpenAlex, arXiv, PubMed, Semantic Scholar, CrossRef, 
         JSTOR, PhilPapers, SSRN, CORE, BASE, HAL
Consensus: called via the LLM if API key available.

All results saved to sources table with type='current'.
Dead links automatically archived.
"""

import re
import uuid
import time
import logging
import requests
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from core import database as db
from core.utils import generate_id, match_themes_to_problem, load_config
from core import llm
from core.rate_limiter import get_limiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source handlers
# ---------------------------------------------------------------------------

class SourceHandler:
    """Base class for all source handlers."""
    SOURCE_ID = "unknown"
    BASE_URL  = ""

    def search(self, query: str, keywords: list[str], limit: int = 10,
               run_id: str = "") -> list[dict]:
        raise NotImplementedError

    def _get(self, url: str, params: dict = None, timeout: int = 15,
             run_id: str = "") -> Optional[dict]:
        """Rate-limited GET with backoff on 429/5xx."""
        limiter = get_limiter(run_id)
        limiter.wait(self.SOURCE_ID)
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=timeout,
                                    headers={"User-Agent": "PipelineResearchBot/1.0 mailto:pipeline@research.local"})
                if resp.status_code == 429:
                    limiter.backoff(self.SOURCE_ID, attempt, 429)
                    continue
                if resp.status_code >= 500:
                    limiter.backoff(self.SOURCE_ID, attempt, resp.status_code)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.Timeout:
                logger.warning(f"[{self.SOURCE_ID}] Timeout (attempt {attempt+1}): {url}")
                if attempt < 2:
                    limiter.backoff(self.SOURCE_ID, attempt, 0)
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"[{self.SOURCE_ID}] Connection error: {e}")
                return None
            except Exception as e:
                logger.warning(f"[{self.SOURCE_ID}] Error: {e}")
                return None
        return None

    def _check_link(self, url: str) -> str:
        """Check if a URL is alive. Returns: active / dead / redirected."""
        if not url:
            return "dead"
        try:
            resp = requests.head(url, timeout=10, allow_redirects=True,
                                 headers={"User-Agent": "PipelineResearchBot/1.0"})
            if resp.status_code == 200:
                if str(resp.url) != url:
                    return "redirected"
                return "active"
            elif resp.status_code == 404:
                return "dead"
            else:
                return "active"  # treat other codes as live
        except Exception:
            return "dead"


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

class OpenAlexHandler(SourceHandler):
    SOURCE_ID = "openalex"
    BASE_URL  = "https://api.openalex.org/works"

    def search(self, query: str, keywords: list[str], limit: int = 10, run_id: str = "") -> list[dict]:
        from core.keys import openalex as get_key
        api_key = get_key()
        params = {"search": query, "per-page": limit,
                  "sort": "relevance_score:desc", "filter": "has_abstract:true"}
        if api_key:
            params["api_key"] = api_key
        else:
            params["mailto"] = "pipeline@research.local"
        data = self._get(self.BASE_URL, params=params, run_id=run_id)
        if not data:
            return []
        results = []
        for w in data.get("results", []):
            doi  = w.get("doi", "")
            link = doi if doi else w.get("id", "")
            abstract = ""
            if w.get("abstract_inverted_index"):
                inv   = w["abstract_inverted_index"]
                words = {}
                for word, positions in inv.items():
                    for pos in positions:
                        words[pos] = word
                abstract = " ".join(words[i] for i in sorted(words.keys()))[:1000]
            results.append({
                "title":       w.get("display_name", ""),
                "authors":     [a.get("author", {}).get("display_name", "") for a in w.get("authorships", [])[:5]],
                "year":        w.get("publication_year"),
                "source_name": self.SOURCE_ID,
                "doi":         doi,
                "abstract":    abstract,
                "active_link": link,
            })
        return results


# ---------------------------------------------------------------------------
# arXiv — official pip library handles 3s rate limit internally
# ---------------------------------------------------------------------------

class ArXivHandler(SourceHandler):
    SOURCE_ID = "arxiv"

    def search(self, query: str, keywords: list[str], limit: int = 10, run_id: str = "") -> list[dict]:
        limiter = get_limiter(run_id)
        limiter.wait(self.SOURCE_ID)
        try:
            import arxiv as arxiv_lib
            client = arxiv_lib.Client(page_size=min(limit, 100), delay_seconds=3, num_retries=3)
            search = arxiv_lib.Search(query=query, max_results=limit,
                                      sort_by=arxiv_lib.SortCriterion.Relevance)
            results = []
            for r in client.results(search):
                results.append({
                    "title":       r.title.replace("\n", " "),
                    "authors":     [a.name for a in r.authors[:5]],
                    "year":        r.published.year if r.published else None,
                    "source_name": self.SOURCE_ID,
                    "doi":         r.doi or "",
                    "abstract":    (r.summary or "")[:1000],
                    "active_link": r.entry_id or "",
                })
            return results
        except ImportError:
            logger.warning("[arXiv] arxiv library not installed — run: pip install arxiv")
            return []
        except Exception as e:
            logger.warning(f"[arXiv] Error: {e}")
            return []


# ---------------------------------------------------------------------------
# PubMed — optional NCBI API key (free): 3 req/s → 10 req/s
# ---------------------------------------------------------------------------

class PubMedHandler(SourceHandler):
    SOURCE_ID       = "pubmed"
    BASE_URL_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    BASE_URL_FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    def _ncbi_params(self, extra: dict) -> dict:
        from core.keys import ncbi_api_key, ncbi_email
        params = {**extra, "tool": "PipelineResearchBot"}
        key   = ncbi_api_key()
        email = ncbi_email()
        if key:
            params["api_key"] = key
        if email:
            params["email"] = email
        return params

    def search(self, query: str, keywords: list[str], limit: int = 10, run_id: str = "") -> list[dict]:
        search_data = self._get(self.BASE_URL_SEARCH,
            params=self._ncbi_params({"db": "pubmed", "term": query,
                                      "retmax": limit, "retmode": "json"}),
            run_id=run_id)
        if not search_data:
            return []
        ids = search_data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        summary_data = self._get(self.BASE_URL_FETCH,
            params=self._ncbi_params({"db": "pubmed", "id": ",".join(ids), "retmode": "json"}),
            run_id=run_id)
        if not summary_data:
            return []
        results = []
        for uid in summary_data.get("result", {}).get("uids", []):
            art      = summary_data["result"].get(uid, {})
            pub_date = art.get("pubdate", "")[:4]
            results.append({
                "title":       art.get("title", ""),
                "authors":     [a.get("name", "") for a in art.get("authors", [])[:5]],
                "year":        int(pub_date) if pub_date.isdigit() else None,
                "source_name": self.SOURCE_ID,
                "doi":         art.get("elocationid", ""),
                "abstract":    "",
                "active_link": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
            })
        return results


# ---------------------------------------------------------------------------
# Semantic Scholar — optional API key (free request): dedicated 1 req/s
# ---------------------------------------------------------------------------

class SemanticScholarHandler(SourceHandler):
    SOURCE_ID = "semantic_scholar"
    BASE_URL  = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search(self, query: str, keywords: list[str], limit: int = 10, run_id: str = "") -> list[dict]:
        from core.keys import semantic_scholar as get_key
        limiter = get_limiter(run_id)
        limiter.wait(self.SOURCE_ID)
        headers = {"User-Agent": "PipelineResearchBot/1.0"}
        key = get_key()
        if key:
            headers["x-api-key"] = key
        for attempt in range(3):
            try:
                resp = requests.get(self.BASE_URL,
                    params={"query": query, "limit": limit,
                            "fields": "title,authors,year,abstract,externalIds,url"},
                    headers=headers, timeout=20)
                if resp.status_code == 429:
                    limiter.backoff(self.SOURCE_ID, attempt, 429)
                    continue
                if resp.status_code >= 500:
                    limiter.backoff(self.SOURCE_ID, attempt, resp.status_code)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                logger.warning(f"[SemanticScholar] Error: {e}")
                return []
        else:
            return []
        results = []
        for p in data.get("data", []):
            doi = p.get("externalIds", {}).get("DOI", "")
            results.append({
                "title":       p.get("title", ""),
                "authors":     [a.get("name", "") for a in p.get("authors", [])[:5]],
                "year":        p.get("year"),
                "source_name": self.SOURCE_ID,
                "doi":         doi,
                "abstract":    (p.get("abstract") or "")[:1000],
                "active_link": p.get("url", "") or (f"https://doi.org/{doi}" if doi else ""),
            })
        return results


# ---------------------------------------------------------------------------
# CORE — optional API key (free): higher rate limits
# ---------------------------------------------------------------------------

class COREHandler(SourceHandler):
    SOURCE_ID = "core"
    BASE_URL  = "https://api.core.ac.uk/v3/search/works"

    def search(self, query: str, keywords: list[str], limit: int = 10, run_id: str = "") -> list[dict]:
        from core.keys import core as get_key
        limiter = get_limiter(run_id)
        limiter.wait(self.SOURCE_ID)
        key = get_key()
        headers = {"User-Agent": "PipelineResearchBot/1.0"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        for attempt in range(3):
            try:
                resp = requests.post(self.BASE_URL, json={"q": query, "limit": limit},
                                     headers=headers, timeout=15)
                if resp.status_code == 429:
                    limiter.backoff(self.SOURCE_ID, attempt, 429)
                    continue
                if resp.status_code >= 500:
                    limiter.backoff(self.SOURCE_ID, attempt, resp.status_code)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                logger.warning(f"[CORE] Error: {e}")
                return []
        else:
            return []
        results = []
        for item in data.get("results", []):
            results.append({
                "title":       item.get("title", ""),
                "authors":     [a.get("name", "") for a in item.get("authors", [])[:5]],
                "year":        item.get("yearPublished"),
                "source_name": self.SOURCE_ID,
                "doi":         item.get("doi", ""),
                "abstract":    (item.get("abstract") or "")[:1000],
                "active_link": item.get("downloadUrl") or item.get("sourceFulltextUrls", [None])[0] or "",
            })
        return results


# ---------------------------------------------------------------------------
# PhilPapers — API ID + Key required (free registration)
# OAI-PMH fallback for open access content without credentials
# ---------------------------------------------------------------------------

class PhilPapersHandler(SourceHandler):
    """PhilPapers — requires free API registration at philpapers.org/utils/create_api_user.html
    Without credentials this handler silently returns nothing.
    Use philarchive or philsci as alternatives."""
    SOURCE_ID = "philpapers"
    BASE_URL  = "https://philpapers.org/api/articles.json"

    def search(self, query: str, keywords: list[str], limit: int = 10, run_id: str = "") -> list[dict]:
        from core.keys import philpapers_id, philpapers_key
        api_id  = philpapers_id()
        api_key = philpapers_key()
        if not api_id or not api_key:
            logger.info("[PhilPapers] No credentials — skipping (add to .env or use philarchive/philsci)")
            return []
        data = self._get(self.BASE_URL, params={
            "q": query, "ps": limit,
            "apiId": api_id, "apiKey": api_key
        }, run_id=run_id)
        if not data:
            return []
        results = []
        for item in (data if isinstance(data, list) else data.get("articles", []))[:limit]:
            results.append({
                "title":       item.get("title", ""),
                "authors":     item.get("authors", [])[:5],
                "year":        item.get("pub_year"),
                "source_name": self.SOURCE_ID,
                "doi":         item.get("doi", ""),
                "abstract":    (item.get("abstract") or "")[:1000],
                "active_link": item.get("url", ""),
            })
        return results




# ---------------------------------------------------------------------------
# PhilArchive — PhilPapers' own open access archive, no account needed
# 115k+ philosophy papers, OAI-PMH interface
# ---------------------------------------------------------------------------

class PhilArchiveHandler(SourceHandler):
    SOURCE_ID = "philarchive"
    OAI_URL   = "https://philarchive.org/oai.pl"

    def search(self, query: str, keywords: list[str], limit: int = 10, run_id: str = "") -> list[dict]:
        import xml.etree.ElementTree as ET
        limiter = get_limiter(run_id)
        limiter.wait(self.SOURCE_ID)
        try:
            resp = requests.get(self.OAI_URL, params={
                "verb": "ListRecords",
                "metadataPrefix": "oai_dc"
            }, timeout=25, headers={"User-Agent": "PipelineResearchBot/1.0"})
            if resp.status_code == 403:
                logger.warning("[PhilArchive] 403 Forbidden — endpoint may require browser session")
                return []
            resp.raise_for_status()
            ns = {
                "oai":    "http://www.openarchives.org/OAI/2.0/",
                "dc":     "http://purl.org/dc/elements/1.1/",
                "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/"
            }
            root = ET.fromstring(resp.text)
            query_terms = query.lower().replace('"', '').split()
            results = []
            for record in root.findall(".//oai:record", ns):
                metadata = record.find(".//oai_dc:dc", ns)
                if metadata is None:
                    continue
                title_el    = metadata.find("dc:title", ns)
                desc_el     = metadata.find("dc:description", ns)
                creator_els = metadata.findall("dc:creator", ns)
                id_el       = metadata.find("dc:identifier", ns)
                date_el     = metadata.find("dc:date", ns)
                title    = (title_el.text or "") if title_el is not None else ""
                abstract = (desc_el.text  or "") if desc_el  is not None else ""
                searchable = (title + " " + abstract).lower()
                if not any(t in searchable for t in query_terms):
                    continue
                link = (id_el.text or "") if id_el is not None else ""
                year = None
                if date_el is not None and date_el.text:
                    year = int(date_el.text[:4]) if date_el.text[:4].isdigit() else None
                results.append({
                    "title":       title,
                    "authors":     [e.text for e in creator_els[:5] if e.text],
                    "year":        year,
                    "source_name": self.SOURCE_ID,
                    "doi":         "",
                    "abstract":    abstract[:1000],
                    "active_link": link,
                })
                if len(results) >= limit:
                    break
            return results
        except Exception as e:
            logger.warning(f"[PhilArchive] Error: {e}")
            return []


# ---------------------------------------------------------------------------
# PhilSci-Archive — open philosophy of science preprints, no account needed
# University of Pittsburgh, fully public OAI-PMH
# ---------------------------------------------------------------------------

class PhilSciHandler(SourceHandler):
    SOURCE_ID = "philsci"
    OAI_URL   = "https://philsci-archive.pitt.edu/cgi/oai2"

    def search(self, query: str, keywords: list[str], limit: int = 10, run_id: str = "") -> list[dict]:
        import xml.etree.ElementTree as ET
        limiter = get_limiter(run_id)
        limiter.wait(self.SOURCE_ID)
        try:
            resp = requests.get(self.OAI_URL, params={
                "verb": "ListRecords",
                "metadataPrefix": "oai_dc"
            }, timeout=25, headers={"User-Agent": "PipelineResearchBot/1.0"})
            if resp.status_code in (403, 503):
                logger.warning(f"[PhilSci] {resp.status_code} — skipping")
                return []
            resp.raise_for_status()
            ns = {
                "oai":    "http://www.openarchives.org/OAI/2.0/",
                "dc":     "http://purl.org/dc/elements/1.1/",
                "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/"
            }
            root = ET.fromstring(resp.text)
            query_terms = query.lower().replace('"', '').split()
            results = []
            for record in root.findall(".//oai:record", ns):
                metadata = record.find(".//oai_dc:dc", ns)
                if metadata is None:
                    continue
                title_el    = metadata.find("dc:title", ns)
                desc_el     = metadata.find("dc:description", ns)
                creator_els = metadata.findall("dc:creator", ns)
                id_el       = metadata.find("dc:identifier", ns)
                date_el     = metadata.find("dc:date", ns)
                title    = (title_el.text or "") if title_el is not None else ""
                abstract = (desc_el.text  or "") if desc_el  is not None else ""
                searchable = (title + " " + abstract).lower()
                if not any(t in searchable for t in query_terms):
                    continue
                link = (id_el.text or "") if id_el is not None else ""
                year = None
                if date_el is not None and date_el.text:
                    year = int(date_el.text[:4]) if date_el.text[:4].isdigit() else None
                results.append({
                    "title":       title,
                    "authors":     [e.text for e in creator_els[:5] if e.text],
                    "year":        year,
                    "source_name": self.SOURCE_ID,
                    "doi":         "",
                    "abstract":    abstract[:1000],
                    "active_link": link,
                })
                if len(results) >= limit:
                    break
            return results
        except Exception as e:
            logger.warning(f"[PhilSci] Error: {e}")
            return []


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

SOURCE_HANDLERS = {
    "openalex":         OpenAlexHandler(),
    "arxiv":            ArXivHandler(),
    "pubmed":           PubMedHandler(),
    "semantic_scholar": SemanticScholarHandler(),
    "core":             COREHandler(),
    "philpapers":       PhilPapersHandler(),
    "philarchive":      PhilArchiveHandler(),
    "philsci":          PhilSciHandler(),
}


# ---------------------------------------------------------------------------
# Relevance rating via LLM
# ---------------------------------------------------------------------------

RATING_SYSTEM = """You are a research relevance evaluator.
Given a research problem, a theme, and a paper's title and abstract,
rate the paper's relevance as High, Medium, or Low.
Respond with ONLY a JSON object: {"rating": "High|Medium|Low", "reason": "one sentence"}
Do not include any other text."""

def rate_relevance(title: str, abstract: str, problem: str, theme_label: str) -> tuple[str, str]:
    """Use LLM to rate relevance. Returns (rating, reason)."""
    prompt = (
        f"Research problem: {problem}\n"
        f"Theme: {theme_label}\n"
        f"Paper title: {title}\n"
        f"Abstract: {abstract[:500]}\n"
        f"Rate the relevance."
    )
    try:
        response = llm.call(prompt, RATING_SYSTEM, agent_name="social")
        # Extract JSON
        import json
        match = re.search(r'\{.*?\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("rating", "Medium"), data.get("reason", "")
    except Exception as e:
        logger.warning(f"Relevance rating failed: {e}")
    return "Medium", "Could not assess relevance"


# ---------------------------------------------------------------------------
# Core collection logic
# ---------------------------------------------------------------------------

def _build_query(theme: dict, expansion: bool = True) -> str:
    """Build search query from theme keywords."""
    keywords = theme.get("keywords", [])
    seeds = [kw.get("seed", "") for kw in keywords if kw.get("seed")]
    if not seeds:
        return theme.get("label", "")
    # Primary query: all seeds
    return " OR ".join(f'"{s}"' for s in seeds[:3])


def _collect_for_theme(
    theme: dict,
    sources: list[str],
    config: dict,
    problem: str = "",
    limit_per_source: int = 8,
    run_id: str = "",
    theme_index: int = 0,
    theme_total: int = 1
) -> list[dict]:
    """Collect papers for a single theme from its configured sources."""
    query = _build_query(theme)
    theme_id = theme.get("theme_id", "")
    theme_label = theme.get("label", theme_id)
    collected = []

    limiter = get_limiter(run_id)

    # Count enabled sources for progress
    enabled_sources = [
        s for s in sources
        if config.get("sources", {}).get(s, {}).get("enabled", True)
        and SOURCE_HANDLERS.get(s)
    ]
    total_sources = len(enabled_sources)

    print(f"\n  {'─'*55}")
    print(f"  Theme {theme_index+1}/{theme_total}: {theme_label} ({theme_id})")
    print(f"  Query: {query[:70]}")
    print(f"  Sources: {', '.join(enabled_sources)}")
    print(f"  {'─'*55}")

    for src_idx, source_id in enumerate(sources):
        # Check if source is enabled in config
        source_cfg = config.get("sources", {}).get(source_id, {})
        if not source_cfg.get("enabled", True):
            continue

        handler = SOURCE_HANDLERS.get(source_id)
        if not handler:
            logger.warning(f"[Social] No handler for source: {source_id}")
            continue

        limiter.print_source_start(source_id, theme_id, query)

        try:
            results = handler.search(
                query,
                [kw.get("seed","") for kw in theme.get("keywords",[])],
                limit_per_source,
                run_id=run_id
            )
            limiter.print_source_done(source_id, len(results))
        except Exception as e:
            logger.warning(f"[Social] {source_id} search failed: {e}")
            print(f"\r  ✗ [{source_id}] failed: {str(e)[:60]}{' '*20}")
            results = []

        for r in results:
            if not r.get("title"):
                continue

            # Rate relevance
            rating, reason = rate_relevance(
                r.get("title", ""),
                r.get("abstract", ""),
                problem or theme_label,
                theme_label
            )

            # Check link
            link_status = handler._check_link(r.get("active_link", ""))

            source_entry = {
                "source_id":       generate_id("SRC"),
                "title":           r.get("title", ""),
                "authors":         r.get("authors", []),
                "year":            r.get("year"),
                "source_name":     source_id,
                "doi":             r.get("doi", ""),
                "abstract":        r.get("abstract", ""),
                "active_link":     r.get("active_link", ""),
                "theme_tags":      [theme_id],
                "type":            "current",
                "relevance_rating": rating,
                "relevance_reason": reason,
                "added_by":        "Social",
                "date_collected":  datetime.now(timezone.utc).isoformat(),
                "last_checked":    datetime.now(timezone.utc).isoformat(),
                "link_status":     link_status,
            }

            # Archive dead links immediately
            if link_status == "dead":
                db.insert("dead_links", {
                    "dead_id":              generate_id("DEAD"),
                    "source_id":            source_entry["source_id"],
                    "title":                source_entry["title"],
                    "original_link":        source_entry["active_link"],
                    "theme_tags":           str([theme_id]),
                    "type":                 "current",
                    "date_collected":       source_entry["date_collected"],
                    "date_confirmed_dead":  source_entry["date_collected"],
                    "last_active":          None
                })
                logger.warning(f"[Social] Dead link archived: {source_entry['title'][:60]}")
                continue  # Don't save dead links to active database

            collected.append(source_entry)

    return collected


# ---------------------------------------------------------------------------
# Link recheck
# ---------------------------------------------------------------------------

def recheck_links() -> dict:
    """
    Recheck all active links in the database.
    Move dead links to dead_links archive.
    Returns summary dict.
    """
    logger.info("[Social] Starting link recheck...")
    sources = db.get_sources_by_type("current")
    sources += db.get_sources_by_type("historical")
    # Seminal links are never auto-archived — flagged only

    summary = {"checked": 0, "active": 0, "redirected": 0, "dead": 0, "flagged": 0}
    handler = SourceHandler()

    for source in sources:
        link = source.get("active_link", "")
        if not link:
            continue

        status = handler._check_link(link)
        summary["checked"] += 1

        if status == "active":
            db.update("sources", {"last_checked": datetime.now(timezone.utc).isoformat()},
                      {"source_id": source["source_id"]})
            summary["active"] += 1

        elif status == "redirected":
            db.update("sources", {
                "link_status": "redirected",
                "last_checked": datetime.now(timezone.utc).isoformat()
            }, {"source_id": source["source_id"]})
            summary["redirected"] += 1
            logger.info(f"[Social] Redirected: {source.get('title','')[:60]}")

        elif status == "dead":
            if source.get("type") == "seminal":
                # Flag for manual review, never auto-archive seminal
                db.update("sources", {"link_status": "flagged"},
                          {"source_id": source["source_id"]})
                summary["flagged"] += 1
                logger.warning(f"[Social] Seminal link flagged (manual review): {source.get('title','')[:60]}")
            else:
                db.archive_dead_link(source)
                summary["dead"] += 1
                logger.warning(f"[Social] Dead link archived: {source.get('title','')[:60]}")

        time.sleep(0.2)

    logger.info(f"[Social] Recheck complete: {summary}")
    return summary


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def collect(config: dict = None) -> dict:
    """
    Mode 1: Passive collector.
    Scans all themes in the config and saves to database.
    Run twice a week via cron.
    Returns summary.
    """
    from core.rate_limiter import reset_limiter
    if config is None:
        config = load_config()

    themes = config.get("themes", [])
    run_id = f"collect-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    limiter = reset_limiter(run_id)

    logger.info(f"[Social] Passive collection starting — {len(themes)} themes")
    summary = {"themes_scanned": 0, "sources_collected": 0, "dead_links": 0}

    print(f"\n  {'='*55}")
    print(f"  Social Agent — Passive Collection")
    print(f"  Themes: {len(themes)} | Started: {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
    print(f"  {'='*55}")

    for idx, theme in enumerate(themes):
        sources_for_theme = theme.get("sources", [])
        collected = _collect_for_theme(
            theme, sources_for_theme, config,
            limit_per_source=10,
            run_id=run_id,
            theme_index=idx,
            theme_total=len(themes)
        )

        for entry in collected:
            db.upsert_source(entry)
            summary["sources_collected"] += 1

        summary["themes_scanned"] += 1
        print(f"  ✓ Theme {idx+1}/{len(themes)}: {theme.get('theme_id')} — {len(collected)} sources")

    # Recheck existing links
    print(f"\n  Rechecking existing links...")
    recheck = recheck_links()
    summary["dead_links"] = recheck["dead"]

    limiter.print_run_summary()
    logger.info(f"[Social] Passive collection complete: {summary}")
    return summary


def feed(problem: str, run_id: str, config: dict = None,
         selected_themes: list = None) -> tuple[list[dict], list[dict]]:
    """
    Mode 2: Pipeline feeder.
    If selected_themes provided (from concept mapper), uses those directly.
    Otherwise falls back to keyword matching.
    """
    from core.rate_limiter import reset_limiter
    if config is None:
        config = load_config()

    themes = config.get("themes", [])
    limiter = reset_limiter(run_id)

    if selected_themes is not None:
        selected = selected_themes
        selected_ids = {t["theme_id"] for t in selected}
        excluded = [{"theme_id": t["theme_id"], "label": t.get("label",""),
                     "reason": "Not activated by concept mapper"}
                    for t in themes if t["theme_id"] not in selected_ids]
    else:
        selected, excluded = match_themes_to_problem(problem, themes)

    logger.info(
        f"[Social] Pipeline feed for run {run_id}: "
        f"{len(selected)} themes selected, {len(excluded)} excluded"
    )

    print(f"\n  {'='*55}")
    print(f"  Social Agent — Pipeline Feed")
    print(f"  Problem: {problem[:60]}")
    print(f"  Themes selected: {len(selected)} | Excluded: {len(excluded)}")
    print(f"  {'='*55}")

    for idx, theme in enumerate(selected):
        sources_for_theme = theme.get("sources", [])
        collected = _collect_for_theme(
            theme, sources_for_theme, config,
            problem=problem, limit_per_source=8,
            run_id=run_id,
            theme_index=idx,
            theme_total=len(selected)
        )
        for entry in collected:
            entry["run_id"] = run_id
            db.upsert_source(entry)

        print(f"  ✓ Theme {idx+1}/{len(selected)}: {theme.get('theme_id')} — {len(collected)} sources")

    # Also pull existing database entries for selected themes
    existing = []
    for theme in selected:
        theme_id = theme.get("theme_id", "")
        # Fetch active sources for this theme from database
        rows = db.fetch("sources", {"link_status": "active"})
        for row in rows:
            theme_tags = row.get("theme_tags", "[]")
            if theme_id in str(theme_tags):
                existing.append(row)

    logger.info(f"[Social] Feed complete — {len(existing)} existing sources available for run {run_id}")
    return selected, excluded


def produce_intelligence_package(run_id: str, selected_themes: list, problem: str) -> str:
    """
    Produce a formatted intelligence package for the pipeline.
    This is what agents receive as 'Social intelligence'.
    """
    sources = db.fetch("sources", {"run_id": run_id})
    if not sources:
        sources = db.get_sources_by_type("current")

    lines = [
        f"# Social Intelligence Package",
        f"**Problem:** {problem}",
        f"**Themes covered:** {', '.join(t.get('theme_id','') for t in selected_themes)}",
        f"**Sources:** {len(sources)}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "---",
        ""
    ]

    # Group by theme
    theme_ids = [t.get("theme_id", "") for t in selected_themes]
    for theme_id in theme_ids:
        theme_sources = [
            s for s in sources
            if theme_id in str(s.get("theme_tags", ""))
        ]
        if not theme_sources:
            continue

        lines.append(f"## {theme_id}")
        lines.append("")

        # High relevance first
        for rating in ["High", "Medium", "Low"]:
            rated = [s for s in theme_sources if s.get("relevance_rating") == rating]
            if rated:
                lines.append(f"### {rating} Relevance")
                for s in rated[:10]:
                    authors = s.get("authors", "[]")
                    if isinstance(authors, str):
                        import json
                        try:
                            authors = json.loads(authors)
                        except Exception:
                            authors = [authors]
                    author_str = ", ".join(authors[:3])
                    lines.append(
                        f"- **{s.get('title','')}** ({author_str}, {s.get('year','n.d.')})"
                        f"\n  {s.get('relevance_reason','')}"
                        f"\n  Link: {s.get('active_link','')}"
                        f"\n  Abstract: {(s.get('abstract') or '')[:200]}..."
                    )
                lines.append("")

    return "\n".join(lines)
