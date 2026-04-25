"""
Grounder Agent
--------------
Excavates the intellectual origins of the research problem.

Pipeline:
  1. DECOMPOSE  — LLM breaks problem into complete sub-question tree
  2. QUERY GEN  — each sub-question → 2-3 contextual search queries
  3. SEARCH     — OpenAlex, arXiv, Semantic Scholar, Google Books, Open Library,
                  web search (Claude native) for each query
  4. SYNTHESIZE — LLM maps intellectual genealogy from all gathered sources
  5. SAVE       — seminal works → DB, foundations doc → artifacts/

Books are first-class sources here alongside papers.
Web search covers breadth that academic APIs miss.
"""

import re
import json
import time
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core import database as db
from core import llm
from core.utils import generate_id, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Problem decomposition prompt
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = """You are a research decomposition specialist.

Given a research problem, decompose it into the most complete and exhaustive tree of sub-questions needed to fully understand and answer it. 

Rules:
- Work from fundamentals upward — start with definitional questions, then structural, then relational, then positional
- Every concept that appears in the problem must be unpacked
- Ask questions that a researcher would need to answer BEFORE addressing the main problem
- Include both empirical questions ("what is X?") and conceptual questions ("what does X mean?")
- Typical depth: 6-12 sub-questions for a philosophical/interdisciplinary problem

Example: "What is the place of AI in human life?"
→ What is intelligence? What forms does intelligence take? What are the defining characteristics of human intelligence? What distinguishes human intelligence from other forms? What is artificial intelligence? What are AI's core characteristics and limitations? How does AI processing differ structurally from human cognition? What does "place" mean — functional role, ontological status, normative position? How have technologies previously been positioned relative to human life? What is the relationship between a tool and the being that uses it? How should AI be positioned in human life given the above?

Output ONLY valid JSON:
{
  "sub_questions": [
    {
      "id": "Q1",
      "question": "full question text",
      "level": "foundational|structural|relational|positional",
      "rationale": "why this sub-question must be answered"
    }
  ],
  "decomposition_logic": "one paragraph explaining the decomposition strategy"
}"""


# ---------------------------------------------------------------------------
# Step 2 — Query generation prompt
# ---------------------------------------------------------------------------

QUERY_GEN_SYSTEM = """You are a research query specialist.

Given a sub-question and its context, generate targeted search queries for academic databases and book catalogs.

Rules for queries:
- NEVER use single words alone — always combine keyword + 1-2 word context
- Academic queries: combine the core concept with its disciplinary context
  BAD: "intelligence"
  GOOD: "human intelligence definition", "intelligence forms cognitive science", "intelligence measurement history"
- Book queries: use author names + concept, or classic title keywords
  GOOD: "Turing computing machinery intelligence", "Dreyfus artificial intelligence critique", "intelligence philosophy mind"
- Generate exactly 3 queries: one for academic papers, one for books, one broader/web

Output ONLY valid JSON:
{
  "paper_query": "2-4 word academic query",
  "book_query": "2-4 word book/monograph query",
  "web_query": "3-5 word broader search query"
}"""


# ---------------------------------------------------------------------------
# Step 3 — Synthesis prompt
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """You are the Grounder agent in a multi-agent research pipeline.

Your role is to excavate the intellectual origins of the research problem using the gathered sources.

You have been given:
- The decomposed sub-questions
- Search results from academic databases, book catalogs, and web search

From this material, synthesize the intellectual foundations:
1. Extract all core themes embedded in the problem
2. For each theme, identify the oldest, most influential foundational works from the results
3. Find where themes intersected and produced foundational questions
4. Extract original definitions — how key concepts were first defined and by whom
5. Establish the fundamental whys — what original motivations gave birth to this problem
6. Map the intellectual genealogy — who built on whom

Search backward in time — prioritize oldest cited works.
Do NOT analyze current state, identify gaps, or propose solutions.
Include BOOKS alongside papers — foundational books matter as much as articles.

Output ONLY valid JSON:
{
  "themes_extracted": [
    {"theme": "name", "description": "why relevant to problem"}
  ],
  "seminal_works": [
    {
      "title": "full title",
      "authors": ["Author Name"],
      "year": 1950,
      "source": "source name",
      "material_type": "paper|book|chapter",
      "doi": "",
      "isbn": "",
      "abstract": "brief description of what it established",
      "active_link": "url if known",
      "seminal_reason": "one line — what it established and why foundational",
      "intersection_tags": ["theme1 x theme2"],
      "theme_tags": ["theme1"]
    }
  ],
  "intellectual_genealogy": "narrative of who built on whom — at least 3 paragraphs",
  "fundamental_whys": "original motivations behind this problem — at least 2 paragraphs",
  "original_definitions": [
    {"concept": "name", "definition": "text", "defined_by": "who", "year": 0}
  ],
  "intersection_points": [
    {"themes": ["t1", "t2"], "description": "how they met and what question emerged"}
  ],
  "proposed_new_themes": [
    {
      "theme_id": "snake_case_id",
      "label": "Human readable label",
      "reason": "why relevant but missing from config",
      "suggested_keywords": [
        {"seed": "keyword", "expansion_depth": 1, "boundary_note": "stay within..."}
      ],
      "suggested_sources": ["openalex"]
    }
  ],
  "assumptions_flagged": [
    {"assumption": "text", "note": "why disputed or unclear"}
  ]
}"""


# ---------------------------------------------------------------------------
# Source handlers — books + papers + web
# ---------------------------------------------------------------------------

def _search_openalex(query: str, limit: int = 5) -> list[dict]:
    """Search OpenAlex with contextual query."""
    from core.keys import openalex as get_key
    try:
        params = {
            "search":   query,
            "per-page": limit,
            "sort":     "relevance_score:desc",
            "filter":   "has_abstract:true",
        }
        key = get_key()
        if key:
            params["api_key"] = key
        else:
            params["mailto"] = "pipeline@research.local"
        resp = requests.get("https://api.openalex.org/works", params=params,
                            timeout=15, headers={"User-Agent": "PipelineResearchBot/1.0"})
        resp.raise_for_status()
        data = resp.json()
        results = []
        for w in data.get("results", []):
            # Reconstruct abstract from inverted index
            abstract = ""
            if w.get("abstract_inverted_index"):
                words = {}
                for word, positions in w["abstract_inverted_index"].items():
                    for pos in positions:
                        words[pos] = word
                abstract = " ".join(words[i] for i in sorted(words))[:800]
            doi = w.get("doi", "")
            results.append({
                "title":         w.get("display_name", ""),
                "authors":       [a.get("author",{}).get("display_name","") for a in w.get("authorships",[])[:3]],
                "year":          w.get("publication_year"),
                "material_type": "paper",
                "source":        "openalex",
                "doi":           doi,
                "abstract":      abstract,
                "active_link":   doi or w.get("id",""),
            })
        return results
    except Exception as e:
        logger.warning(f"[Grounder/OpenAlex] {e}")
        return []


def _search_semantic_scholar(query: str, limit: int = 5) -> list[dict]:
    """Search Semantic Scholar with contextual query."""
    from core.keys import semantic_scholar as get_key
    try:
        headers = {"User-Agent": "PipelineResearchBot/1.0"}
        key = get_key()
        if key:
            headers["x-api-key"] = key
        time.sleep(3.5)  # rate limit
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": limit,
                    "fields": "title,authors,year,abstract,externalIds,url"},
            headers=headers, timeout=20
        )
        resp.raise_for_status()
        results = []
        for p in resp.json().get("data", []):
            doi = p.get("externalIds", {}).get("DOI", "")
            results.append({
                "title":         p.get("title", ""),
                "authors":       [a.get("name","") for a in p.get("authors",[])[:3]],
                "year":          p.get("year"),
                "material_type": "paper",
                "source":        "semantic_scholar",
                "doi":           doi,
                "abstract":      (p.get("abstract") or "")[:800],
                "active_link":   p.get("url","") or (f"https://doi.org/{doi}" if doi else ""),
            })
        return results
    except Exception as e:
        logger.warning(f"[Grounder/SemanticScholar] {e}")
        return []


def _search_google_books(query: str, limit: int = 5) -> list[dict]:
    """Search Google Books API for foundational books."""
    from core.keys import get as get_key
    try:
        api_key = get_key("GOOGLE_BOOKS_API_KEY")
        params = {"q": query, "maxResults": limit, "orderBy": "relevance",
                  "printType": "books", "langRestrict": "en"}
        if api_key:
            params["key"] = api_key
        resp = requests.get("https://www.googleapis.com/books/v1/volumes",
                            params=params, timeout=15,
                            headers={"User-Agent": "PipelineResearchBot/1.0"})
        resp.raise_for_status()
        results = []
        for item in resp.json().get("items", [])[:limit]:
            info = item.get("volumeInfo", {})
            isbn = ""
            for id_obj in info.get("industryIdentifiers", []):
                if id_obj.get("type") in ("ISBN_13", "ISBN_10"):
                    isbn = id_obj.get("identifier", "")
                    break
            year = None
            pub_date = info.get("publishedDate", "")
            if pub_date and len(pub_date) >= 4:
                year = int(pub_date[:4]) if pub_date[:4].isdigit() else None
            results.append({
                "title":         info.get("title", ""),
                "authors":       info.get("authors", [])[:3],
                "year":          year,
                "material_type": "book",
                "source":        "google_books",
                "doi":           "",
                "isbn":          isbn,
                "abstract":      (info.get("description") or "")[:800],
                "active_link":   info.get("canonicalVolumeLink", "")
                                 or f"https://books.google.com/books?id={item.get('id','')}",
            })
        return results
    except Exception as e:
        logger.warning(f"[Grounder/GoogleBooks] {e}")
        return []


def _search_open_library(query: str, limit: int = 5) -> list[dict]:
    """Search Open Library for foundational books — no key needed."""
    try:
        time.sleep(1.0)  # polite
        resp = requests.get("https://openlibrary.org/search.json",
                            params={"q": query, "limit": limit,
                                    "fields": "title,author_name,first_publish_year,isbn,key,subject"},
                            timeout=15,
                            headers={"User-Agent": "PipelineResearchBot/1.0 (pipeline@research.local)"})
        resp.raise_for_status()
        results = []
        for doc in resp.json().get("docs", [])[:limit]:
            key  = doc.get("key", "")
            link = f"https://openlibrary.org{key}" if key else ""
            isbn_list = doc.get("isbn", [])
            isbn = isbn_list[0] if isbn_list else ""
            results.append({
                "title":         doc.get("title", ""),
                "authors":       doc.get("author_name", [])[:3],
                "year":          doc.get("first_publish_year"),
                "material_type": "book",
                "source":        "open_library",
                "doi":           "",
                "isbn":          isbn,
                "abstract":      "",
                "active_link":   link,
            })
        return results
    except Exception as e:
        logger.warning(f"[Grounder/OpenLibrary] {e}")
        return []


def _search_web(query: str) -> list[dict]:
    """
    Use Claude's web search tool via Anthropic API to get broader coverage.
    Returns results as pseudo-sources for the synthesis prompt.
    """
    from core.keys import anthropic as get_key
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "") or get_key()
    if not api_key:
        logger.warning("[Grounder/WebSearch] No Anthropic key — skipping web search")
        return []
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{
                "role": "user",
                "content": (
                    f"Search for foundational academic sources on: {query}\n"
                    f"List the most important books, papers, and authors. "
                    f"Include publication years and authors where known."
                )
            }]
        )
        # Extract text from response
        text_parts = [b.text for b in response.content if hasattr(b, "text") and b.text]
        full_text  = "\n".join(text_parts)
        if not full_text:
            return []
        # Return as a single web-search result entry for the synthesis prompt
        return [{"title": f"Web search: {query}", "source": "web_search",
                 "material_type": "web", "abstract": full_text[:1500],
                 "authors": [], "year": None, "doi": "", "active_link": ""}]
    except Exception as e:
        logger.warning(f"[Grounder/WebSearch] {e}")
        return []


def _search_consensus(query: str) -> list[dict]:
    """
    Semantic search via Consensus MCP — 200M+ peer-reviewed papers.
    Particularly valuable for Grounder because Consensus finds conceptually
    relevant seminal works even when exact keyword terms don't match.
    Falls back silently if not authenticated or unavailable.
    """
    try:
        from core.consensus_mcp import search_consensus
        results = search_consensus(query)
        out = []
        for r in results:
            out.append({
                "title":         r.get("title", ""),
                "authors":       r.get("authors", []),
                "year":          r.get("year"),
                "source_name":   "consensus",
                "doi":           r.get("doi", ""),
                "abstract":      r.get("abstract", ""),
                "active_link":   r.get("active_link", ""),
                "cited_by":      r.get("cited_by", 0),
                "material_type": "paper",
                "link_status":   "active",
            })
        return out
    except Exception as e:
        logger.warning(f"[Grounder/Consensus] {e}")
        return []


# ---------------------------------------------------------------------------
# Link verification
# ---------------------------------------------------------------------------

def _verify_link(url: str) -> str:
    if not url:
        return "dead"
    try:
        resp = requests.head(url, timeout=8, allow_redirects=True,
                             headers={"User-Agent": "PipelineResearchBot/1.0"})
        return "active" if resp.status_code < 400 else "dead"
    except Exception:
        return "dead"


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(context: str, run_id: str, **kwargs):
    logger.info(f"[Grounder] Starting for run {run_id}")

    problem = ""
    if "PROBLEM:" in context:
        problem = context.split("PROBLEM:")[1].split("\n\n")[0].strip()

    # Load agent source config
    _config = load_config()
    _allowed = set(_config.get("agent_sources", {}).get("grounder",
        ["openalex", "semantic_scholar", "consensus",
         "google_books", "open_library", "web"]))
    def _src_on(name: str) -> bool:
        return name in _allowed

    # Initialize argument tree
    from core.argument_tree import TreeBuilder
    tree = TreeBuilder(run_id)
    root_id = tree.create_root(problem)
    print(f"  [Grounder] Tree root created: {root_id}")

    # -----------------------------------------------------------------------
    # Step 1 — Decompose problem into sub-questions
    # -----------------------------------------------------------------------
    print("  [Grounder] Step 1 — decomposing problem into sub-questions...")
    try:
        decomp_resp = llm.call(
            f"Research problem to decompose:\n\n{problem}",
            DECOMPOSE_SYSTEM, agent_name="grounder"
        )
        decomp_clean = re.sub(r"```(?:json)?|```", "", decomp_resp).strip()
        decomp_data  = json.loads(decomp_clean)
    except Exception as e:
        logger.warning(f"[Grounder] Decomposition failed: {e} — using problem as single question")
        decomp_data = {
            "sub_questions": [{"id": "Q1", "question": problem,
                                "level": "positional", "rationale": "original problem"}],
            "decomposition_logic": "Decomposition failed — using original problem"
        }

    sub_questions = decomp_data.get("sub_questions", [])
    print(f"  [Grounder] {len(sub_questions)} sub-questions generated")

    # Add sub-questions as tree nodes
    q_node_map: dict[str, str] = {}  # maps Q1, Q2... → tree node_id
    for sq in sub_questions:
        q_id = sq.get("id", "?")
        node_id = tree.add_question(
            root_id, sq.get("question", ""),
            question_level=sq.get("level", "foundational"),
            agent="grounder",
        )
        q_node_map[q_id] = node_id
        print(f"    {q_id} [{sq.get('level','')}] {sq.get('question','')[:60]}")

    # -----------------------------------------------------------------------
    # Step 2 — Generate queries per sub-question and search all sources
    #          For each result, save to DB AND add to tree
    # -----------------------------------------------------------------------
    all_sources: list[dict] = []
    source_texts: list[str] = []

    for sq in sub_questions:
        q_text = sq.get("question", "")
        q_id   = sq.get("id", "?")
        q_node = q_node_map.get(q_id, root_id)
        print(f"  [Grounder] Querying sources for {q_id}: {q_text[:60]}...")

        # Generate contextual queries
        try:
            qgen_resp = llm.call(
                f"Sub-question: {q_text}\nProblem context: {problem}",
                QUERY_GEN_SYSTEM, agent_name="social"
            )
            qgen_clean = re.sub(r"```(?:json)?|```", "", qgen_resp).strip()
            queries    = json.loads(qgen_clean)
        except Exception as e:
            logger.warning(f"[Grounder] Query gen failed for {q_id}: {e}")
            words = [w for w in q_text.lower().split()
                     if len(w) > 4 and w not in {"what","does","have","that","this","with","from","into"}]
            queries = {
                "paper_query": " ".join(words[:3]),
                "book_query":  " ".join(words[:2]) + " history",
                "web_query":   " ".join(words[:4])
            }

        paper_query = queries.get("paper_query", "")
        book_query  = queries.get("book_query", "")
        web_query   = queries.get("web_query", "")

        print(f"    Papers: '{paper_query}' | Books: '{book_query}' | Web: '{web_query}'")

        def _process_results(results: list[dict], source_label: str,
                             evidence_type: str = "paper"):
            """Save each result to all_sources and build tree nodes."""
            for r in results:
                all_sources.append(r)
                title = r.get('title', '')
                authors = r.get('authors', [])
                year = r.get('year', '?')
                abstract = r.get('abstract', '')[:300]
                author_str = ', '.join(authors[:2])

                source_texts.append(
                    f"[{q_id}/{source_label}] {title} ({year}) "
                    f"— {author_str} | {abstract}"
                )

                # Generate a temporary source_id for tree linkage
                # (will be replaced with real DB source_id after synthesis)
                temp_src_id = generate_id("TSRC")
                r['_temp_src_id'] = temp_src_id
                r['_question_id'] = q_id
                r['_evidence_type'] = evidence_type

                # Add claim + evidence to tree under this question
                claim_text = (
                    f"{title} ({author_str}, {year}): "
                    f"{abstract[:150]}"
                )
                claim_id = tree.add_claim(
                    q_node, claim_text,
                    confidence=0.5,  # initial — refined in synthesis
                    source_ids=[temp_src_id],
                    agent="grounder",
                )
                tree.add_evidence(
                    claim_id, temp_src_id,
                    evidence_type=evidence_type,
                    relationship="supports",
                    snippet=abstract[:300],
                    agent="grounder",
                    metadata={
                        "title": title,
                        "authors": authors[:3],
                        "year": year,
                        "source_name": r.get('source_name', source_label.lower()),
                    },
                )

        # Academic papers — OpenAlex
        if paper_query and _src_on("openalex"):
            results = _search_openalex(paper_query, limit=4)
            time.sleep(0.2)
            _process_results(results, "OpenAlex", "paper")

        # Academic papers — Semantic Scholar
        if paper_query and _src_on("semantic_scholar"):
            results = _search_semantic_scholar(paper_query, limit=3)
            _process_results(results, "S2", "paper")

        # Academic papers — Consensus
        if paper_query and _src_on("consensus"):
            results = _search_consensus(paper_query)
            _process_results(results, "Consensus", "paper")

        # Books — Google Books
        if book_query and _src_on("google_books"):
            results = _search_google_books(book_query, limit=3)
            time.sleep(0.5)
            _process_results(results, "GoogleBooks", "book")

        # Books — Open Library
        if book_query and _src_on("open_library"):
            results = _search_open_library(book_query, limit=3)
            _process_results(results, "OpenLibrary", "book")

        # Web search — broader coverage
        if web_query and _src_on("web"):
            results = _search_web(web_query)
            _process_results(results, "WebSearch", "other")

    print(f"  [Grounder] {len(all_sources)} total sources gathered across {len(sub_questions)} sub-questions")
    tree_stats = tree.get_stats()
    print(f"  [Grounder] Tree: {tree_stats['total_nodes']} nodes, {tree_stats['unique_sources']} sources")

    # -----------------------------------------------------------------------
    # Step 3 — LLM synthesis from all gathered material
    # -----------------------------------------------------------------------
    print("  [Grounder] Step 3 — synthesizing intellectual foundations...")

    synthesis_prompt = f"""{context}

---
PROBLEM DECOMPOSITION:
{decomp_data.get('decomposition_logic','')}

Sub-questions explored:
{chr(10).join(f"  {sq['id']}: {sq['question']}" for sq in sub_questions)}

---
GATHERED SOURCES ({len(all_sources)} total):

{chr(10).join(source_texts[:80])}

---
Using the above sources, synthesize the intellectual foundations of the problem.
Prioritize genuinely old, foundational works. Include books alongside papers.
For each seminal work, use the exact title and author from the sources above where possible."""

    try:
        response = llm.call(synthesis_prompt, SYNTHESIS_SYSTEM, agent_name="grounder")
    except Exception as e:
        logger.error(f"[Grounder] Synthesis LLM call failed: {e}")
        raise

    try:
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        data  = json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end   = clean.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(clean[start:end+1])
            except Exception:
                data = {}
        else:
            data = {}
        if not data:
            logger.warning("[Grounder] Synthesis JSON parse failed — partial result")
            data = {
                "themes_extracted": [], "seminal_works": [],
                "intellectual_genealogy": response[:3000],
                "fundamental_whys": "", "original_definitions": [],
                "intersection_points": [], "proposed_new_themes": [],
                "assumptions_flagged": []
            }

    # -----------------------------------------------------------------------
    # Step 4 — Save seminal works to database + update tree with real source_ids
    # -----------------------------------------------------------------------
    saved = 0
    for work in data.get("seminal_works", []):
        if not work.get("title"):
            continue
        link_status = _verify_link(work.get("active_link", ""))
        source_id = generate_id("SEM")
        ok = db.upsert_source({
            "source_id":         source_id,
            "title":             work.get("title", ""),
            "authors":           work.get("authors", []),
            "year":              work.get("year"),
            "source_name":       work.get("source", "grounder"),
            "doi":               work.get("doi", ""),
            "abstract":          work.get("abstract", ""),
            "active_link":       work.get("active_link", ""),
            "theme_tags":        work.get("theme_tags", []),
            "type":              "seminal",
            "seminal_reason":    work.get("seminal_reason", ""),
            "intersection_tags": work.get("intersection_tags", []),
            "added_by":          "Grounder",
            "date_collected":    datetime.now(timezone.utc).isoformat(),
            "last_checked":      datetime.now(timezone.utc).isoformat(),
            "link_status":       link_status,
            "run_id":            run_id,
        })
        if ok:
            saved += 1
        time.sleep(0.05)

    # Save proposed themes to seminal bank
    for proposal in data.get("proposed_new_themes", []):
        if not proposal.get("theme_id"):
            continue
        db.insert_seminal_proposal({
            "bank_id":            generate_id("BANK"),
            "proposed_theme":     proposal.get("theme_id", ""),
            "problem_origin":     problem,
            "reason":             proposal.get("reason", ""),
            "suggested_keywords": proposal.get("suggested_keywords", []),
            "suggested_sources":  proposal.get("suggested_sources", []),
        })

    # -----------------------------------------------------------------------
    # Step 5 — Save decomposition + foundations document
    # -----------------------------------------------------------------------
    _save_doc(run_id, problem, data, sub_questions, decomp_data.get("decomposition_logic",""))

    # Print tree summary
    final_stats = tree.get_stats()
    gaps = tree.find_gaps()
    n_books  = sum(1 for s in data.get("seminal_works",[]) if s.get("material_type") == "book")
    n_papers = sum(1 for s in data.get("seminal_works",[]) if s.get("material_type") != "book")
    print(f"  [Grounder] {saved} seminal works saved ({n_papers} papers, {n_books} books) | "
          f"{len(data.get('themes_extracted',[]))} themes | "
          f"{len(data.get('proposed_new_themes',[]))} proposals to seminal bank")
    print(f"  [Grounder] Tree: {final_stats['total_nodes']} nodes | "
          f"{len(gaps)} gaps (unanswered questions / unsupported claims)")
    tree.close()
    logger.info("[Grounder] Complete")


# ---------------------------------------------------------------------------
# Document writer — fixed to actually populate all sections
# ---------------------------------------------------------------------------

def _save_doc(run_id: str, problem: str, data: dict,
              sub_questions: list = None, decomp_logic: str = ""):
    path = Path(__file__).parent.parent / "artifacts" / f"{run_id}_grounder_foundations.md"
    path.parent.mkdir(exist_ok=True)

    lines = [
        "# Foundations Document — Grounder",
        f"**Run:** {run_id}",
        f"**Problem:** {problem}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "", "---", "",
    ]

    # Problem decomposition
    if sub_questions:
        lines += ["## Problem Decomposition", "", decomp_logic, ""]
        for sq in sub_questions:
            lines.append(f"- **{sq.get('id','?')}** [{sq.get('level','')}] {sq.get('question','')}")
            if sq.get("rationale"):
                lines.append(f"  *{sq['rationale']}*")
        lines.append("")

    # Themes
    themes = data.get("themes_extracted", [])
    lines += ["## Themes Extracted", ""]
    if themes:
        for t in themes:
            lines.append(f"- **{t.get('theme','')}**: {t.get('description','')}")
    else:
        lines.append("*(none extracted)*")
    lines.append("")

    # Fundamental Whys
    lines += ["## Fundamental Whys", ""]
    fw = data.get("fundamental_whys", "")
    lines.append(fw if fw else "*(not produced)*")
    lines.append("")

    # Intellectual Genealogy
    lines += ["## Intellectual Genealogy", ""]
    ig = data.get("intellectual_genealogy", "")
    lines.append(ig if ig else "*(not produced)*")
    lines.append("")

    # Original Definitions
    defs = data.get("original_definitions", [])
    lines += ["## Original Definitions", ""]
    if defs:
        for d in defs:
            lines.append(
                f"- **{d.get('concept','')}** "
                f"({d.get('defined_by','')}, {d.get('year','')}): "
                f"{d.get('definition','')}"
            )
    else:
        lines.append("*(none extracted)*")
    lines.append("")

    # Seminal Works — split books and papers
    works = data.get("seminal_works", [])
    lines += ["## Seminal Works", ""]
    books  = [w for w in works if w.get("material_type") == "book"]
    papers = [w for w in works if w.get("material_type") != "book"]

    if papers:
        lines += ["### Papers & Articles", ""]
        for w in sorted(papers, key=lambda x: x.get("year") or 9999):
            authors_str = ", ".join(w.get("authors", [])[:3])
            lines.append(
                f"- **[{w.get('year','n.d.')}] {w.get('title','')}**  "
                f"— {authors_str}"
            )
            lines.append(f"  *{w.get('seminal_reason','')}*")
            if w.get("active_link"):
                lines.append(f"  Link: {w['active_link']}")
            lines.append("")

    if books:
        lines += ["### Books", ""]
        for w in sorted(books, key=lambda x: x.get("year") or 9999):
            authors_str = ", ".join(w.get("authors", [])[:3])
            lines.append(
                f"- **[{w.get('year','n.d.')}] {w.get('title','')}**  "
                f"— {authors_str}"
            )
            lines.append(f"  *{w.get('seminal_reason','')}*")
            if w.get("active_link"):
                lines.append(f"  Link: {w['active_link']}")
            lines.append("")

    if not works:
        lines.append("*(none found)*")
        lines.append("")

    # Intersection Points
    intersections = data.get("intersection_points", [])
    lines += ["## Intersection Points", ""]
    if intersections:
        for i in intersections:
            themes_str = " × ".join(i.get("themes", []))
            lines.append(f"- **{themes_str}**: {i.get('description','')}")
    else:
        lines.append("*(none identified)*")
    lines.append("")

    # Assumptions Flagged
    assumptions = data.get("assumptions_flagged", [])
    if assumptions:
        lines += ["## Assumptions Flagged", ""]
        for a in assumptions:
            lines.append(f"- **{a.get('assumption','')}**: {a.get('note','')}")
        lines.append("")

    # Proposed New Themes
    proposals = data.get("proposed_new_themes", [])
    if proposals:
        lines += ["## Proposed New Themes (→ Seminal Bank)", ""]
        for p in proposals:
            lines.append(f"- **{p.get('theme_id','')}** ({p.get('label','')}): {p.get('reason','')}")
        lines.append("")

    path.write_text("\n".join(lines))
    logger.info(f"[Grounder] Foundations document saved: {path}")
