"""
Concept Mapper
--------------
Pre-processing module that sits before Social's feed.
Translates a raw research problem into its full conceptual territory
before any keyword or theme matching happens.

Three layers:
  1. ConceptNet API  — semantic expansion of raw terms (cached locally)
  2. Concept clusters — curated disciplinary translation map
  3. LLM synthesis   — catches what static map missed, produces final theme list

Results cached in SQLite: concept_expansions and concept_cache tables.
"""

import re
import json
import time
import hashlib
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from core import llm
from core.utils import generate_id

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "db" / "pipeline.db"
CONCEPT_MAP_PATH = Path(__file__).parent.parent / "concept_map.json"

CONCEPTNET_DB_PATH = Path(__file__).parent.parent / "db" / "conceptnet.db"

# ---------------------------------------------------------------------------
# Database — extend pipeline.db with two new tables
# ---------------------------------------------------------------------------

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS concept_cache (
    cache_key      TEXT PRIMARY KEY,
    term           TEXT NOT NULL,
    relations      TEXT NOT NULL,   -- JSON array of {rel, target, weight}
    fetched_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS concept_expansions (
    expansion_id   TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL,
    problem        TEXT NOT NULL,
    raw_terms      TEXT NOT NULL,   -- JSON array
    expanded_concepts TEXT NOT NULL,-- JSON array of {concept, source, weight, cluster_ids}
    activated_clusters TEXT NOT NULL,-- JSON array of cluster_ids
    activated_disciplines TEXT NOT NULL,-- JSON array
    bridge_concepts TEXT NOT NULL,  -- JSON array
    final_themes   TEXT NOT NULL,   -- JSON array of theme_ids to activate
    llm_reasoning  TEXT,
    created_at     TEXT NOT NULL
);
"""

def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(CACHE_SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# ConceptNet local SQLite query
# ---------------------------------------------------------------------------

def _cache_key(term: str) -> str:
    return hashlib.md5(term.lower().strip().encode()).hexdigest()


def _conceptnet_available() -> bool:
    """Check if the local ConceptNet SQLite database exists and has data."""
    if not CONCEPTNET_DB_PATH.exists():
        return False
    try:
        cn_conn = sqlite3.connect(str(CONCEPTNET_DB_PATH))
        count = cn_conn.execute("SELECT COUNT(*) FROM edges LIMIT 1").fetchone()[0]
        cn_conn.close()
        return count > 0
    except Exception:
        return False


def _fetch_conceptnet(term: str, limit: int = 30) -> list[dict]:
    """
    Fetch ConceptNet relations for a term from local SQLite.
    Returns list of {rel, target, weight}.
    Checks pipeline.db cache first — queries conceptnet.db if missing.
    Falls back to empty list if conceptnet.db not available.
    """
    cache_key_val = _cache_key(term)
    conn = _get_conn()

    # Check pipeline.db cache
    row = conn.execute(
        "SELECT relations FROM concept_cache WHERE cache_key = ?", (cache_key_val,)
    ).fetchone()
    if row:
        conn.close()
        logger.debug(f"[ConceptMapper] Cache hit: {term}")
        return json.loads(row["relations"])

    # Check if local ConceptNet DB is available
    if not _conceptnet_available():
        logger.warning(
            f"[ConceptMapper] conceptnet.db not found at {CONCEPTNET_DB_PATH}. "
            f"Run: python3 tools/import_conceptnet.py --input /path/to/conceptnet-assertions-5.7.0.csv.gz"
        )
        conn.close()
        return []

    # Query local conceptnet.db
    logger.info(f"[ConceptMapper] Querying local ConceptNet DB: '{term}'")
    relations = []
    try:
        cn_conn = sqlite3.connect(str(CONCEPTNET_DB_PATH))
        cn_conn.row_factory = sqlite3.Row

        # Forward direction: term → target
        rows_fwd = cn_conn.execute(
            "SELECT relation, target, weight FROM edges WHERE term = ? ORDER BY weight DESC LIMIT ?",
            (term.lower(), limit)
        ).fetchall()

        # Reverse direction for symmetric relations
        rows_rev = cn_conn.execute(
            """SELECT relation, term as target, weight FROM edges
               WHERE target = ?
               AND relation IN ('/r/RelatedTo', '/r/SimilarTo', '/r/Synonym')
               ORDER BY weight DESC LIMIT ?""",
            (term.lower(), limit // 2)
        ).fetchall()

        cn_conn.close()

        seen = set()
        for r in list(rows_fwd) + list(rows_rev):
            t = r["target"]
            k = f"{r['relation']}:{t}"
            if k not in seen and t != term.lower():
                seen.add(k)
                relations.append({
                    "rel":    r["relation"],
                    "target": t,
                    "weight": round(r["weight"], 3)
                })

        relations.sort(key=lambda x: x["weight"], reverse=True)
        relations = relations[:limit]

    except Exception as e:
        logger.warning(f"[ConceptMapper] Local DB query failed for '{term}': {e}")
        conn.close()
        return []

    # Cache into pipeline.db
    conn.execute(
        "INSERT OR REPLACE INTO concept_cache (cache_key, term, relations, fetched_at) VALUES (?,?,?,?)",
        (cache_key_val, term, json.dumps(relations), datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()

    return relations


# ---------------------------------------------------------------------------
# Concept cluster map loader
# ---------------------------------------------------------------------------

def load_concept_map() -> dict:
    if not CONCEPT_MAP_PATH.exists():
        logger.warning(f"[ConceptMapper] concept_map.json not found at {CONCEPT_MAP_PATH}")
        return {"concept_clusters": []}
    with open(CONCEPT_MAP_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Core expansion logic
# ---------------------------------------------------------------------------

def _extract_raw_terms(problem: str) -> list[str]:
    """
    Extract meaningful terms from problem statement.
    Removes stopwords and very short tokens.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "can", "shall", "of", "in", "on",
        "at", "to", "for", "with", "by", "from", "about", "as", "into",
        "through", "during", "what", "where", "when", "why", "how", "which",
        "who", "that", "this", "these", "those", "it", "its", "and", "or",
        "but", "if", "not", "no", "nor", "so", "yet", "both", "either",
        "place", "role", "impact", "effect", "relation", "relationship",
        "between", "among", "within", "without", "there", "their", "they",
        "them", "than", "then", "now", "very", "just", "also", "more",
        "most", "such", "any", "all", "each", "every", "some"
    }

    # Clean and tokenize
    clean = re.sub(r"[^\w\s]", " ", problem.lower())
    tokens = clean.split()

    # Always keep important short terms
    keep_short = {"ai", "ml", "nlp", "sts", "dna", "rna", "llm"}

    # Filter
    terms = [t for t in tokens if (t not in stopwords and len(t) > 3) or t in keep_short]

    # Also extract bigrams (two-word phrases)
    words = problem.lower().split()
    bigrams = []
    for i in range(len(words) - 1):
        w1, w2 = re.sub(r"[^\w]","",words[i]), re.sub(r"[^\w]","",words[i+1])
        if w1 not in stopwords and w2 not in stopwords and len(w1) > 2 and len(w2) > 2:
            bigrams.append(f"{w1} {w2}")

    return list(dict.fromkeys(terms + bigrams))  # deduplicate, preserve order


def _match_clusters(
    concepts: list[str],
    concept_map: dict,
    threshold: float = 0.0
) -> tuple[list[str], list[str], list[str]]:
    """
    Match expanded concepts against cluster trigger_concepts.
    Returns (activated_cluster_ids, activated_disciplines, bridge_concepts).
    """
    clusters = concept_map.get("concept_clusters", [])
    activated_cluster_ids = set()
    activated_disciplines = set()
    bridge_concepts = set()

    concepts_lower = {c.lower() for c in concepts}

    for cluster in clusters:
        triggers = {t.lower() for t in cluster.get("trigger_concepts", [])}
        # Check overlap between expanded concepts and trigger concepts
        overlap = concepts_lower & triggers
        if overlap:
            activated_cluster_ids.add(cluster["cluster_id"])
            for d in cluster.get("disciplines", []):
                activated_disciplines.add(d)
            for b in cluster.get("bridge_concepts", []):
                bridge_concepts.add(b)

    return (
        list(activated_cluster_ids),
        list(activated_disciplines),
        list(bridge_concepts)
    )


def _disciplines_to_themes(disciplines: list[str], config: dict) -> list[str]:
    """
    Map activated disciplines to theme_ids in config.json.
    Returns theme_ids that are both activated AND present in config.
    """
    config_theme_ids = {t["theme_id"] for t in config.get("themes", [])}
    matched = []
    for d in disciplines:
        # Exact match
        if d in config_theme_ids:
            matched.append(d)
            continue
        # Fuzzy match — discipline is substring of theme_id or vice versa
        for tid in config_theme_ids:
            if d.lower() in tid.lower() or tid.lower() in d.lower():
                matched.append(tid)
                break
    return list(dict.fromkeys(matched))


# ---------------------------------------------------------------------------
# LLM synthesis layer
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """You are a conceptual knowledge mapper for a research pipeline.

Given a research problem and its automatically expanded concept network, your job is to:
1. Identify ALL relevant academic disciplines and research domains this problem touches
2. Identify bridge concepts that connect different domains
3. Suggest specific search themes that should be activated

Be comprehensive — include obvious AND non-obvious disciplines.
A problem about "AI in human life" should include not just AI/CS but also philosophy, sociology, anthropology, history of technology, ethics, political science, economics, psychology, etc.

Output ONLY valid JSON:
{
  "disciplines_identified": ["list of all relevant disciplines"],
  "bridge_concepts": ["key concepts that bridge multiple disciplines"],
  "suggested_themes": [
    {
      "theme_id": "snake_case_id matching config if possible",
      "label": "Human readable label",
      "relevance_reason": "one line why this theme is relevant to the problem"
    }
  ],
  "conceptual_translation": "2-3 sentences explaining how you translated this problem into its conceptual territory",
  "overlooked_angles": ["disciplines or angles that might be non-obvious but important"]
}"""


def _llm_synthesis(
    problem: str,
    raw_terms: list[str],
    expanded_concepts: list[dict],
    activated_clusters: list[str],
    bridge_concepts: list[str],
    config: dict
) -> dict:
    """LLM synthesis — catches what static map missed."""
    config_themes = [{"theme_id": t["theme_id"], "label": t.get("label","")}
                     for t in config.get("themes", [])]

    # Top expanded concepts by weight
    top_concepts = [c["concept"] for c in sorted(
        expanded_concepts, key=lambda x: x.get("weight", 0), reverse=True
    )[:30]]

    prompt = f"""Research problem: "{problem}"

Raw terms extracted: {', '.join(raw_terms[:15])}

Top expanded concepts (from ConceptNet): {', '.join(top_concepts)}

Activated conceptual clusters: {', '.join(activated_clusters)}

Bridge concepts identified: {', '.join(bridge_concepts[:20])}

Available themes in config: {json.dumps(config_themes, indent=2)}

Based on all of the above, identify the complete disciplinary territory of this problem.
Be especially thorough about non-obvious connections.
Match suggested themes to existing config theme_ids where possible."""

    try:
        response = llm.call(prompt, SYNTHESIS_SYSTEM, agent_name="social")
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        return json.loads(clean)
    except Exception as e:
        logger.warning(f"[ConceptMapper] LLM synthesis failed: {e}")
        return {
            "disciplines_identified": activated_clusters,
            "bridge_concepts": bridge_concepts,
            "suggested_themes": [],
            "conceptual_translation": "",
            "overlooked_angles": []
        }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def expand(problem: str, run_id: str, config: dict) -> dict:
    """
    Main entry point. Fully expands a problem into its conceptual territory.

    Returns:
    {
      "raw_terms": [...],
      "expanded_concepts": [...],
      "activated_clusters": [...],
      "activated_disciplines": [...],
      "bridge_concepts": [...],
      "final_themes": [...],   # theme_ids to activate in pipeline
      "llm_reasoning": "...",
      "overlooked_angles": [...]
    }
    """
    logger.info(f"[ConceptMapper] Expanding problem for run {run_id}")
    concept_map = load_concept_map()

    # Layer 1: Extract raw terms
    raw_terms = _extract_raw_terms(problem)
    logger.info(f"[ConceptMapper] Extracted {len(raw_terms)} raw terms: {raw_terms[:10]}")

    # Layer 1b: ConceptNet expansion
    all_concepts = set(raw_terms)
    expanded_concepts = []

    for term in raw_terms[:12]:  # Limit API calls — top 12 terms
        relations = _fetch_conceptnet(term, limit=25)
        for r in relations[:15]:  # Top 15 relations per term
            target = r["target"].lower()
            if len(target) > 2 and target not in all_concepts:
                all_concepts.add(target)
                expanded_concepts.append({
                    "concept":     r["target"],
                    "source_term": term,
                    "relation":    r["rel"],
                    "weight":      r["weight"],
                    "cluster_ids": []
                })

    logger.info(f"[ConceptMapper] Expanded to {len(all_concepts)} concepts via ConceptNet")

    # Layer 2: Cluster matching
    all_concept_list = list(all_concepts)
    activated_clusters, activated_disciplines, bridge_concepts = _match_clusters(
        all_concept_list, concept_map
    )
    logger.info(
        f"[ConceptMapper] Activated {len(activated_clusters)} clusters, "
        f"{len(activated_disciplines)} disciplines"
    )

    # Tag expanded concepts with their cluster IDs
    clusters = concept_map.get("concept_clusters", [])
    for ec in expanded_concepts:
        c_lower = ec["concept"].lower()
        for cluster in clusters:
            triggers = {t.lower() for t in cluster.get("trigger_concepts", [])}
            if c_lower in triggers:
                ec["cluster_ids"].append(cluster["cluster_id"])

    # Layer 3: LLM synthesis
    llm_result = _llm_synthesis(
        problem, raw_terms, expanded_concepts,
        activated_clusters, bridge_concepts, config
    )

    # Merge LLM disciplines with cluster disciplines
    all_disciplines = list(set(
        activated_disciplines +
        llm_result.get("disciplines_identified", [])
    ))
    all_bridges = list(set(
        bridge_concepts +
        llm_result.get("bridge_concepts", [])
    ))

    # Map to config theme_ids
    cluster_themes = _disciplines_to_themes(all_disciplines, config)

    # Add LLM-suggested themes that match config
    config_theme_ids = {t["theme_id"] for t in config.get("themes", [])}
    llm_themes = [
        s["theme_id"] for s in llm_result.get("suggested_themes", [])
        if s.get("theme_id") in config_theme_ids
    ]

    final_themes = list(dict.fromkeys(cluster_themes + llm_themes))

    # If still nothing matched — activate ALL themes (better broad than blind)
    if not final_themes:
        final_themes = [t["theme_id"] for t in config.get("themes", [])]
        logger.warning("[ConceptMapper] No themes matched — activating all themes")

    logger.info(f"[ConceptMapper] Final themes activated: {final_themes}")

    result = {
        "raw_terms":            raw_terms,
        "expanded_concepts":    expanded_concepts,
        "activated_clusters":   activated_clusters,
        "activated_disciplines": all_disciplines,
        "bridge_concepts":      all_bridges,
        "final_themes":         final_themes,
        "llm_reasoning":        llm_result.get("conceptual_translation", ""),
        "overlooked_angles":    llm_result.get("overlooked_angles", []),
        "llm_suggested_themes": llm_result.get("suggested_themes", [])
    }

    # Save to database
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO concept_expansions
           (expansion_id, run_id, problem, raw_terms, expanded_concepts,
            activated_clusters, activated_disciplines, bridge_concepts,
            final_themes, llm_reasoning, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            generate_id("EXP"), run_id, problem,
            json.dumps(raw_terms),
            json.dumps(expanded_concepts),
            json.dumps(activated_clusters),
            json.dumps(all_disciplines),
            json.dumps(all_bridges),
            json.dumps(final_themes),
            result["llm_reasoning"],
            datetime.now(timezone.utc).isoformat()
        )
    )
    conn.commit()
    conn.close()

    return result


def get_expansion(run_id: str) -> Optional[dict]:
    """Retrieve a cached expansion for a run."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM concept_expansions WHERE run_id = ? ORDER BY created_at DESC LIMIT 1",
        (run_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def print_expansion_report(result: dict):
    """Print a readable expansion report to terminal."""
    print(f"\n{'─'*60}")
    print(f"  CONCEPT MAPPER — Expansion Report")
    print(f"{'─'*60}")
    print(f"  Raw terms:     {', '.join(result['raw_terms'][:10])}")
    print(f"  ConceptNet:    {len(result['expanded_concepts'])} concepts expanded")
    print(f"  Clusters:      {', '.join(result['activated_clusters'][:8])}")
    print(f"  Disciplines:   {len(result['activated_disciplines'])} identified")
    print(f"  Bridge concepts: {', '.join(result['bridge_concepts'][:8])}")
    print(f"  Final themes:  {', '.join(result['final_themes'])}")
    if result.get("llm_reasoning"):
        print(f"\n  Translation:   {result['llm_reasoning'][:200]}...")
    if result.get("overlooked_angles"):
        print(f"\n  Non-obvious angles:")
        for a in result["overlooked_angles"][:5]:
            print(f"    - {a}")
    print(f"{'─'*60}\n")
