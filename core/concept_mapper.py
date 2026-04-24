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

    A cluster activates only if it has MEANINGFUL overlap with the problem:
    - At least 2 matching concepts, OR
    - 1 match that is a substantive concept (longer than 5 chars and not a stopword)
    This prevents single generic word matches (like 'influence' or 'production')
    from activating entire disciplinary clusters.
    """
    # Words that should never alone activate a cluster
    WEAK_TRIGGERS = {
        'influence', 'production', 'account', 'dimensions', 'units',
        'taking', 'impact', 'effect', 'role', 'process', 'function',
        'relationship', 'interaction', 'analysis', 'study', 'research',
        'approach', 'method', 'model', 'system', 'structure', 'form',
        'type', 'level', 'factor', 'aspect', 'element', 'component',
        'change', 'development', 'data', 'results', 'context', 'case',
        'work', 'field', 'area', 'domain', 'topic', 'issue', 'problem',
        'affect', 'effects', 'cause', 'causes', 'gene', 'genes',
        'nature', 'human', 'life', 'world', 'time', 'place', 'space',
        'general', 'global', 'local', 'social', 'cultural', 'political',
        'economic', 'natural', 'physical', 'technical', 'modern',
        'practice', 'theory', 'concept', 'idea', 'question', 'answer',
        'performance', 'scale', 'content', 'platform', 'network', 'movement',
        'quality', 'measure', 'balance', 'value', 'community',
        'thought', 'thinking', 'knowledge', 'understanding', 'experience',
        'awareness', 'sense', 'feeling', 'mind', 'brain', 'body',
    }

    clusters = concept_map.get("concept_clusters", [])
    activated_cluster_ids = set()
    activated_disciplines = set()
    bridge_concepts = set()

    concepts_lower = {c.lower() for c in concepts}

    for cluster in clusters:
        triggers = {t.lower() for t in cluster.get("trigger_concepts", [])}
        overlap  = concepts_lower & triggers

        if not overlap:
            continue

        # Check if overlap is meaningful:
        # 1. Two or more matching concepts, OR
        # 2. At least one match that is substantive (not a weak generic word)
        substantive_matches = [m for m in overlap if m not in WEAK_TRIGGERS]
        weak_only_matches   = [m for m in overlap if m in WEAK_TRIGGERS]

        is_meaningful = (
            len(overlap) >= 2 or          # multiple matches
            len(substantive_matches) >= 1  # at least one non-generic match
        )

        if not is_meaningful:
            continue

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
    Uses explicit override map first, then fuzzy matching as fallback.
    Explicit map prevents 'neuroscience' from matching unrelated themes
    when its cluster was activated by a marginal generic-word match.
    """
    config_theme_ids = {t["theme_id"] for t in config.get("themes", [])}

    # Explicit discipline → theme_id overrides (highest priority)
    # Covers all 122 disciplines referenced in concept_map.json
    EXPLICIT_MAP = {
        # Agriculture & environment
        "agriculture":              "agriculture_food_systems",
        "ecology":                  "biology_life_sciences",
        "environmental_science":    "environmental_studies",
        "environmental_philosophy": "environmental_studies",
        "environmental_ethics":     "environmental_studies",
        "political_ecology":        "environmental_studies",
        # Politics & governance
        "political_philosophy":     "political_science",
        "public_policy":            "political_science",
        "public_administration":    "political_science",
        "international_relations":  "political_science",
        "political_economy":        "economics",
        "political_science":        "political_science",
        # Economics & labor
        "labor_studies":            "economics",
        "behavioral_economics":     "economics",
        "development_studies":      "development_studies",
        # History
        "historiography":           "history",
        "intellectual_history":     "history",
        "history_of_science":       "history",
        "philosophy_of_history":    "history",
        "memory_studies":           "history",
        "area_studies":             "history",
        # Sociology & social
        "critical_theory":          "sociology",
        "cultural_studies":         "anthropology",
        "gender_studies":           "sociology",
        "race_studies":             "sociology",
        "postcolonial_studies":     "sociology",
        "urban_studies":            "sociology",
        "demography":               "sociology",
        "sociology_of_knowledge":   "sociology",
        "sociology_of_law":         "law",
        "sociology_of_health":      "psychology",
        "sociology_of_education":   "education_science",
        "sociology_of_art":         "anthropology",
        "criminology":              "sociology",
        "queer_theory":             "sociology",
        "feminist_theory":          "sociology",
        "disability_studies":       "sociology",
        # Psychology
        "social_psychology":        "psychology",
        "developmental_psychology": "psychology",
        "clinical_psychology":      "psychology",
        "evolutionary_psychology":  "psychology",
        "psychiatry":               "psychology",
        "psychoanalysis":           "psychology",
        "behavioral_science":       "psychology",
        # Media & communication
        "media_studies":            "media_communication",
        "communication_studies":    "media_communication",
        "information_science":      "media_communication",
        "journalism":               "media_communication",
        "film_studies":             "media_communication",
        # Philosophy
        "philosophy":               "philosophy_general",
        "metaphysics":              "philosophy_general",
        "epistemology":             "philosophy_general",
        "phenomenology":            "philosophy_general",
        "ethics":                   "ethics",
        "applied_ethics":           "ethics",
        "bioethics":                "ethics",
        "metaethics":               "ethics",
        "moral_philosophy":         "ethics",
        "business_ethics":          "ethics",
        "engineering_ethics":       "ethics",
        "AI_ethics":                "ethics",
        "philosophy_of_science":    "philosophy_general",
        "philosophy_of_technology": "science_technology_studies",
        "philosophy_of_education":  "education_science",
        "philosophy_of_medicine":   "medicine_clinical",
        "philosophy_of_biology":    "biology_life_sciences",
        "philosophy_of_religion":   "religious_studies",
        "philosophy_of_mathematics": "mathematics_logic",
        # Sciences
        "biology":                  "biology_life_sciences",
        "evolutionary_biology":     "biology_life_sciences",
        "genetics":                 "biomedical_research",
        "biochemistry":             "biomedical_research",
        "molecular_biology":        "biomedical_research",
        "medicine":                 "medicine_clinical",
        "clinical_psychology":      "medicine_clinical",
        "psychiatry":               "medicine_clinical",
        "nursing_science":          "medicine_clinical",
        "health_informatics":       "medicine_clinical",
        "epidemiology":             "public_health",
        "public_health":            "public_health",
        "sociology_of_health":      "public_health",
        "health_policy":            "public_health",
        "biostatistics":            "public_health",
        "environmental_health":     "public_health",
        "pharmacology":             "biomedical_research",
        "medicinal_chemistry":      "biomedical_research",
        "biomedical_research":      "biomedical_research",
        "translational_medicine":   "biomedical_research",
        "physics":                  "complexity_systems",
        # Technology & computation
        "AI":                       "artificial_intelligence",
        "NLP":                      "artificial_intelligence",
        "AI_law":                   "law",
        "computer_science":         "artificial_intelligence",
        "theoretical_computer_science": "mathematics_logic",
        "human_computer_interaction": "science_technology_studies",
        "educational_technology":   "education_science",
        "cybernetics":              "complexity_systems",
        "systems_theory":           "complexity_systems",
        "complexity_science":       "complexity_systems",
        # Other
        "jurisprudence":            "law",
        "comparative_religion":     "religious_studies",
        "theology":                 "religious_studies",
        "contemplative_studies":    "religious_studies",
        "archaeology":              "anthropology",
        "folklore":                 "anthropology",
        "geography":                "development_studies",
        "rhetoric":                 "philosophy_of_language",
        "semiotics":                "linguistics",
        "cognitive_linguistics":    "linguistics",
        "literary_theory":          "philosophy_of_language",
        "aesthetics":               "philosophy_general",
        "art_history":              "anthropology",
        "musicology":               "anthropology",
        "architecture":             "science_technology_studies",
        "statistics":               "mathematics_logic",
        "logic":                    "mathematics_logic",
        "foundations_of_mathematics": "mathematics_logic",
        "mathematics":              "mathematics_logic",
        "learning_sciences":        "education_science",
    }

    matched = set()
    for d in disciplines:
        d_lower = d.lower()
        if d_lower in EXPLICIT_MAP:
            tid = EXPLICIT_MAP[d_lower]
            if tid in config_theme_ids:
                matched.add(tid)
            continue
        if d_lower in config_theme_ids:
            matched.add(d_lower)
            continue
        for tid in config_theme_ids:
            if d_lower in tid.lower() or tid.lower() in d_lower:
                matched.add(tid)
                break

    return list(dict.fromkeys(matched))


# ---------------------------------------------------------------------------
# LLM synthesis layer
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """You are a disciplinary relevance filter for a research pipeline.

Your job is NOT to brainstorm every possible connection.
Your job is to identify the CORE disciplines that a researcher would genuinely search when studying this specific problem, AND to prune disciplines that were activated by incidental keyword matches but are topically off-target.

You will be shown:
  - The research problem
  - A candidate theme list already proposed by automated cluster matching
  - The full set of themes available in the pipeline

You must decide, for each candidate theme, whether a researcher working on THIS problem would genuinely search that discipline's literature. Remove the ones they would not.

Rules for EXCLUSION (prune from candidates):
- If the candidate theme was activated by an incidental word match (e.g. "evidence" activating epistemology, "existence" activating metaphysics, "university" activating education-psychology) and the actual problem is not about that discipline's subject matter, EXCLUDE it.
- philosophy_of_mind, philosophy_of_language, philosophy_general, cognitive_science, psychology, neuroscience, linguistics, religious_studies, mathematics_logic — these should be EXCLUDED unless the problem is explicitly about cognition, mind, language-as-system, psychology, neural processes, linguistic theory, religion, or mathematical/logical foundations.
- science_technology_studies — EXCLUDE unless the problem is explicitly about the social study of science, technology, or scientific practice.
- media_communication — EXCLUDE unless the problem is about media, journalism, communication systems, or information platforms.
- Be aggressive about exclusion. False positives from the static matcher are common; leaving irrelevant themes in poisons retrieval.

Rules for INCLUSION (additions beyond candidates):
- Only add a theme if the static matcher clearly missed something a researcher would definitely search.
- Each addition must have a specific concrete reason tied to the actual problem, not generic language.
- If unsure, do not add.

Rules for REASONS:
- Each exclusion and addition must have a reason of at least 20 characters naming WHY for this specific problem.
- Generic reasons like "relevant" or "connected" are not acceptable.

Output ONLY valid JSON:
{
  "conceptual_translation": "1-2 sentences: what is this problem fundamentally about?",
  "themes_to_exclude": [
    {
      "theme_id": "snake_case_id from the candidate list",
      "reason": "specific reason this discipline is NOT what a researcher on THIS problem would search"
    }
  ],
  "themes_to_add": [
    {
      "theme_id": "snake_case_id matching an available theme",
      "label": "Human readable label",
      "relevance_reason": "specific concrete reason a researcher on THIS problem would search this discipline"
    }
  ],
  "disciplines_identified": ["core disciplines actually relevant to this specific problem"],
  "bridge_concepts": ["key concepts connecting the core disciplines"]
}"""


def _llm_synthesis(
    problem: str,
    raw_terms: list[str],
    expanded_concepts: list[dict],
    activated_clusters: list[str],
    activated_disciplines: list[str],
    bridge_concepts: list[str],
    candidate_themes: list[str],
    config: dict
) -> dict:
    """LLM synthesis — prunes false positives and catches what static map missed.

    candidate_themes is the list of theme_ids that cluster matching proposes to
    activate. The LLM is asked to exclude any of these that are topically off
    for the specific problem, and to add any genuinely missing ones.
    """
    config_themes = [{"theme_id": t["theme_id"], "label": t.get("label","")}
                     for t in config.get("themes", [])]

    # Top expanded concepts by weight (shown to LLM for semantic context)
    top_concepts = [c["concept"] for c in sorted(
        expanded_concepts, key=lambda x: x.get("weight", 0), reverse=True
    )[:30]]

    prompt = f"""Research problem: "{problem}"

Raw terms extracted: {', '.join(raw_terms[:15])}

Conceptual clusters activated by automated matching:
{', '.join(activated_clusters)}

Candidate themes proposed for activation (derived from clusters above):
{json.dumps(candidate_themes, indent=2)}

All themes available in the pipeline config:
{json.dumps(config_themes, indent=2)}

Task:
1. Produce a conceptual translation of what this problem is fundamentally about.
2. Review the CANDIDATE THEMES list above. For each candidate that a researcher
   studying this specific problem would NOT actually search, list it under
   "themes_to_exclude" with a specific reason. Be aggressive about exclusion —
   the automated matcher over-activates on incidental word matches (e.g.
   "evidence" -> epistemology -> philosophy_of_mind; "existence" -> metaphysics
   -> philosophy_general).
3. If there are themes in the available list that the candidates clearly miss
   AND a researcher on this problem would genuinely search them, add them
   under "themes_to_add" with a specific reason.
4. Be conservative on additions and aggressive on exclusions.

Return ONLY the JSON object specified in the system instructions."""

    try:
        response = llm.call(prompt, SYNTHESIS_SYSTEM, agent_name="social")
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        return json.loads(clean)
    except Exception as e:
        logger.warning(f"[ConceptMapper] LLM synthesis failed: {e}")
        return {
            "conceptual_translation": "",
            "themes_to_exclude": [],
            "themes_to_add": [],
            "disciplines_identified": activated_clusters,
            "bridge_concepts": bridge_concepts,
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

    # Layer 2: Cluster matching — ONLY from raw terms, not ConceptNet expansions
    # ConceptNet expansions are too semantically broad: "agricultural" → "human" →
    # "consciousness" → philosophy_of_mind. We use raw terms for cluster activation
    # and ConceptNet expansions only to enrich the LLM synthesis prompt.
    activated_clusters, activated_disciplines, bridge_concepts = _match_clusters(
        raw_terms, concept_map
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

    # Map cluster disciplines to theme_ids FIRST, so the LLM can see the
    # concrete candidate list it is being asked to prune.
    cluster_themes = _disciplines_to_themes(activated_disciplines, config)
    logger.info(
        f"[ConceptMapper] Cluster matching proposes {len(cluster_themes)} candidate themes: "
        f"{cluster_themes}"
    )

    # Layer 3: LLM synthesis — prune incidental activations, add missed themes
    llm_result = _llm_synthesis(
        problem, raw_terms, expanded_concepts,
        activated_clusters, activated_disciplines, bridge_concepts,
        cluster_themes, config
    )

    llm_disciplines = llm_result.get("disciplines_identified", [])
    llm_bridges     = llm_result.get("bridge_concepts", [])

    # Keep all cluster disciplines, add LLM disciplines only if they map to config
    all_disciplines = list(dict.fromkeys(
        activated_disciplines +
        [d for d in llm_disciplines if d.lower() in {
            t["theme_id"].replace("_", " ") for t in config.get("themes", [])
        } or any(
            d.lower() in tid.lower() or tid.lower() in d.lower()
            for tid in {t["theme_id"] for t in config.get("themes", [])}
        )]
    ))
    all_bridges = list(set(bridge_concepts + llm_bridges))

    config_theme_ids = {t["theme_id"] for t in config.get("themes", [])}

    # Apply LLM exclusions to cluster_themes.
    # An exclusion is only honored if:
    #   - the theme_id was actually a candidate
    #   - the reason is substantive (>= 20 chars, not generic)
    MIN_REASON_LEN = 20
    excluded_themes = set()
    exclusion_log = []
    for x in llm_result.get("themes_to_exclude", []):
        tid    = x.get("theme_id", "")
        reason = (x.get("reason") or "").strip()
        if tid in cluster_themes and len(reason) >= MIN_REASON_LEN:
            excluded_themes.add(tid)
            exclusion_log.append({"theme_id": tid, "reason": reason})
    pruned_themes = [t for t in cluster_themes if t not in excluded_themes]

    # Apply LLM additions, capped relative to the PRUNED candidate list so the
    # LLM cannot compensate for aggressive exclusions by over-adding.
    MAX_LLM_ADDITIONS = max(2, len(pruned_themes) // 2)
    llm_additions = []
    addition_log = []
    # Accept both the new "themes_to_add" field and the legacy "suggested_themes"
    suggested = llm_result.get("themes_to_add") or llm_result.get("suggested_themes") or []
    for s in suggested:
        if len(llm_additions) >= MAX_LLM_ADDITIONS:
            break
        tid    = s.get("theme_id", "")
        reason = (s.get("relevance_reason") or s.get("reason") or "").strip()
        if (tid in config_theme_ids
                and tid not in pruned_themes
                and tid not in excluded_themes
                and len(reason) >= MIN_REASON_LEN):
            llm_additions.append(tid)
            addition_log.append({"theme_id": tid, "reason": reason})

    # Final list: pruned cluster themes + LLM additions
    seen = set()
    final_themes = []
    for t in pruned_themes + llm_additions:
        if t not in seen:
            seen.add(t)
            final_themes.append(t)

    # If nothing matched — fallback to all themes rather than running blind
    if not final_themes:
        final_themes = [t["theme_id"] for t in config.get("themes", [])]
        logger.warning("[ConceptMapper] No themes matched — activating all themes")

    logger.info(f"[ConceptMapper] Candidate themes (from clusters): {cluster_themes}")
    if exclusion_log:
        logger.info(f"[ConceptMapper] LLM excluded {len(exclusion_log)} themes: "
                    f"{[e['theme_id'] for e in exclusion_log]}")
    if addition_log:
        logger.info(f"[ConceptMapper] LLM added {len(addition_log)} themes: "
                    f"{[a['theme_id'] for a in addition_log]}")
    logger.info(f"[ConceptMapper] Final themes activated: {final_themes}")

    result = {
        "raw_terms":            raw_terms,
        "expanded_concepts":    expanded_concepts,
        "activated_clusters":   activated_clusters,
        "activated_disciplines": all_disciplines,
        "bridge_concepts":      all_bridges,
        "candidate_themes":     cluster_themes,
        "excluded_themes":      exclusion_log,
        "added_themes":         addition_log,
        "final_themes":         final_themes,
        "llm_reasoning":        llm_result.get("conceptual_translation", ""),
        "llm_suggested_themes": llm_result.get("themes_to_add")
                                 or llm_result.get("suggested_themes", [])
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
        print(f"\n  Translation:   {result['llm_reasoning'][:200]}")
    print(f"{'─'*60}\n")
