"""
Test: Grounder tree-building logic (Phase 2)
---------------------------------------------
Tests the tree-building part of the Grounder without making real API calls.
Simulates sub-questions, search results, and verifies the tree structure.
"""
import sys, json, sqlite3
from pathlib import Path

# Setup test DB
TEST_DB = Path("/tmp/test_grounder_tree.db")
if TEST_DB.exists():
    TEST_DB.unlink()

# Patch DB path before imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import core.argument_tree as at
at.DB_PATH = TEST_DB

# Create minimal schema
conn = sqlite3.connect(str(TEST_DB))
conn.executescript("""
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, problem TEXT, created_at TEXT,
    status TEXT DEFAULT 'active', break0_done INTEGER DEFAULT 0,
    break1_done INTEGER DEFAULT 0, break2_done INTEGER DEFAULT 0,
    completed_at TEXT);
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY, run_id TEXT, title TEXT, authors TEXT,
    year INTEGER, source_name TEXT, doi TEXT, abstract TEXT,
    type TEXT, seminal_reason TEXT, historical_reason TEXT,
    active_link TEXT, theme_tags TEXT, relevance_rating TEXT,
    link_status TEXT, material_type TEXT, cited_by INTEGER DEFAULT 0);
""")
conn.commit()
conn.close()

at.init_tree_table()

# ─── Test 1: Tree structure after simulated Grounder search ───────────────

print("=" * 60)
print("  TEST: Grounder Tree-Building")
print("=" * 60)

run_id = "TEST-GROUNDER-001"
tree = at.TreeBuilder(run_id)

# Step 1: Create root
root = tree.create_root("How does identity influence intelligence?")
assert root.startswith("ROOT-"), f"Root ID format wrong: {root}"
print(f"  ✓ Root created: {root}")

# Step 2: Add sub-questions (simulating decomposition)
questions = [
    ("Q1", "What is identity?", "foundational"),
    ("Q2", "What is intelligence?", "foundational"),
    ("Q3", "What are the core tenets of identity theory?", "foundational"),
    ("Q4", "How does identity influence cognition?", "relational"),
    ("Q5", "Does identity explain intelligence?", "positional"),
]

q_nodes = {}
for qid, text, level in questions:
    node_id = tree.add_question(root, text, question_level=level)
    q_nodes[qid] = node_id
    assert node_id.startswith("Q-"), f"Question ID format wrong: {node_id}"

print(f"  ✓ {len(q_nodes)} questions added")

# Step 3: Simulate search results and add claims + evidence
# (this is what _process_results does in the real Grounder)
simulated_results = [
    {
        "question": "Q1",
        "title": "Mind, Self, and Society",
        "authors": ["George Herbert Mead"],
        "year": 1934,
        "abstract": "The self emerges through social interaction and symbolic communication",
        "source_name": "openalex",
        "evidence_type": "book",
    },
    {
        "question": "Q1",
        "title": "Social Identity and Intergroup Relations",
        "authors": ["Henri Tajfel"],
        "year": 1982,
        "abstract": "Social identity theory proposes that group membership is central to self-concept",
        "source_name": "semantic_scholar",
        "evidence_type": "paper",
    },
    {
        "question": "Q2",
        "title": "General Intelligence Objectively Determined and Measured",
        "authors": ["Charles Spearman"],
        "year": 1904,
        "abstract": "Proposes the g-factor theory of general intelligence based on psychometric analysis",
        "source_name": "openalex",
        "evidence_type": "paper",
    },
    {
        "question": "Q3",
        "title": "Identity Theory and Social Identity Theory",
        "authors": ["Jan E. Stets", "Peter J. Burke"],
        "year": 2000,
        "abstract": "Compares and contrasts structural identity theory with social identity theory",
        "source_name": "semantic_scholar",
        "evidence_type": "paper",
    },
    {
        "question": "Q4",
        "title": "Stereotype threat and the intellectual test performance of African Americans",
        "authors": ["Claude M. Steele", "Joshua Aronson"],
        "year": 1995,
        "abstract": "Social identity threat reduces cognitive performance on standardized tests",
        "source_name": "openalex",
        "evidence_type": "paper",
    },
]

for result in simulated_results:
    q_node = q_nodes[result["question"]]
    claim_text = f"{result['title']} ({', '.join(result['authors'][:2])}, {result['year']}): {result['abstract'][:100]}"
    claim_id = tree.add_claim(
        q_node, claim_text,
        confidence=0.5,
        source_ids=[f"TSRC-{result['title'][:10]}"],
        agent="grounder",
    )
    tree.add_evidence(
        claim_id,
        f"TSRC-{result['title'][:10]}",
        evidence_type=result["evidence_type"],
        relationship="supports",
        snippet=result["abstract"],
        agent="grounder",
        metadata={
            "title": result["title"],
            "authors": result["authors"],
            "year": result["year"],
            "source_name": result["source_name"],
        },
    )

print(f"  ✓ {len(simulated_results)} search results added as claims + evidence")

# ─── Verify tree structure ────────────────────────────────────────────────

stats = tree.get_stats()
print(f"\n  Tree stats:")
print(f"    Total nodes:    {stats['total_nodes']}")
print(f"    By type:        {stats['by_type']}")
print(f"    Unique sources: {stats['unique_sources']}")

assert stats['by_type'].get('root', 0) == 1, "Should have 1 root"
assert stats['by_type'].get('question', 0) == 5, "Should have 5 questions"
assert stats['by_type'].get('claim', 0) == 5, "Should have 5 claims"
assert stats['by_type'].get('evidence', 0) == 5, "Should have 5 evidence"
print(f"  ✓ Node counts correct")

# Check gaps — Q5 has no claims
gaps = tree.find_gaps()
unanswered = [g for g in gaps if g['gap_type'] == 'unanswered_question']
assert len(unanswered) == 1, f"Should have 1 unanswered question (Q5), got {len(unanswered)}"
assert "Does identity explain" in unanswered[0]['content'], "Gap should be Q5"
print(f"  ✓ Gap detection: {len(unanswered)} unanswered question (Q5)")

# Check tree context rendering
ctx = tree.to_context(max_depth=3)
assert "ROOT:" in ctx
assert "CLAIM [supported]" in ctx
assert "EVIDENCE [book]" in ctx
print(f"  ✓ Tree context rendering: {len(ctx)} chars")

# Check full tree structure
full = tree.get_tree()
assert full['node_type'] == 'root'
assert len(full['children']) == 5  # 5 questions
q1_children = full['children'][0]['children']  # claims under Q1
assert len(q1_children) == 2, f"Q1 should have 2 claims, got {len(q1_children)}"
print(f"  ✓ Full tree: root → {len(full['children'])} questions")
for q in full['children']:
    n_claims = len(q['children'])
    print(f"      Q ({q['content'][:40]}...): {n_claims} claims")

# ─── Test counter-argument ────────────────────────────────────────────────

# Add a counter to the first claim
first_claim = full['children'][0]['children'][0]
ctr = tree.add_counter(
    first_claim['node_id'],
    "Essentialists argue identity has innate biological components",
    "TSRC-counter"
)
# Verify parent status changed to contested
node = tree.get_node(first_claim['node_id'])
assert node['status'] == 'contested', f"Claim should be contested, got {node['status']}"
print(f"\n  ✓ Counter-argument: claim status → contested")

# ─── Test bridge ──────────────────────────────────────────────────────────

# Bridge between Q1 claim (1934) and Q4 claim (1995)
q1_claim = full['children'][0]['children'][0]['node_id']
q4_claim = full['children'][3]['children'][0]['node_id']
brg = tree.add_bridge(
    q1_claim, q4_claim, "TSRC-bridge",
    bridge_type="temporal",
    description="Stryker 1980 connects Mead's symbolic interactionism to modern identity-cognition research"
)
print(f"  ✓ Bridge created: {brg}")

# ─── Final stats ──────────────────────────────────────────────────────────

final_stats = tree.get_stats()
print(f"\n  Final tree: {final_stats['total_nodes']} nodes")
print(f"    {json.dumps(final_stats['by_type'], indent=6)}")
print(f"    Claim statuses: {final_stats['claim_statuses']}")

tree.close()

# Cleanup
TEST_DB.unlink()

print(f"\n{'='*60}")
print(f"  ALL GROUNDER TREE TESTS PASSED ✓")
print(f"{'='*60}")
