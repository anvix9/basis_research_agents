"""
Test: Historian tree audit + extension (Phase 4)
-------------------------------------------------
Tests the audit logic and tree extension without making real LLM calls.
"""
import sys, json, sqlite3
from pathlib import Path

TEST_DB = Path("/tmp/test_historian_tree.db")
if TEST_DB.exists():
    TEST_DB.unlink()

sys.path.insert(0, str(Path(__file__).parent.parent))
import core.argument_tree as at
at.DB_PATH = TEST_DB

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

print("=" * 60)
print("  TEST: Historian Tree Audit + Extension")
print("=" * 60)

run_id = "TEST-HISTORIAN-001"
tree = at.TreeBuilder(run_id)
root = tree.create_root("How does identity influence intelligence?")

# ── Build a tree with varying evidence quality ────────────────────────────

q1 = tree.add_question(root, "What is identity?")
q2 = tree.add_question(root, "What is intelligence?")
q3 = tree.add_question(root, "How are they related?")  # will be unanswered

# Q1: well-supported claim (2 evidence, no counter)
c1 = tree.add_claim(q1, "Identity is socially constructed", confidence=0.5,
                     source_ids=["SRC-1", "SRC-2"])
tree.add_evidence(c1, "SRC-1", "book", "establishes", "Mead 1934",
                   metadata={"year": 1934})
tree.add_evidence(c1, "SRC-2", "paper", "supports", "Tajfel 1979",
                   metadata={"year": 1979})

# Q1: contested claim (1 evidence + 1 counter)
c2 = tree.add_claim(q1, "Identity has innate biological components",
                     confidence=0.5, source_ids=["SRC-3"])
tree.add_evidence(c2, "SRC-3", "paper", "supports", "Pinker 2002",
                   metadata={"year": 2002})
tree.add_counter(c2, "Social constructionists deny biological basis", "SRC-4")

# Q2: weak claim (1 evidence only)
c3 = tree.add_claim(q2, "Intelligence is the g-factor", confidence=0.5,
                     source_ids=["SRC-5"])
tree.add_evidence(c3, "SRC-5", "paper", "establishes", "Spearman 1904",
                   metadata={"year": 1904})

# Q3: no claims (unanswered)

print(f"  Tree built: {tree.get_stats()['total_nodes']} nodes")

# ── Test audit logic ──────────────────────────────────────────────────────

# Simulate what Historian Job 1 does
claims = tree.get_nodes_by_type("claim")
assert len(claims) == 3, f"Should have 3 claims, got {len(claims)}"

for claim in claims:
    children = tree.get_children(claim["node_id"])
    evidence_nodes = [c for c in children if c["node_type"] == "evidence"]
    counter_nodes = [c for c in children if c["node_type"] == "counter"]

    if len(evidence_nodes) >= 2 and not counter_nodes:
        tree.add_audit_note(claim["node_id"], "Well-supported",
                           new_status="solid", new_confidence=0.85)
    elif counter_nodes:
        tree.add_audit_note(claim["node_id"], "Contested",
                           new_status="contested")
    elif len(evidence_nodes) == 1:
        tree.add_audit_note(claim["node_id"], "Single source — weak",
                           new_status="weak", new_confidence=0.4)
    else:
        tree.add_audit_note(claim["node_id"], "Unsupported",
                           new_status="unsupported", new_confidence=0.1)

# Verify audit results
c1_node = tree.get_node(c1)
assert c1_node["status"] == "solid", f"C1 should be solid, got {c1_node['status']}"
print(f"  ✓ Claim 1 (2 evidence, 0 counter): {c1_node['status']} ({c1_node['confidence']})")

c2_node = tree.get_node(c2)
assert c2_node["status"] == "contested", f"C2 should be contested, got {c2_node['status']}"
print(f"  ✓ Claim 2 (1 evidence, 1 counter): {c2_node['status']}")

c3_node = tree.get_node(c3)
assert c3_node["status"] == "weak", f"C3 should be weak, got {c3_node['status']}"
print(f"  ✓ Claim 3 (1 evidence only): {c3_node['status']} ({c3_node['confidence']})")

# Check gaps
gaps = tree.find_gaps()
unanswered = [g for g in gaps if g["gap_type"] == "unanswered_question"]
assert len(unanswered) == 1, f"Should have 1 unanswered question, got {len(unanswered)}"
assert "How are they related" in unanswered[0]["content"]
print(f"  ✓ Gap: {len(unanswered)} unanswered question (Q3)")

# ── Test tree extension (historical + external nodes) ─────────────────────

# Simulate Historian adding historical context
hist1 = tree.add_historical(root, "Descartes 1641 — mind-body dualism set the frame",
                             year=1641, source_id="HIST-001")
hist2 = tree.add_historical(root, "Locke 1690 — personal identity tied to memory",
                             year=1690, source_id="HIST-002")

# Simulate adding external factors
ext1 = tree.add_external(root, "WWII disrupted European psychology departments, shifting center to US",
                          factor_type="war", year=1945)
ext2 = tree.add_external(root, "Cognitive Revolution (1950s) — shift from behaviorism opened identity-cognition questions",
                          factor_type="institutional", year=1955)
ext3 = tree.add_external(root, "Brown v. Board of Education (1954) — brought racial identity into intelligence research",
                          factor_type="legal_change", year=1954)

print(f"\n  ✓ Added {2} historical nodes")
print(f"  ✓ Added {3} external factor nodes")

# Verify final tree
final = tree.get_stats()
assert final["by_type"].get("audit_note", 0) == 3
assert final["by_type"].get("historical", 0) == 2
assert final["by_type"].get("external", 0) == 3
print(f"\n  Final tree: {final['total_nodes']} nodes")
print(f"    {json.dumps(final['by_type'], indent=6)}")

# Verify context rendering includes all node types
ctx = tree.to_context()
assert "AUDIT:" in ctx
assert "HISTORICAL:" in ctx
assert "EXTERNAL [war]:" in ctx
assert "EXTERNAL [legal_change]:" in ctx
print(f"  ✓ All node types appear in context rendering")

tree.close()
TEST_DB.unlink()

print(f"\n{'='*60}")
print(f"  ALL HISTORIAN TREE TESTS PASSED ✓")
print(f"{'='*60}")
