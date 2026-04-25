"""
Test: Social bridge detection logic (Phase 3)
----------------------------------------------
Tests bridge gap detection and tree extension without real API calls.
"""
import sys, json, sqlite3
from pathlib import Path

TEST_DB = Path("/tmp/test_social_bridge.db")
if TEST_DB.exists():
    TEST_DB.unlink()

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

print("=" * 60)
print("  TEST: Social Bridge Detection")
print("=" * 60)

# Build a tree with temporal gaps
run_id = "TEST-SOCIAL-001"
tree = at.TreeBuilder(run_id)
root = tree.create_root("How does identity influence intelligence?")

q1 = tree.add_question(root, "What is the social construction of identity?")

# Claim with evidence from 1934
c1 = tree.add_claim(q1, "Mead established symbolic interactionism", confidence=0.8,
                     source_ids=["SEM-001"])
e1 = tree.add_evidence(c1, "SEM-001", evidence_type="book",
                        relationship="establishes",
                        snippet="Mind Self and Society",
                        metadata={"year": 1934, "title": "Mind, Self, and Society"})

# Claim with evidence from 2020 (86-year gap!)
c2 = tree.add_claim(q1, "Modern identity research uses fMRI", confidence=0.6,
                     source_ids=["SEM-002"])
e2 = tree.add_evidence(c2, "SEM-002", evidence_type="paper",
                        relationship="supports",
                        snippet="Neural correlates of self-referential processing",
                        metadata={"year": 2020, "title": "Neural Self-Reference"})

# Also add a question with a smaller gap (no bridge needed)
q2 = tree.add_question(root, "How is intelligence measured?")
c3 = tree.add_claim(q2, "IQ tests are standard", confidence=0.7,
                     source_ids=["SEM-003"])
e3 = tree.add_evidence(c3, "SEM-003", evidence_type="paper",
                        relationship="establishes", snippet="WAIS-IV manual",
                        metadata={"year": 2008, "title": "WAIS-IV"})
c4 = tree.add_claim(q2, "Raven's matrices are culture-fair", confidence=0.6,
                     source_ids=["SEM-004"])
e4 = tree.add_evidence(c4, "SEM-004", evidence_type="paper",
                        relationship="supports", snippet="Raven's SPM",
                        metadata={"year": 2003, "title": "Raven's SPM"})

# ─── Test bridge detection ────────────────────────────────────────────────

bridge_needs = tree.find_bridge_needs(min_gap_years=15)

print(f"\n  Bridge gaps detected: {len(bridge_needs)}")
for b in bridge_needs:
    print(f"    Q: {b['question'][:50]}...")
    print(f"    Gap: {b['earlier_year']} → {b['later_year']} ({b['gap_years']} years)")

# Should find the 1934→2020 gap but NOT the 2003→2008 gap
assert len(bridge_needs) >= 1, "Should detect at least 1 bridge gap"
big_gap = [b for b in bridge_needs if b['gap_years'] > 50]
assert len(big_gap) == 1, f"Should have 1 big gap (1934→2020), got {len(big_gap)}"
assert big_gap[0]['earlier_year'] == 1934
assert big_gap[0]['later_year'] == 2020
assert big_gap[0]['gap_years'] == 86
print(f"  ✓ Correctly identified 86-year gap (1934→2020)")

small_gaps = [b for b in bridge_needs if b['gap_years'] < 15]
assert len(small_gaps) == 0, "Should not flag gaps < 15 years"
print(f"  ✓ Correctly ignored small gaps")

# ─── Test bridge addition ─────────────────────────────────────────────────

# Simulate Social finding a bridge paper
bridge_id = tree.add_bridge(
    big_gap[0]['earlier_node'],
    big_gap[0]['later_node'],
    "BRG-SRC-001",
    bridge_type="temporal",
    description="[1968] Stryker, Sheldon — Identity Theory: Developments and Extensions",
    agent="social",
)
assert bridge_id.startswith("BRG-"), f"Bridge ID wrong: {bridge_id}"
print(f"  ✓ Bridge added: {bridge_id}")

# Verify tree now has the bridge
stats = tree.get_stats()
assert stats['by_type'].get('bridge', 0) == 1
print(f"  ✓ Tree has {stats['by_type']['bridge']} bridge node(s)")

# Render tree context to verify bridge shows up
ctx = tree.to_context()
assert "BRIDGE [temporal]" in ctx
print(f"  ✓ Bridge appears in tree context")

tree.close()
TEST_DB.unlink()

print(f"\n{'='*60}")
print(f"  ALL SOCIAL BRIDGE TESTS PASSED ✓")
print(f"{'='*60}")
