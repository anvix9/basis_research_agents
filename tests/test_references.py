"""
Test: Reference section generator (Phase 6)
---------------------------------------------
Builds a tree with source references, then generates reference sections
in APA, Chicago, and simple formats.
"""
import sys, json, sqlite3
from pathlib import Path

TEST_DB = Path("/tmp/test_refs.db")
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
    break1_done INTEGER DEFAULT 0, break2_done INTEGER DEFAULT 0, completed_at TEXT);
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY, run_id TEXT, title TEXT, authors TEXT,
    year INTEGER, source_name TEXT, doi TEXT, abstract TEXT,
    type TEXT, seminal_reason TEXT, historical_reason TEXT,
    active_link TEXT, theme_tags TEXT, relevance_rating TEXT,
    link_status TEXT, material_type TEXT, cited_by INTEGER DEFAULT 0);
""")

# Insert test sources
sources = [
    ("SRC-001", "TEST-REF", "Mind, Self, and Society", '["George Herbert Mead"]',
     1934, "openalex", "", "", "seminal", "Founded symbolic interactionism", "", "", "[]", "", "", "book", 0),
    ("SRC-002", "TEST-REF", "Social Identity and Intergroup Relations", '["Henri Tajfel"]',
     1982, "semantic_scholar", "10.1234/test", "", "seminal", "Social identity theory", "", "", "[]", "", "", "paper", 500),
    ("SRC-003", "TEST-REF", "Stereotype Threat and Performance", '["Claude M. Steele", "Joshua Aronson"]',
     1995, "openalex", "10.5678/test", "", "historical", "", "Demonstrated identity effects on cognition", "", "[]", "", "", "paper", 3000),
    ("SRC-004", "TEST-REF", "Neural Correlates of Self-Referential Processing", '["David A. Smith", "Jane B. Johnson", "Robert C. Williams"]',
     2020, "semantic_scholar", "", "", "current", "", "", "", "[]", "High", "", "paper", 50),
    ("SRC-005", "TEST-REF", "Geneva Conventions Commentary", '["ICRC"]',
     1952, "openalex", "", "", "seminal", "Core IHL document", "", "", "[]", "", "", "legal_document", 0),
]

for s in sources:
    conn.execute("INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", s)

conn.execute("INSERT INTO runs VALUES ('TEST-REF', 'Test problem', '2026-01-01', 'active', 0, 0, 0, NULL)")
conn.commit()
conn.close()

at.init_tree_table()

# Build tree referencing some of these sources
tree = at.TreeBuilder("TEST-REF")
root = tree.create_root("Test problem")
q1 = tree.add_question(root, "What is identity?")
c1 = tree.add_claim(q1, "Mead's interactionism", confidence=0.8, source_ids=["SRC-001"])
tree.add_evidence(c1, "SRC-001", "book", "establishes", "Mind Self and Society")
tree.add_evidence(c1, "SRC-002", "paper", "extends", "Tajfel's social identity")
c2 = tree.add_claim(q1, "Identity affects cognition", confidence=0.7, source_ids=["SRC-003"])
tree.add_evidence(c2, "SRC-003", "paper", "demonstrates", "Stereotype threat")
# SRC-004 and SRC-005 are NOT in the tree — only 001-003 should appear in tree-only refs
tree.close()

print("=" * 60)
print("  TEST: Reference Section Generator")
print("=" * 60)

# Import and patch
import tools.generate_references as rg

# Test 1: Tree-only references (should be 3 sources, not 5)
tree_ids = rg.load_tree_sources(TEST_DB, "TEST-REF")
assert len(tree_ids) == 3, f"Tree should reference 3 sources, got {len(tree_ids)}"
assert "SRC-001" in tree_ids
assert "SRC-002" in tree_ids
assert "SRC-003" in tree_ids
assert "SRC-004" not in tree_ids  # not in tree
assert "SRC-005" not in tree_ids  # not in tree
print(f"  ✓ Tree references: {len(tree_ids)} sources (correctly excludes non-tree sources)")

# Test 2: APA formatting
report_apa = rg.generate_reference_section(TEST_DB, "TEST-REF", fmt="apa")
assert "Mead, G. H." in report_apa, f"APA should have 'Mead, G. H.' — got: {report_apa[:500]}"
assert "Tajfel, H." in report_apa
assert "Steele, C. M." in report_apa
assert "(1934)" in report_apa
assert "(1982)" in report_apa
print(f"  ✓ APA format: correct author inversions and years")

# Test 3: Chicago formatting
report_chi = rg.generate_reference_section(TEST_DB, "TEST-REF", fmt="chicago")
assert "Mead, George Herbert" in report_chi
assert '1934' in report_chi
print(f"  ✓ Chicago format: correct")

# Test 4: Simple formatting
report_sim = rg.generate_reference_section(TEST_DB, "TEST-REF", fmt="simple")
assert "George Herbert Mead" in report_sim
print(f"  ✓ Simple format: correct")

# Test 5: All sources (not tree-only)
report_all = rg.generate_reference_section(TEST_DB, "TEST-REF", fmt="apa", tree_only=False)
assert "ICRC" in report_all, "All-sources should include SRC-005"
assert "Smith" in report_all, "All-sources should include SRC-004"
all_count = report_all.count("\n- ")
assert all_count == 5, f"All-sources should have 5 refs, got {all_count}"
print(f"  ✓ All-sources mode: {all_count} references (includes non-tree sources)")

# Test 6: Grouping by type
assert "## Seminal Works" in report_all
assert "## Historical Sources" in report_all
assert "## Contemporary Literature" in report_all
print(f"  ✓ References grouped by source type")

# Show a sample
print(f"\n  --- Sample APA output ---")
for line in report_apa.split("\n"):
    if line.startswith("- "):
        print(f"    {line}")

TEST_DB.unlink()

print(f"\n{'='*60}")
print(f"  ALL REFERENCE GENERATOR TESTS PASSED ✓")
print(f"{'='*60}")
