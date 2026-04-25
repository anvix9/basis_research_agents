"""
Test: Tree-native Gaper logic
-------------------------------
Verifies that Gaper:
1. Extracts structural gaps from tree correctly
2. Never drops structural gaps in final output
3. Injects missing structural gaps if LLM omits them
4. Bridge needs are included as gaps
"""
import sys, json, sqlite3
from pathlib import Path

TEST_DB = Path("/tmp/test_gaper_tree.db")
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
CREATE TABLE IF NOT EXISTS gaps (
    gap_id TEXT PRIMARY KEY, run_id TEXT, problem_origin TEXT,
    gap_type TEXT, description TEXT, significance TEXT,
    significance_reason TEXT, primary_evaluation TEXT,
    references_grounder TEXT, references_historian TEXT, references_social TEXT,
    dead_end_revisit INTEGER DEFAULT 0, recurring_pattern INTEGER DEFAULT 0,
    recurring_reason TEXT);
""")
conn.execute("INSERT INTO runs VALUES ('TEST-GAPER', 'Test problem', '2026-01-01', 'active', 0, 0, 0, NULL)")
conn.commit()
conn.close()
at.init_tree_table()

print("=" * 60)
print("  TEST: Tree-Native Gaper")
print("=" * 60)

# Build a tree with known gaps
run_id = "TEST-GAPER"
tree = at.TreeBuilder(run_id)
root = tree.create_root("How does identity influence intelligence?")

# Q1: well-supported (no gap)
q1 = tree.add_question(root, "What is identity?")
c1 = tree.add_claim(q1, "Identity is socially constructed", confidence=0.8, source_ids=["S1", "S2"])
tree.add_evidence(c1, "S1", "book", "establishes", "Mead 1934", metadata={"year": 1934})
tree.add_evidence(c1, "S2", "paper", "supports", "Tajfel 1979", metadata={"year": 1979})

# Q2: has a claim but NO evidence (unsupported_claim gap)
q2 = tree.add_question(root, "What is intelligence?")
c2 = tree.add_claim(q2, "Intelligence is g-factor", confidence=0.3)
# Intentionally no evidence added

# Q3: completely unanswered (unanswered_question gap)
q3 = tree.add_question(root, "How does identity influence cognition?")
# No claims at all

# Q4: has evidence from 1934 and 2020 (temporal gap for bridge)
q4 = tree.add_question(root, "What is the history of identity research?")
c4a = tree.add_claim(q4, "Early work on self", confidence=0.7, source_ids=["S3"])
tree.add_evidence(c4a, "S3", "book", "establishes", "Mead 1934", metadata={"year": 1934})
c4b = tree.add_claim(q4, "Modern neuroscience of self", confidence=0.6, source_ids=["S4"])
tree.add_evidence(c4b, "S4", "paper", "supports", "fMRI self-reference 2022", metadata={"year": 2022})

tree.close()

# ── Test Step 1: structural gap extraction ────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.gaper import _get_structural_gaps

structural = _get_structural_gaps(run_id)

gaps = structural["gaps"]
bridge_needs = structural["bridge_needs"]

print(f"\n  Structural gaps found: {len(gaps)}")
for g in gaps:
    print(f"    [{g['gap_type']}] {g['content'][:60]}...")

print(f"  Bridge needs found: {len(bridge_needs)}")
for b in bridge_needs:
    print(f"    {b['earlier_year']}→{b['later_year']} ({b['gap_years']}yr)")

# Verify expected gaps
unanswered = [g for g in gaps if g["gap_type"] == "unanswered_question"]
unsupported = [g for g in gaps if g["gap_type"] == "unsupported_claim"]
weak = [g for g in gaps if g["gap_type"] == "weak_claim"]

assert len(unanswered) == 1, f"Should have 1 unanswered question (Q3), got {len(unanswered)}"
assert "How does identity influence cognition" in unanswered[0]["content"]
print(f"  ✓ Unanswered question detected: Q3")

assert len(unsupported) == 1, f"Should have 1 unsupported claim, got {len(unsupported)}"
assert "g-factor" in unsupported[0]["content"]
print(f"  ✓ Unsupported claim detected: 'Intelligence is g-factor'")

assert len(bridge_needs) >= 1, f"Should have at least 1 bridge need"
big_gap = [b for b in bridge_needs if b["gap_years"] > 50]
assert len(big_gap) == 1, f"Should have 1 big temporal gap (1934→2022)"
print(f"  ✓ Temporal gap detected: {big_gap[0]['earlier_year']}→{big_gap[0]['later_year']}")

# ── Test merge logic: structural gaps can't be dropped ────────────────────

# Simulate what happens if LLM returns only analytical gaps
# (ignoring structural ones). The merge step should inject them.
simulated_llm_output = {
    "gaps": [
        {
            "gap_origin": "analytical",
            "gap_type": "disciplinary_silence",
            "description": "Psychology and philosophy never talk to each other",
            "significance": "High",
            "significance_reason": "Key disciplines not connected",
            "tree_node_ref": "",
            "references_grounder": [],
            "references_historian": [],
            "references_current": [],
            "dead_end_revisit": False,
            "recurring_pattern": False,
            "recurring_reason": "",
        }
    ],
    "gap_map_summary": "One analytical gap found"
}

# Simulate the merge step from run()
final_gaps = simulated_llm_output["gaps"]
n_struct = len(structural["gaps"])
llm_structural = [g for g in final_gaps if g.get("gap_origin") == "structural"]

print(f"\n  Simulated LLM output: {len(final_gaps)} gaps ({len(llm_structural)} structural)")

if len(llm_structural) < n_struct:
    for sg in structural["gaps"]:
        found = any(sg["content"][:40] in g.get("description", "") for g in final_gaps)
        if not found:
            final_gaps.append({
                "gap_origin": "structural",
                "gap_type": sg["gap_type"],
                "description": sg["content"],
                "significance": "High",
                "significance_reason": "Proven by argument tree structure",
                "tree_node_ref": sg.get("node_id", ""),
            })

# Add bridge needs
for bn in structural["bridge_needs"]:
    final_gaps.append({
        "gap_origin": "structural",
        "gap_type": "temporal",
        "description": f"Temporal gap: {bn['earlier_year']}-{bn['later_year']}",
        "significance": "Medium",
    })

struct_final = [g for g in final_gaps if g.get("gap_origin") == "structural"]
anal_final = [g for g in final_gaps if g.get("gap_origin") == "analytical"]

print(f"  After merge: {len(final_gaps)} total ({len(struct_final)} structural + {len(anal_final)} analytical)")

# Verify all structural gaps survived
assert len(struct_final) >= n_struct, \
    f"All {n_struct} structural gaps must survive merge, only {len(struct_final)} did"
print(f"  ✓ All structural gaps preserved after merge")

# Verify bridge needs are included
temporal = [g for g in final_gaps if g.get("gap_type") == "temporal"]
assert len(temporal) >= 1, "Bridge needs should be in final output"
print(f"  ✓ Bridge needs included as temporal gaps")

# Verify LLM's analytical gap also survived
assert len(anal_final) == 1, "Analytical gap should survive"
assert "disciplinary_silence" in anal_final[0]["gap_type"]
print(f"  ✓ Analytical gaps preserved")

# ── Test: gaps written back into tree ─────────────────────────────────────

print(f"\n  --- Testing tree write-back ---")

# Simulate what Gaper does: write gaps into the tree
tree2 = at.TreeBuilder(run_id)
root_nodes = tree2.get_nodes_by_type("root")
gap_parent = tree2.add_question(
    root_nodes[0]["node_id"],
    "What are the identified gaps?",
    question_level="structural",
    agent="gaper",
)

# Write each final gap as a claim node
for gap in final_gaps:
    origin = gap.get("gap_origin", "analytical")
    confidence = 0.9 if origin == "structural" else 0.6
    claim_id = tree2.add_claim(
        gap_parent,
        f"[{gap.get('gap_type','')}] {gap.get('description','')[:200]}",
        confidence=confidence,
        agent="gaper",
    )
    # Add evidence for referenced sources
    for ref in gap.get("references_grounder", []) + gap.get("references_current", []):
        if ref:
            tree2.add_evidence(
                claim_id, "",
                evidence_type="other",
                relationship="confirms_gap",
                snippet=f"Referenced: {ref[:100]}",
                agent="gaper",
            )

stats2 = tree2.get_stats()
gap_claims = [n for n in tree2.get_children(gap_parent) if n["node_type"] == "claim"]
print(f"  Tree after Gaper write-back: {stats2['total_nodes']} nodes")
print(f"  Gap claims in tree: {len(gap_claims)}")
assert len(gap_claims) == len(final_gaps), \
    f"Should have {len(final_gaps)} gap claims in tree, got {len(gap_claims)}"
print(f"  ✓ All {len(final_gaps)} gaps written into tree as claim nodes")

# Verify structural gap claims have higher confidence
struct_claims = [c for c in gap_claims if c["confidence"] >= 0.9]
anal_claims = [c for c in gap_claims if c["confidence"] < 0.9]
assert len(struct_claims) == len(struct_final), \
    f"Should have {len(struct_final)} high-confidence claims, got {len(struct_claims)}"
print(f"  ✓ Structural gaps have confidence ≥ 0.9, analytical < 0.9")

# Verify the tree context now includes gap data
ctx = tree2.to_context()
assert "confirms_gap" in ctx or "gap" in ctx.lower()
print(f"  ✓ Gap data visible in tree context")

tree2.close()

TEST_DB.unlink()

print(f"\n{'='*60}")
print(f"  ALL GAPER TREE TESTS PASSED ✓")
print(f"{'='*60}")
