#%%
"""
Analyze extraction results from spr_extraction_results_two_step_bindingdb.

Reports:
  - Per (PMID, DESCRIPTION) pair: how many succeeded vs failed in Step 1 and Step 2
  - Per protein-ligand pair: how many have / lack structured_description
"""

import json
import pandas as pd
from pathlib import Path

# ============================================================
# Load all result JSONs
# ============================================================
result_dir = Path("spr_extraction_results_two_step_bindingdb")
json_files = sorted(result_dir.glob("*.json"))
print(f"Total PMID JSON files: {len(json_files)}")

rows = []
for jf in json_files:
    with open(jf, 'r', encoding='utf-8') as f:
        pmid_data = json.load(f)
    for reactant_set_id, entry in pmid_data.items():
        original_paragraph = entry.get("original_paragraph", {})
        structured_description = entry.get("structured_description")

        # Determine if paragraph was found (non-empty dict with real content)
        paragraph_found = bool(original_paragraph) and original_paragraph != {}
        has_structured = structured_description is not None and structured_description != {}

        rows.append({
            "reactant_set_id": reactant_set_id,
            "pmid": str(entry.get("pmid", jf.stem)),
            "protein": entry.get("protein"),
            "ligand_smiles": entry.get("ligand", {}).get("smiles"),
            "DESCRIPTION": entry.get("DESCRIPTION", ""),
            "paragraph_found": paragraph_found,
            "has_structured": has_structured,
        })

df = pd.DataFrame(rows)
print(f"Total protein-ligand pair entries: {len(df)}")

# ============================================================
# Analysis at (PMID, DESCRIPTION) level
# ============================================================
desc_df = df.groupby(["pmid", "DESCRIPTION"]).agg(
    n_pairs=("reactant_set_id", "count"),
    paragraph_found=("paragraph_found", "first"),
    has_structured=("has_structured", "first"),
).reset_index()

total_desc = len(desc_df)
step1_success = desc_df["paragraph_found"].sum()
step1_fail = total_desc - step1_success
step2_success = desc_df["has_structured"].sum()
step2_fail = total_desc - step2_success

print(f"\n{'='*60}")
print(f"(PMID, DESCRIPTION) pair analysis")
print(f"{'='*60}")
print(f"Total unique (PMID, DESCRIPTION) pairs: {total_desc}")
print(f"  Step 1 (paragraph extraction):")
print(f"    Success: {step1_success} ({100*step1_success/total_desc:.1f}%)")
print(f"    Failed:  {step1_fail} ({100*step1_fail/total_desc:.1f}%)")
print(f"  Step 2 (structured description):")
print(f"    Success: {step2_success} ({100*step2_success/total_desc:.1f}%)")
print(f"    Failed:  {step2_fail} ({100*step2_fail/total_desc:.1f}%)")

# ============================================================
# Analysis at PMID level
# ============================================================
pmid_df = desc_df.groupby("pmid").agg(
    n_descs=("DESCRIPTION", "count"),
    n_pairs=("n_pairs", "sum"),
    any_success=("has_structured", "any"),
    all_success=("has_structured", "all"),
).reset_index()

total_pmids = len(pmid_df)
pmid_all_fail = (~pmid_df["any_success"]).sum()
pmid_all_success = pmid_df["all_success"].sum()
pmid_partial = total_pmids - pmid_all_fail - pmid_all_success

print(f"\n{'='*60}")
print(f"PMID-level analysis")
print(f"{'='*60}")
print(f"Total unique PMIDs: {total_pmids}")
print(f"  All descriptions succeeded:  {pmid_all_success} ({100*pmid_all_success/total_pmids:.1f}%)")
print(f"  Partial success:             {pmid_partial} ({100*pmid_partial/total_pmids:.1f}%)")
print(f"  All descriptions failed:     {pmid_all_fail} ({100*pmid_all_fail/total_pmids:.1f}%)")

# ============================================================
# Analysis at protein-ligand pair level
# ============================================================
total_pairs = len(df)
pairs_with_struct = df["has_structured"].sum()
pairs_without_struct = total_pairs - pairs_with_struct

print(f"\n{'='*60}")
print(f"Protein-ligand pair analysis")
print(f"{'='*60}")
print(f"Total protein-ligand pairs: {total_pairs}")
print(f"  With structured_description:    {pairs_with_struct} ({100*pairs_with_struct/total_pairs:.1f}%)")
print(f"  Without structured_description: {pairs_without_struct} ({100*pairs_without_struct/total_pairs:.1f}%)")

# ============================================================
# List failed (PMID, DESCRIPTION) pairs
# ============================================================
failed_descs = desc_df[~desc_df["has_structured"]].copy()
failed_descs = failed_descs.sort_values(["pmid", "DESCRIPTION"])

print(f"\n{'='*60}")
print(f"Failed (PMID, DESCRIPTION) pairs ({len(failed_descs)} total)")
print(f"{'='*60}")
for _, row in failed_descs.iterrows():
    reason = "Step 1 failed (no paragraph)" if not row["paragraph_found"] else "Step 2 failed (no structured desc)"
    print(f"  PMID {row['pmid']} | {row['n_pairs']} pairs | {reason}")
    print(f"    DESC: {row['DESCRIPTION'][:120]}")

# %%