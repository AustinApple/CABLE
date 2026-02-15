#%%
"""
Two-Step SPR Assay Extraction for ALL protein-ligand pairs from BindingDB.

Input: BindingDB_assay_description.tsv
Selection criteria:
  1. assay_type == "Surface Plasmon Resonance (SPR)"
  2. Kd (nM) is not null
  3. Ki (nM), IC50 (nM), EC50 (nM) are all null (Kd-only entries)

Two-Step Approach:
- Step 1: Extract original_paragraph using vision model (with paper images)
         Run ONCE per unique (PMID, DESCRIPTION) combination
- Step 2: Fill structured_description from extracted text only (no images)
         Run ONCE per unique (PMID, DESCRIPTION) — shared across all pairs
"""

import json
import pandas as pd
from pathlib import Path
from agent.assay_extraction_agent_two_step import TwoStepAssayExtractionAgent

# ============================================================
# Load and filter data from BindingDB
# ============================================================
tsv_path = Path('/data/mwu11/LLM_affinity/BindingDB/BindingDB_assay_description.tsv')
data = pd.read_csv(tsv_path, sep='\t', low_memory=False)

# Filter: SPR assay type only
data = data[data['assay_type'] == 'Surface Plasmon Resonance (SPR)'].copy()

# Filter: Kd must have a value, and other affinity types must be NaN
data = data[
    data['Kd (nM)'].notna() &
    data['Ki (nM)'].isna() &
    data['IC50 (nM)'].isna() &
    data['EC50 (nM)'].isna()
]

# Parse Kd values: extract relation (>, <, =) and numeric value
def parse_kd(kd_str):
    kd_str = str(kd_str).strip()
    if kd_str.startswith('>'):
        return '>', float(kd_str[1:])
    elif kd_str.startswith('<'):
        return '<', float(kd_str[1:])
    else:
        return '=', float(kd_str)

data[['kd_relation', 'kd_value']] = data['Kd (nM)'].apply(
    lambda x: pd.Series(parse_kd(x))
)

# Convert PMID to string (drop NaN PMIDs)
data = data.dropna(subset=['PMID'])
data['PMID'] = data['PMID'].astype(float).astype(int).astype(str)

# Rename columns for convenience
data = data.rename(columns={
    'BindingDB Reactant_set_id': 'reactant_set_id',
    'Ligand SMILES': 'ligand_smiles',
    'Target Name': 'protein',
})

print(f"Total SPR Kd-only entries: {len(data)}")
print(f"Unique PMIDs: {data['PMID'].nunique()}")
print(f"Unique (PMID, DESCRIPTION) combinations: {data.groupby(['PMID', 'DESCRIPTION']).ngroups}")

#%%
# Initialize Two-Step agent
agent = TwoStepAssayExtractionAgent(
    model_name="Qwen/Qwen3-VL-32B-Instruct",
    text_model_name=None,  # Use same model for Step 2 (text-only mode)
    pdf_dir="/data/mwu11/LLM_affinity/gemini_test/downloaded_paper_kd",
    torch_dtype="bfloat16",
    device="cuda:0",
    temperature=0.0,
    search_supplementary=True,
    search_references=True,
    ncbi_api_key="2877565f02e8c0800b1698e0b12f3e4b1108"
)

#%%
# Process using optimized two-step approach:
# Step 1: Extract paragraph ONCE per unique (PMID, DESCRIPTION)
# Step 2: Fill structured_description ONCE per unique (PMID, DESCRIPTION)
# Then copy results to all pairs sharing the same (PMID, DESCRIPTION)

output_dir = Path("spr_extraction_results_two_step_bindingdb")
output_dir.mkdir(parents=True, exist_ok=True)

saved_files = {}
pmid_groups = data.groupby("PMID")

total_pmids = len(pmid_groups)
pmid_count = 0

for pmid, pmid_group in pmid_groups:
    pmid_count += 1
    pmid_str = str(pmid)
    pmid_results = {}

    # Skip if already processed (resume support)
    json_path = output_dir / f"{pmid_str}.json"
    if json_path.exists():
        print(f"\n[{pmid_count}/{total_pmids}] PMID {pmid_str}: already exists, skipping")
        saved_files[pmid_str] = json_path
        continue

    print(f"\n{'='*80}")
    print(f"Processing PMID {pmid_count}/{total_pmids}: {pmid_str} ({len(pmid_group)} pairs)")
    print(f"{'='*80}")

    # Group by DESCRIPTION within this PMID
    desc_groups = pmid_group.groupby("DESCRIPTION")
    unique_descs = len(desc_groups)
    print(f"  Unique DESCRIPTIONs: {unique_descs}")

    desc_count = 0
    for description, desc_group in desc_groups:
        desc_count += 1
        print(f"\n  --- DESCRIPTION {desc_count}/{unique_descs} ({len(desc_group)} pairs) ---")
        print(f"  {description[:100]}...")

        # ========== STEP 1: Extract paragraph ONCE for this DESCRIPTION ==========
        print(f"\n  [Step 1] Extracting paragraph (vision model)...")
        step1_result = agent.extract_paragraph(
            pmid=pmid_str,
            assay_description=description,
            max_pages=None,
            max_new_tokens=2048
        )

        extracted_paragraph = step1_result.get("original_paragraph", {})
        paragraph_found = agent._is_paragraph_found(extracted_paragraph)

        if paragraph_found:
            print(f"  [Step 1] SUCCESS - Paragraph extracted")
        else:
            print(f"  [Step 1] NOT FOUND - No relevant paragraph")

        # ========== STEP 2: Fill structured_description ONCE for this DESCRIPTION ==========
        structured_description = None
        if paragraph_found:
            print(f"\n    [Step 2] Extracting structured_description (text-only)...")
            step2_result = agent.fill_structured_description(
                extracted_paragraph=extracted_paragraph,
                assay_description=description,
                max_new_tokens=2048
            )
            structured_description = step2_result.get("structured_description")

        # ========== Copy results to ALL pairs with this DESCRIPTION ==========
        for _, row in desc_group.iterrows():
            reactant_set_id = row["reactant_set_id"]
            protein = str(row["protein"]) if pd.notna(row["protein"]) else None
            ligand_smiles = str(row["ligand_smiles"]) if pd.notna(row["ligand_smiles"]) else None

            affinity_data = {
                "type": "Kd",
                "value": row["kd_value"],
                "relation": row["kd_relation"],
                "unit": "nM"
            }

            entry = {
                "reactant_set_id": int(reactant_set_id),
                "pmid": int(pmid),
                "protein": protein,
                "ligand": {"smiles": ligand_smiles},
                "affinity_data": affinity_data,
                "DESCRIPTION": description,
                "search_path": step1_result.get("search_path", []),
                "supplementary_source": step1_result.get("supplementary_source", []),
                "references_previous": step1_result.get("references_previous"),
                "original_paragraph": extracted_paragraph,
                "structured_description": structured_description
            }

            key = str(int(reactant_set_id))
            pmid_results[key] = entry

    # Save results for this PMID
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(pmid_results, f, indent=4, ensure_ascii=False)
    print(f"\n  Saved: {json_path}")
    saved_files[pmid_str] = json_path

# Print token usage
agent._print_token_summary()
print(f"\nTotal PMIDs processed: {len(saved_files)}")

#%%
# Cleanup
agent.cleanup()

# %%