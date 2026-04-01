#%%
"""
Two-Step FPA Assay Extraction for a subset of protein-ligand pairs from BindingDB.

Input: BindingDB_assay_description.tsv
Selection criteria:
  1. assay_type == "Fluorescence Polarization / Anisotropy (FP)"
  2. Kd (nM), Ki (nM) or IC50 (nM) is not null (at least one)
  3. EC50 (nM) is null

Two-Step Approach (no Step 0 classification gate):
- Step 1: Extract original_paragraph using vision model (with paper images)
         Run ONCE per unique (PMID, DESCRIPTION) combination
- Step 2: Fill structured_description from extracted text only (no images)
         Run ONCE per unique (PMID, DESCRIPTION) — shared across all pairs
"""

import json
import pandas as pd
from pathlib import Path
from agent.assay_extraction_agent_two_step import TwoStepAssayExtractionAgent
from agent.prompts_fpa_two_step import (
    get_paragraph_extraction_prompt as get_fpa_paragraph_extraction_prompt,
    get_structured_description_from_text_prompt as get_fpa_structured_description_from_text_prompt
)

# ============================================================
# Load and filter data from BindingDB
# ============================================================
tsv_path = Path('/data/mwu11/LLM_affinity/BindingDB/BindingDB_assay_description.tsv')
data = pd.read_csv(tsv_path, sep='\t', low_memory=False)

# Filter: FPA assay type only
data = data[data['assay_type'] == 'Fluorescence Polarization / Anisotropy (FP)'].copy()

# Filter: Kd, Ki or IC50 must have a value (at least one), and EC50 must be NaN
data = data[
    (data['Kd (nM)'].notna() | data['Ki (nM)'].notna() | data['IC50 (nM)'].notna()) &
    data['EC50 (nM)'].isna()
]

# Parse affinity values: extract relation (>, <, =) and numeric value
def parse_affinity(val_str):
    val_str = str(val_str).strip()
    if val_str.startswith('>'):
        return '>', float(val_str[1:])
    elif val_str.startswith('<'):
        return '<', float(val_str[1:])
    else:
        return '=', float(val_str)

# Parse Kd values where present
kd_mask = data['Kd (nM)'].notna()
data.loc[kd_mask, ['kd_relation', 'kd_value']] = data.loc[kd_mask, 'Kd (nM)'].apply(
    lambda x: pd.Series(parse_affinity(x))
).values

# Parse Ki values where present
ki_mask = data['Ki (nM)'].notna()
data.loc[ki_mask, ['ki_relation', 'ki_value']] = data.loc[ki_mask, 'Ki (nM)'].apply(
    lambda x: pd.Series(parse_affinity(x))
).values

# Parse IC50 values where present
ic50_mask = data['IC50 (nM)'].notna()
data.loc[ic50_mask, ['ic50_relation', 'ic50_value']] = data.loc[ic50_mask, 'IC50 (nM)'].apply(
    lambda x: pd.Series(parse_affinity(x))
).values

# Convert PMID to string (drop NaN PMIDs)
data = data.dropna(subset=['PMID'])
data['PMID'] = data['PMID'].astype(float).astype(int).astype(str)

# Select a subset of PMIDs for testing
data = data[data['PMID'].isin(['21899328', '22913511', '24973029', '28797774', '34225180', '37708384', '30019901', '22608961', '19366247'])]



# Rename columns for convenience
data = data.rename(columns={
    'BindingDB Reactant_set_id': 'reactant_set_id',
    'Ligand SMILES': 'ligand_smiles',
    'Target Name': 'protein',
})

print(f"Total FPA Ki/IC50 entries: {len(data)}")
print(f"Unique PMIDs: {data['PMID'].nunique()}")
print(f"Unique (PMID, DESCRIPTION) combinations: {data.groupby(['PMID', 'DESCRIPTION']).ngroups}")

#%%
# Initialize Two-Step agent with FPA-specific prompts
agent = TwoStepAssayExtractionAgent(
    model_name="Qwen/Qwen3.5-27B",
    text_model_name=None,  # Use same model for Step 2 (text-only mode)
    pdf_dir="/data484_1/mwu11/downloaded_paper",
    torch_dtype="bfloat16",
    device="cuda:0",
    temperature=0.0,
    search_supplementary=True,
    search_references=True,
    ncbi_api_key="2877565f02e8c0800b1698e0b12f3e4b1108",
    paragraph_prompt_fn=get_fpa_paragraph_extraction_prompt,
    structured_prompt_fn=get_fpa_structured_description_from_text_prompt
)

#%%
# Process using optimized two-step approach:
# Step 1: Extract paragraph ONCE per unique (PMID, DESCRIPTION)
# Step 2: Fill structured_description ONCE per unique (PMID, DESCRIPTION)
# Then copy results to all pairs sharing the same (PMID, DESCRIPTION)

output_dir = Path("fpa_extraction_results_two_step_bindingdb")
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
    # Re-run if any entry has empty original_paragraph or null structured_description
    json_path = output_dir / f"{pmid_str}.json"
    if json_path.exists():
        needs_rerun = False
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            for entry in existing_data.values():
                op = entry.get("original_paragraph")
                sd = entry.get("structured_description")
                if isinstance(op, list) and len(op) == 0:
                    needs_rerun = True
                    break
                if sd is None:
                    needs_rerun = True
                    break
        except (json.JSONDecodeError, Exception):
            needs_rerun = True

        if not needs_rerun:
            print(f"\n[{pmid_count}/{total_pmids}] PMID {pmid_str}: already exists, skipping")
            saved_files[pmid_str] = json_path
            continue
        else:
            print(f"\n[{pmid_count}/{total_pmids}] PMID {pmid_str}: incomplete results, re-running")

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
            max_new_tokens=32768
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
                max_new_tokens=32768
            )
            structured_description = step2_result.get("structured_description")

        # ========== Copy results to ALL pairs with this DESCRIPTION ==========
        for _, row in desc_group.iterrows():
            reactant_set_id = row["reactant_set_id"]
            protein = str(row["protein"]) if pd.notna(row["protein"]) else None
            ligand_smiles = str(row["ligand_smiles"]) if pd.notna(row["ligand_smiles"]) else None

            affinity_data = []
            if pd.notna(row.get("kd_value")):
                affinity_data.append({
                    "type": "Kd",
                    "value": row["kd_value"],
                    "relation": row["kd_relation"],
                    "unit": "nM"
                })
            if pd.notna(row.get("ki_value")):
                affinity_data.append({
                    "type": "Ki",
                    "value": row["ki_value"],
                    "relation": row["ki_relation"],
                    "unit": "nM"
                })
            if pd.notna(row.get("ic50_value")):
                affinity_data.append({
                    "type": "IC50",
                    "value": row["ic50_value"],
                    "relation": row["ic50_relation"],
                    "unit": "nM"
                })

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
