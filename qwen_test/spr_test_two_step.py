#%%
"""
Test script for the Two-Step SPR Assay Extraction method.

Two-Step Approach:
- Step 1: Extract original_paragraph using vision model (with paper images)
         Run ONCE per unique (PMID, DESCRIPTION) combination
- Step 2: Fill structured_description from extracted text only (no images)
         Run for each protein-ligand pair with their specific context

This is more efficient when multiple pairs share the same DESCRIPTION.
"""

import json
import pandas as pd
from pathlib import Path
from agent.assay_extraction_agent_two_step import TwoStepAssayExtractionAgent

# Load all ground truth files from spr_ground_truth directory
ground_truth_dir = Path('/data/mwu11/LLM_affinity/spr_ground_truth')
all_ground_truth = {}  # {pmid: {reactant_set_id: entry}}

rows = []
for gt_file in sorted(ground_truth_dir.glob('*.json')):
    pmid = gt_file.stem  # e.g., "21513882"
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    all_ground_truth[pmid] = gt_data

    for reactant_set_id, entry in gt_data.items():
        rows.append({
            "reactant_set_id": entry["reactant_set_id"],
            "PMID": entry["pmid"],
            "protein": entry["protein"],
            "ligand_smiles": entry["ligand"]["smiles"],
            "ligand_name": entry["ligand"]["reference_name"],
            "affinity_type": entry["affinity_data"]["type"],
            "affinity_value": entry["affinity_data"]["value"],
            "affinity_relation": entry["affinity_data"]["relation"],
            "affinity_unit": entry["affinity_data"]["unit"],
            "DESCRIPTION": entry["DESCRIPTION"],
        })

data = pd.DataFrame(rows)
print(f"Loaded {len(all_ground_truth)} ground truth files")
print(f"Total entries to process: {len(data)}")
print(f"\nEntries per PMID:")
print(data.groupby("PMID").size().to_string())

# Show unique DESCRIPTION counts per PMID
print(f"\nUnique DESCRIPTIONs per PMID:")
for pmid, group in data.groupby("PMID"):
    unique_descs = group["DESCRIPTION"].nunique()
    total_pairs = len(group)
    print(f"  PMID {pmid}: {unique_descs} unique DESCRIPTION(s) for {total_pairs} pairs")

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
# Step 2: Fill structured_description for each pair

output_dir = Path("spr_extraction_results_two_step")
output_dir.mkdir(parents=True, exist_ok=True)

saved_files = {}
pmid_groups = data.groupby("PMID")

total_pmids = len(pmid_groups)
pmid_count = 0

for pmid, pmid_group in pmid_groups:
    pmid_count += 1
    pmid_str = str(int(pmid))
    pmid_results = {}

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

        # ========== STEP 2: Fill structured_description for EACH pair ==========
        for _, row in desc_group.iterrows():
            reactant_set_id = row["reactant_set_id"]
            protein = str(row["protein"]) if pd.notna(row["protein"]) else None
            ligand_name = str(row["ligand_name"]) if pd.notna(row["ligand_name"]) else None
            ligand_smiles = str(row["ligand_smiles"]) if pd.notna(row["ligand_smiles"]) else None

            affinity_data = {
                "type": str(row["affinity_type"]) if pd.notna(row["affinity_type"]) else "Kd",
                "value": float(row["affinity_value"]) if pd.notna(row["affinity_value"]) else None,
                "relation": str(row["affinity_relation"]) if pd.notna(row["affinity_relation"]) else "=",
                "unit": str(row["affinity_unit"]) if pd.notna(row["affinity_unit"]) else "nM"
            }

            print(f"\n    [Step 2] Processing pair: {ligand_name} / {protein}")

            if paragraph_found:
                # Run Step 2 with this pair's specific context
                step2_result = agent.fill_structured_description(
                    extracted_paragraph=extracted_paragraph,
                    assay_description=description,
                    protein=protein,
                    ligand_smiles=ligand_smiles,
                    affinity_data=affinity_data,
                    max_new_tokens=2048
                )
                structured_description = step2_result.get("structured_description")
            else:
                structured_description = None

            # Build result entry
            entry = {
                "reactant_set_id": int(reactant_set_id),
                "pmid": int(pmid),
                "protein": protein,
                "ligand": {"reference_name": ligand_name, "smiles": ligand_smiles},
                "affinity_data": affinity_data,
                "DESCRIPTION": description,
                "search_path": step1_result.get("search_path", []),
                "supplementary_source": step1_result.get("supplementary_source", []),
                "references_previous": step1_result.get("references_previous"),
                "original_paragraph": extracted_paragraph,
                "structured_description": structured_description
            }

            key = str(reactant_set_id)
            pmid_results[key] = entry

    # Save results for this PMID
    json_path = output_dir / f"{pmid_str}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(pmid_results, f, indent=4, ensure_ascii=False)
    print(f"\n  Saved: {json_path}")
    saved_files[pmid_str] = json_path

# Print token usage
agent._print_token_summary()
print(f"\nSaved JSON files: {list(saved_files.keys())}")

#%%
# Compare extracted results against ground truth
print("\n" + "="*80)
print("COMPARISON WITH GROUND TRUTH (Two-Step Method)")
print("="*80)

total_comparisons = 0
total_matches = 0
field_stats = {}  # {field_name: {"matches": 0, "total": 0}}

# Track paragraph extraction success
paragraph_found_count = 0
paragraph_not_found_count = 0

for pmid_str, json_path in saved_files.items():
    with open(json_path, 'r') as f:
        extracted_data = json.load(f)

    gt_for_pmid = all_ground_truth.get(pmid_str, {})

    print(f"\n{'='*60}")
    print(f"PMID: {pmid_str}")
    print(f"{'='*60}")

    for rsid, extracted_entry in extracted_data.items():
        gt_entry = gt_for_pmid.get(rsid, {})
        gt_sd = gt_entry.get("structured_description", {})
        extracted_sd = extracted_entry.get("structured_description", {})
        extracted_paragraph = extracted_entry.get("original_paragraph", {})

        print(f"\n--- Reactant Set ID: {rsid} ---")
        print(f"Ligand: {gt_entry.get('ligand', {}).get('reference_name', 'N/A')}")
        print(f"Search Path: {extracted_entry.get('search_path', 'N/A')}")

        # Check if paragraph was extracted
        if extracted_paragraph and isinstance(extracted_paragraph, dict):
            has_content = any(v and str(v).strip() and not str(v).startswith("error")
                            for v in extracted_paragraph.values())
            if has_content:
                paragraph_found_count += 1
                print(f"  [Step 1] Paragraph: FOUND")
            else:
                paragraph_not_found_count += 1
                print(f"  [Step 1] Paragraph: NOT FOUND")
        else:
            paragraph_not_found_count += 1
            print(f"  [Step 1] Paragraph: NOT FOUND")

        if not extracted_sd:
            print("  [Step 2] Structured: FAILED")
            continue

        print(f"  [Step 2] Structured: SUCCESS")

        # Compare key fields
        comparisons = [
            ("instrument.model", gt_sd.get("instrument", {}).get("model"),
             extracted_sd.get("instrument", {}).get("model") if extracted_sd.get("instrument") else None),
            ("sensor_chip.type", gt_sd.get("sensor_chip", {}).get("type"),
             extracted_sd.get("sensor_chip", {}).get("type") if extracted_sd.get("sensor_chip") else None),
            ("immobilization.strategy", gt_sd.get("immobilization", {}).get("strategy"),
             extracted_sd.get("immobilization", {}).get("strategy") if extracted_sd.get("immobilization") else None),
            ("assay_conditions.pH", gt_sd.get("assay_conditions", {}).get("pH"),
             extracted_sd.get("assay_conditions", {}).get("pH") if extracted_sd.get("assay_conditions") else None),
            ("assay_conditions.assay_type", gt_sd.get("assay_conditions", {}).get("assay_type"),
             extracted_sd.get("assay_conditions", {}).get("assay_type") if extracted_sd.get("assay_conditions") else None),
            ("data_analysis.fitting_model", gt_sd.get("data_analysis", {}).get("fitting_model"),
             extracted_sd.get("data_analysis", {}).get("fitting_model") if extracted_sd.get("data_analysis") else None),
        ]

        for field, gt_val, ext_val in comparisons:
            is_match = str(gt_val) == str(ext_val)
            match_symbol = "✓" if is_match else "✗"
            print(f"  {match_symbol} {field}: GT={gt_val} | Extracted={ext_val}")

            total_comparisons += 1
            if is_match:
                total_matches += 1

            if field not in field_stats:
                field_stats[field] = {"matches": 0, "total": 0}
            field_stats[field]["total"] += 1
            if is_match:
                field_stats[field]["matches"] += 1

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS (Two-Step Method)")
print("="*80)

print(f"\n--- Step 1: Paragraph Extraction ---")
total_paragraphs = paragraph_found_count + paragraph_not_found_count
if total_paragraphs > 0:
    print(f"Paragraphs found: {paragraph_found_count}/{total_paragraphs} ({100*paragraph_found_count/total_paragraphs:.1f}%)")

print(f"\n--- Step 2: Structured Description ---")
print(f"Overall accuracy: {total_matches}/{total_comparisons} ({100*total_matches/total_comparisons:.1f}%)" if total_comparisons > 0 else "No comparisons made")

print("\nPer-field accuracy:")
for field, stats in sorted(field_stats.items()):
    acc = 100 * stats["matches"] / stats["total"] if stats["total"] > 0 else 0
    print(f"  {field}: {stats['matches']}/{stats['total']} ({acc:.1f}%)")

# Cleanup
agent.cleanup()

# %%
