#%%
import json
import pandas as pd
from pathlib import Path
from agent.assay_extraction_agent_qwen import AssayExtractionAgentQwen

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

#%%
# Initialize Qwen3-VL agent
agent = AssayExtractionAgentQwen(
    model_name="Qwen/Qwen3-VL-32B-Instruct",
    pdf_dir="/data/mwu11/LLM_affinity/gemini_test/downloaded_paper_kd",
    torch_dtype="bfloat16",
    device="cuda:0",
    ncbi_api_key="2877565f02e8c0800b1698e0b12f3e4b1108"
)

# Process dataframe and save as JSON files per PMID
output_dir = "spr_extraction_results"
saved_files = agent.process_dataframe_to_json(
    data,
    output_dir=output_dir,
    pmid_col="PMID",
    description_col="DESCRIPTION",
    reactant_set_id_col="reactant_set_id",
    protein_col="protein",
    ligand_name_col="ligand_name",
    ligand_smiles_col="ligand_smiles",
    affinity_type_col="affinity_type",
    affinity_value_col="affinity_value",
    affinity_relation_col="affinity_relation",
    affinity_unit_col="affinity_unit",
    limit=None,
    max_pages=None,
    delay=0.0
)

print(f"\nSaved JSON files: {list(saved_files.keys())}")

#%%
# Compare extracted results against ground truth
print("\n" + "="*80)
print("COMPARISON WITH GROUND TRUTH")
print("="*80)

total_comparisons = 0
total_matches = 0
field_stats = {}  # {field_name: {"matches": 0, "total": 0}}

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

        print(f"\n--- Reactant Set ID: {rsid} ---")
        print(f"Ligand: {gt_entry.get('ligand', {}).get('reference_name', 'N/A')}")

        if not extracted_sd:
            print("  Extraction: FAILED (no structured_description)")
            continue

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

            # Track statistics
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
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nOverall accuracy: {total_matches}/{total_comparisons} ({100*total_matches/total_comparisons:.1f}%)" if total_comparisons > 0 else "No comparisons made")

print("\nPer-field accuracy:")
for field, stats in sorted(field_stats.items()):
    acc = 100 * stats["matches"] / stats["total"] if stats["total"] > 0 else 0
    print(f"  {field}: {stats['matches']}/{stats['total']} ({acc:.1f}%)")

# Cleanup
agent.cleanup()

# %%