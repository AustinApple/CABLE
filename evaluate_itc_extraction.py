#!/usr/bin/env python3
"""
Evaluation script to compare agent-extracted ITC structured_description with ground truth.

This script evaluates the performance of the assay extraction agent by comparing
the extracted structured_description fields against manually curated ground truth
for Isothermal Titration Calorimetry (ITC) experiments.

Metrics computed:
1. Per-field accuracy (string fuzzy match, numeric tolerance, enum exact match)
2. Per-section weighted scores
3. Critical field accuracy
4. Overall field coverage and match rate
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# Reuse helper functions from SPR evaluation
from evaluate_agent_extraction import (
    compare_string_fields,
    compare_numeric_fields,
    compare_enum_fields,
    compare_nested_object,
    evaluate_structured_field,
    evaluate_structured_description as _evaluate_structured_description_generic,
    evaluate_batch as _evaluate_batch_generic,
    load_json_file,
    load_from_directory,
)


# =============================================================================
# ITC-Specific Field Configuration
# =============================================================================

# Field configuration for ITC structured_description comparison
# Each field has: type (string/numeric/enum/nested), weight, critical flag, and type-specific params
FIELD_CONFIG = {
    "instrument": {
        "model": {"type": "string", "weight": 2.0, "critical": True}
    },
    "protein": {
        "name": {"type": "string", "weight": 1.5},
        "location": {"type": "enum", "weight": 1.5, "critical": True},
        "concentration": {"type": "string", "weight": 1.0},
        "solution_volume": {"type": "string", "weight": 0.5}
    },
    "ligand": {
        "name": {"type": "string", "weight": 1.0},
        "location": {"type": "enum", "weight": 1.5, "critical": True},
        "concentration": {"type": "string", "weight": 1.0}
    },
    "assay_conditions": {
        "buffer_composition": {"type": "string", "weight": 1.0},
        "pH": {"type": "numeric", "weight": 1.0, "tolerance": 0.5},
        "temperature_c": {"type": "numeric", "weight": 1.0, "tolerance": 1.0},
        "stirring_speed": {"type": "string", "weight": 0.5},
        "reference_cell_content": {"type": "string", "weight": 0.5},
        "control_method": {"type": "string", "weight": 0.5},
        "injection_parameters": {"type": "nested", "weight": 1.5}
    },
    "data_analysis": {
        "binding_model": {"type": "string", "weight": 2.0, "critical": True}
    }
}


# =============================================================================
# ITC-Specific Evaluation (overriding FIELD_CONFIG)
# =============================================================================

def evaluate_structured_description(gt_struct, ext_struct):
    """Evaluate ITC structured_description field-by-field using ITC FIELD_CONFIG."""
    result = {
        "evaluation_status": "comparable",
        "section_scores": {},
        "field_details": {},
        "aggregate": {
            "overall_score": 0.0,
            "critical_score": 0.0,
            "field_coverage": 0.0,
            "total_fields": 0,
            "matched_fields": 0,
            "critical_fields_total": 0,
            "critical_fields_matched": 0
        }
    }

    if gt_struct is None and ext_struct is None:
        result["evaluation_status"] = "both_missing"
        return result
    if gt_struct is None:
        result["evaluation_status"] = "ground_truth_missing"
        return result
    if ext_struct is None:
        result["evaluation_status"] = "extraction_missing"
        return result

    all_scores = []
    all_weights = []
    critical_scores = []
    total_fields = 0
    matched_fields = 0
    critical_total = 0
    critical_matched = 0

    for section_name, section_config in FIELD_CONFIG.items():
        gt_section = gt_struct.get(section_name, {}) or {}
        ext_section = ext_struct.get(section_name, {}) or {}

        section_scores = []
        section_weights = []
        section_details = {}

        for field_name, field_config in section_config.items():
            gt_value = gt_section.get(field_name) if isinstance(gt_section, dict) else None
            ext_value = ext_section.get(field_name) if isinstance(ext_section, dict) else None

            if gt_value is None or (isinstance(gt_value, str) and not gt_value.strip()):
                continue

            total_fields += 1
            field_result = evaluate_structured_field(gt_value, ext_value, field_config)
            section_details[field_name] = field_result

            weight = field_config.get("weight", 1.0)
            section_scores.append(field_result["score"])
            section_weights.append(weight)
            all_scores.append(field_result["score"])
            all_weights.append(weight)

            if field_result["is_match"]:
                matched_fields += 1

            if field_config.get("critical", False):
                critical_total += 1
                critical_scores.append(field_result["score"])
                if field_result["is_match"]:
                    critical_matched += 1

        if section_scores:
            weighted_sum = sum(s * w for s, w in zip(section_scores, section_weights))
            total_weight = sum(section_weights)
            result["section_scores"][section_name] = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            result["section_scores"][section_name] = None

        result["field_details"][section_name] = section_details

    if all_scores:
        weighted_sum = sum(s * w for s, w in zip(all_scores, all_weights))
        total_weight = sum(all_weights)
        result["aggregate"]["overall_score"] = weighted_sum / total_weight if total_weight > 0 else 0.0

    if critical_scores:
        result["aggregate"]["critical_score"] = sum(critical_scores) / len(critical_scores)

    result["aggregate"]["field_coverage"] = matched_fields / total_fields if total_fields > 0 else 0.0
    result["aggregate"]["total_fields"] = total_fields
    result["aggregate"]["matched_fields"] = matched_fields
    result["aggregate"]["critical_fields_total"] = critical_total
    result["aggregate"]["critical_fields_matched"] = critical_matched

    return result


def evaluate_batch(ground_truth_data, extracted_data, id_key="reactant_set_id"):
    """Evaluate a batch of ITC extractions using ITC-specific field config."""
    ext_lookup = {e.get(id_key): e for e in extracted_data if e.get(id_key) is not None}

    results = []
    structured_stats = defaultdict(list)
    section_stats = defaultdict(list)

    for gt_entry in ground_truth_data:
        entry_id = gt_entry.get(id_key)
        ext_entry = ext_lookup.get(entry_id, {})

        gt_struct = gt_entry.get("structured_description")
        ext_struct = ext_entry.get("structured_description")

        struct_result = evaluate_structured_description(gt_struct, ext_struct)

        eval_result = {
            id_key: entry_id,
            "pmid": gt_entry.get("pmid", "N/A"),
            "description": gt_entry.get("DESCRIPTION", "N/A")[:100] if gt_entry.get("DESCRIPTION") else "N/A",
            "evaluation": struct_result
        }
        results.append(eval_result)

        if struct_result["evaluation_status"] == "comparable":
            agg = struct_result["aggregate"]
            structured_stats["overall_score"].append(agg["overall_score"])
            structured_stats["critical_score"].append(agg["critical_score"])
            structured_stats["field_coverage"].append(agg["field_coverage"])
            structured_stats["matched_fields"].append(agg["matched_fields"])
            structured_stats["total_fields"].append(agg["total_fields"])

            for section, score in struct_result["section_scores"].items():
                if score is not None:
                    section_stats[section].append(score)

    # Compute aggregate statistics
    aggregate = {
        "total_entries": len(ground_truth_data),
        "entries_matched": len([r for r in results if r.get(id_key) in ext_lookup]),
        "entries_evaluated": len(structured_stats["overall_score"]),
    }

    if structured_stats["overall_score"]:
        aggregate["mean_overall_score"] = sum(structured_stats["overall_score"]) / len(structured_stats["overall_score"])
        aggregate["mean_critical_score"] = sum(structured_stats["critical_score"]) / len(structured_stats["critical_score"])
        aggregate["mean_field_coverage"] = sum(structured_stats["field_coverage"]) / len(structured_stats["field_coverage"])

        total_matched = sum(structured_stats["matched_fields"])
        total_fields = sum(structured_stats["total_fields"])
        aggregate["total_fields_evaluated"] = total_fields
        aggregate["total_fields_matched"] = total_matched
        aggregate["overall_match_rate"] = total_matched / total_fields if total_fields > 0 else 0.0

    aggregate["section_scores"] = {s: sum(v)/len(v) for s, v in section_stats.items() if v}

    return results, aggregate


# =============================================================================
# Report Printing
# =============================================================================

def print_report(results: List[Dict], aggregate: Dict):
    """Print a formatted ITC evaluation report."""
    print("\n" + "=" * 80)
    print("ITC STRUCTURED DESCRIPTION EVALUATION REPORT")
    print("=" * 80)

    print(f"\nTotal entries: {aggregate['total_entries']}")
    print(f"Entries matched: {aggregate['entries_matched']}")
    print(f"Entries evaluated: {aggregate['entries_evaluated']}")

    if aggregate.get("mean_overall_score") is not None:
        print(f"\n--- OVERALL METRICS ---")
        print(f"Mean Overall Score:  {aggregate['mean_overall_score']:.2%}")
        print(f"Mean Critical Score: {aggregate['mean_critical_score']:.2%}")
        print(f"Mean Field Coverage: {aggregate['mean_field_coverage']:.2%}")

        print(f"\nTotal Fields Evaluated: {aggregate['total_fields_evaluated']}")
        print(f"Fields Matched:         {aggregate['total_fields_matched']}")
        print(f"Overall Match Rate:     {aggregate['overall_match_rate']:.2%}")

    if aggregate.get("section_scores"):
        print(f"\n--- PER-SECTION ACCURACY ---")
        for section in ["instrument", "protein", "ligand", "assay_conditions", "data_analysis"]:
            if section in aggregate["section_scores"]:
                print(f"  {section:20s}: {aggregate['section_scores'][section]:.2%}")

    # Collect and display field errors
    field_errors = defaultdict(list)
    for r in results:
        struct_eval = r.get("evaluation", {})
        if struct_eval.get("evaluation_status") != "comparable":
            continue

        field_details = struct_eval.get("field_details", {})
        for section, fields in field_details.items():
            for field_name, field_result in fields.items():
                if not field_result.get("is_match", True) and field_result.get("score", 1.0) < 0.7:
                    field_errors[f"{section}.{field_name}"].append({
                        "reactant_set_id": r.get("reactant_set_id"),
                        "pmid": r.get("pmid"),
                        "score": field_result.get("score", 0),
                        "gt": field_result.get("gt_value"),
                        "ext": field_result.get("ext_value"),
                        "is_critical": field_result.get("is_critical", False)
                    })

    if field_errors:
        print(f"\n--- FIELD MISMATCHES ---")
        sorted_errors = sorted(field_errors.items(), key=lambda x: len(x[1]), reverse=True)
        for field_path, errors in sorted_errors[:15]:
            critical_marker = " [CRITICAL]" if errors[0].get("is_critical") else ""
            print(f"\n{field_path}{critical_marker}: {len(errors)} mismatches")
            for err in errors[:3]:
                print(f"  PMID {err['pmid']}, ID {err['reactant_set_id']} (score: {err['score']:.2f})")
                gt_str = str(err['gt'])[:50] if err['gt'] else "None"
                ext_str = str(err['ext'])[:50] if err['ext'] else "None"
                print(f"    GT:  {gt_str}")
                print(f"    EXT: {ext_str}")

    print("\n" + "=" * 80)


def print_per_paper_summary(per_paper_data: Dict[str, Dict], id_key: str):
    """Print per-paper evaluation summary."""
    print("\n" + "=" * 80)
    print("PER-PAPER SUMMARY")
    print("=" * 80)
    print(f"\n{'Paper (PMID)':<15} {'Entries':<10} {'Overall':<12} {'Critical':<12} {'Match Rate':<12}")
    print("-" * 65)

    paper_results = []
    for paper_name, data in sorted(per_paper_data.items()):
        results, aggregate = evaluate_batch(data["gt"], data["ext"], id_key=id_key)

        n_entries = aggregate.get("entries_evaluated", 0)
        overall = aggregate.get("mean_overall_score", 0)
        critical = aggregate.get("mean_critical_score", 0)
        match_rate = aggregate.get("overall_match_rate", 0)

        paper_results.append({
            "paper": paper_name,
            "entries": n_entries,
            "overall_score": overall,
            "critical_score": critical,
            "match_rate": match_rate
        })

        print(f"{paper_name:<15} {n_entries:<10} {overall:<12.2%} {critical:<12.2%} {match_rate:<12.2%}")

    print("-" * 65)
    return paper_results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate ITC agent extraction (structured_description only)")
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to ground truth JSON file or directory"
    )
    parser.add_argument(
        "--extracted",
        type=str,
        required=True,
        help="Path to agent extraction results JSON file or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed results (JSON)"
    )
    parser.add_argument(
        "--id-key",
        type=str,
        default="reactant_set_id",
        help="Key for matching entries between datasets"
    )

    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    ext_path = Path(args.extracted)

    # Check if both are directories
    if gt_path.is_dir() and ext_path.is_dir():
        # Directory mode: evaluate all matching files
        all_gt_data, all_ext_data, per_paper_data = load_from_directory(gt_path, ext_path)

        if not all_gt_data:
            print("Error: No matching JSON files found in directories")
            return

        # Print per-paper summary
        paper_results = print_per_paper_summary(per_paper_data, args.id_key)

        # Run overall evaluation
        print("\n" + "=" * 80)
        print("OVERALL EVALUATION (ALL PAPERS COMBINED)")
        print("=" * 80)
        results, aggregate = evaluate_batch(all_gt_data, all_ext_data, id_key=args.id_key)
        aggregate["per_paper_results"] = paper_results

    elif gt_path.is_file() and ext_path.is_file():
        # Single file mode
        print(f"Loading ground truth from {args.ground_truth}...")
        gt_data = load_json_file(gt_path)
        print(f"  Loaded {len(gt_data)} entries")

        print(f"Loading extracted data from {args.extracted}...")
        ext_data = load_json_file(ext_path)
        print(f"  Loaded {len(ext_data)} entries")

        print("Running evaluation...")
        results, aggregate = evaluate_batch(gt_data, ext_data, id_key=args.id_key)
    else:
        print("Error: --ground-truth and --extracted must both be files or both be directories")
        return

    # Print report
    print_report(results, aggregate)

    # Save results
    if args.output:
        output_data = {
            "summary": aggregate,
            "individual_results": results
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()