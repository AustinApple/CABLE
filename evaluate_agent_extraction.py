#!/usr/bin/env python3
"""
Evaluation script to compare agent-extracted structured_description with ground truth.

This script evaluates the performance of the assay extraction agent by comparing
the extracted structured_description fields against manually curated ground truth.

Metrics computed:
1. Per-field accuracy (string fuzzy match, numeric tolerance, enum exact match)
2. Per-section weighted scores
3. Critical field accuracy
4. Overall field coverage and match rate
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import difflib


# =============================================================================
# Field Configuration
# =============================================================================

# Field configuration for structured_description comparison
# Each field has: type (string/numeric/enum/nested), weight, critical flag, and type-specific params
FIELD_CONFIG = {
    "instrument": {
        "manufacturer": {"type": "string", "weight": 1.0},
        "model": {"type": "string", "weight": 2.0, "critical": True}
    },
    "sensor_chip": {
        "type": {"type": "string", "weight": 1.5},
        "manufacturer": {"type": "string", "weight": 1.0}
    },
    "immobilization": {
        "ligand_name": {"type": "string", "weight": 1.5},
        "strategy": {"type": "string", "weight": 1.5},
        "density_ru": {"type": "numeric_string", "weight": 1.0, "tolerance": 0.15},
        "concentration_for_immobilization": {"type": "string", "weight": 0.5}
    },
    "analyte": {
        "description": {"type": "string", "weight": 1.0},
        "concentration_range": {"type": "string", "weight": 1.0}
    },
    "assay_conditions": {
        "running_buffer_composition": {"type": "string", "weight": 1.0},
        "pH": {"type": "numeric", "weight": 1.0, "tolerance": 0.5},
        "assay_type": {"type": "enum", "weight": 2.0, "critical": True},
        "assay_flow_rate": {"type": "nested", "weight": 1.0},
        "temperature_c": {"type": "string", "weight": 0.5},
        "association_time_s": {"type": "numeric_string", "weight": 1.5, "tolerance": 0.1, "critical": True},
        "dissociation_time_s": {"type": "numeric_string", "weight": 1.5, "tolerance": 0.1, "critical": True},
        "regeneration_solution": {"type": "string", "weight": 0.5}
    },
    "data_analysis": {
        "reference_description": {"type": "string", "weight": 1.0},
        "subtraction_method": {"type": "string", "weight": 1.0},
        "fitting_model": {"type": "string", "weight": 1.5, "critical": True},
        "software": {"type": "string", "weight": 0.5}
    }
}


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_string_for_comparison(text: str) -> str:
    """Normalize a string for fuzzy comparison."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace('μ', 'u').replace('µ', 'u')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_numeric_from_string(value) -> Optional[float]:
    """Extract a numeric value from a string or return the value if already numeric."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r'[-+]?[\d,]+\.?\d*', value.replace(',', ''))
        if match:
            try:
                return float(match.group().replace(',', ''))
            except ValueError:
                return None
    return None


def compare_string_fields(gt: str, extracted: str, threshold: float = 0.7) -> Tuple[float, bool]:
    """Compare two string fields using fuzzy matching."""
    if gt is None and extracted is None:
        return 1.0, True
    if gt is None or extracted is None:
        return 0.0, False

    gt_norm = normalize_string_for_comparison(str(gt))
    ext_norm = normalize_string_for_comparison(str(extracted))

    if not gt_norm and not ext_norm:
        return 1.0, True
    if not gt_norm or not ext_norm:
        return 0.0, False

    similarity = difflib.SequenceMatcher(None, gt_norm, ext_norm).ratio()

    if gt_norm in ext_norm or ext_norm in gt_norm:
        similarity = max(similarity, 0.85)

    return similarity, similarity >= threshold


def compare_numeric_fields(gt, extracted, tolerance: float = 0.1) -> Tuple[float, bool]:
    """Compare two numeric fields with tolerance."""
    gt_num = parse_numeric_from_string(gt)
    ext_num = parse_numeric_from_string(extracted)

    if gt_num is None and ext_num is None:
        return 1.0, True
    if gt_num is None or ext_num is None:
        return 0.0, False

    if gt_num == 0:
        if ext_num == 0:
            return 1.0, True
        return 0.0, False

    relative_error = abs(gt_num - ext_num) / abs(gt_num)

    if relative_error <= tolerance:
        accuracy = 1.0 - (relative_error / tolerance) * 0.2
    else:
        accuracy = max(0.0, 1.0 - relative_error)

    return accuracy, relative_error <= tolerance


def compare_enum_fields(gt: str, extracted: str) -> Tuple[float, bool]:
    """Compare enum fields (exact match after normalization)."""
    if gt is None and extracted is None:
        return 1.0, True
    if gt is None or extracted is None:
        return 0.0, False

    gt_norm = normalize_string_for_comparison(str(gt))
    ext_norm = normalize_string_for_comparison(str(extracted))

    is_match = gt_norm == ext_norm
    return (1.0 if is_match else 0.0), is_match


def compare_nested_object(gt: dict, extracted: dict) -> Tuple[float, bool]:
    """Compare nested objects (like assay_flow_rate with value and unit)."""
    if gt is None and extracted is None:
        return 1.0, True
    if gt is None or extracted is None:
        return 0.0, False
    if not isinstance(gt, dict) or not isinstance(extracted, dict):
        return compare_string_fields(str(gt), str(extracted))

    scores = []
    all_match = True

    for key in gt.keys():
        gt_val = gt.get(key)
        ext_val = extracted.get(key) if isinstance(extracted, dict) else None

        if key == "value":
            score, match = compare_numeric_fields(gt_val, ext_val, tolerance=0.1)
        else:
            score, match = compare_string_fields(str(gt_val) if gt_val else "",
                                                  str(ext_val) if ext_val else "")

        scores.append(score)
        if not match:
            all_match = False

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return avg_score, all_match


def evaluate_structured_field(gt_value, ext_value, field_config: dict) -> dict:
    """Evaluate a single structured field based on its configuration."""
    field_type = field_config.get("type", "string")
    tolerance = field_config.get("tolerance", 0.1)

    result = {
        "field_type": field_type,
        "gt_value": gt_value,
        "ext_value": ext_value,
        "score": 0.0,
        "is_match": False,
        "is_critical": field_config.get("critical", False),
        "weight": field_config.get("weight", 1.0)
    }

    if field_type == "string":
        score, is_match = compare_string_fields(gt_value, ext_value)
    elif field_type == "numeric":
        score, is_match = compare_numeric_fields(gt_value, ext_value, tolerance)
    elif field_type == "numeric_string":
        score, is_match = compare_numeric_fields(gt_value, ext_value, tolerance)
    elif field_type == "enum":
        score, is_match = compare_enum_fields(gt_value, ext_value)
    elif field_type == "nested":
        score, is_match = compare_nested_object(gt_value, ext_value)
    else:
        score, is_match = compare_string_fields(str(gt_value) if gt_value else "",
                                                 str(ext_value) if ext_value else "")

    result["score"] = score
    result["is_match"] = is_match

    return result


def evaluate_structured_description(gt_struct: Optional[dict], ext_struct: Optional[dict]) -> dict:
    """Evaluate structured_description field-by-field."""
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


# =============================================================================
# Batch Evaluation
# =============================================================================

def evaluate_batch(
    ground_truth_data: List[Dict],
    extracted_data: List[Dict],
    id_key: str = "reactant_set_id"
) -> Tuple[List[Dict], Dict]:
    """
    Evaluate a batch of extractions.

    Args:
        ground_truth_data: List of ground truth entries
        extracted_data: List of extracted entries
        id_key: Key for matching entries between datasets

    Returns:
        Tuple of (individual results, aggregate statistics)
    """
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
    """Print a formatted evaluation report."""
    print("\n" + "=" * 80)
    print("STRUCTURED DESCRIPTION EVALUATION REPORT")
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
        for section in ["instrument", "sensor_chip", "immobilization", "analyte", "assay_conditions", "data_analysis"]:
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


# =============================================================================
# Data Loading
# =============================================================================

def load_json_file(filepath: Path) -> List[Dict]:
    """Load a JSON file and convert to list format."""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        first_key = next(iter(raw.keys()), None)
        if first_key and isinstance(raw[first_key], dict):
            return list(raw.values())
        else:
            return [raw]
    else:
        return raw


def load_from_directory(gt_dir: Path, ext_dir: Path) -> Tuple[List[Dict], List[Dict], Dict[str, Dict]]:
    """
    Load all matching JSON files from ground truth and extracted directories.

    Returns:
        Tuple of (all_gt_data, all_ext_data, per_paper_data)
        per_paper_data is a dict mapping filename to {"gt": [...], "ext": [...]}
    """
    gt_files = {f.stem: f for f in gt_dir.glob("*.json")}
    ext_files = {f.stem: f for f in ext_dir.glob("*.json")}

    # Find matching files
    matching = set(gt_files.keys()) & set(ext_files.keys())
    gt_only = set(gt_files.keys()) - set(ext_files.keys())
    ext_only = set(ext_files.keys()) - set(gt_files.keys())

    print(f"\nDirectory mode:")
    print(f"  Ground truth directory: {gt_dir}")
    print(f"  Extracted directory:    {ext_dir}")
    print(f"  Matching files:         {len(matching)}")
    if gt_only:
        print(f"  GT only (no extraction): {len(gt_only)} - {list(gt_only)[:5]}{'...' if len(gt_only) > 5 else ''}")
    if ext_only:
        print(f"  Extracted only (no GT):  {len(ext_only)} - {list(ext_only)[:5]}{'...' if len(ext_only) > 5 else ''}")

    all_gt_data = []
    all_ext_data = []
    per_paper_data = {}

    for name in sorted(matching):
        gt_data = load_json_file(gt_files[name])
        ext_data = load_json_file(ext_files[name])

        all_gt_data.extend(gt_data)
        all_ext_data.extend(ext_data)
        per_paper_data[name] = {"gt": gt_data, "ext": ext_data}

    print(f"  Total GT entries:       {len(all_gt_data)}")
    print(f"  Total extracted entries: {len(all_ext_data)}")

    return all_gt_data, all_ext_data, per_paper_data


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
    parser = argparse.ArgumentParser(description="Evaluate agent extraction (structured_description only)")
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
