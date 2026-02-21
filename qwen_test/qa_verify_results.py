#%%
"""
QA Verification for extraction results.

Standalone script that reads existing extraction result JSON files and
verifies whether structured_description faithfully represents original_paragraph.

Usage:
    # Verify all SPR results
    python qwen_test/qa_verify_results.py \
        --results-dir qwen_test/spr_extraction_results_two_step_bindingdb \
        --output-dir qwen_test/spr_qa_results_two_step_bindingdb \
        --assay-type spr

    # Verify all ITC results
    python qwen_test/qa_verify_results.py \
        --results-dir qwen_test/itc_extraction_results_two_step_bindingdb \
        --output-dir qwen_test/itc_qa_results_two_step_bindingdb \
        --assay-type itc

    # Verify a single file
    python qwen_test/qa_verify_results.py \
        --results-file qwen_test/spr_extraction_results_two_step_bindingdb/16175541.json \
        --assay-type spr
"""

import argparse
import json
from pathlib import Path

from agent.qa_verification_agent import QAVerificationAgent


def main():
    parser = argparse.ArgumentParser(
        description="QA verification of extraction results"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--results-dir",
        type=str,
        help="Directory containing extraction result JSON files",
    )
    group.add_argument(
        "--results-file",
        type=str,
        help="Single extraction result JSON file to verify",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save QA-annotated results (default: parallel *_qa_* dir)",
    )
    parser.add_argument(
        "--assay-type",
        type=str,
        choices=["spr", "itc"],
        required=True,
        help="Assay type (spr or itc)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-30B-A3B",
        help="Model name for QA verification (default: Qwen/Qwen3-30B-A3B)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device (default: cuda:0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6, thinking models need > 0)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-run QA even if output already exists",
    )

    args = parser.parse_args()

    # Initialize agent
    agent = QAVerificationAgent(
        model_name=args.model,
        device=args.device,
        temperature=args.temperature,
        assay_type=args.assay_type,
    )

    if args.results_file:
        # Single file mode
        results_file = Path(args.results_file)
        if args.output_dir:
            output_path = Path(args.output_dir) / results_file.name
        else:
            # Default: save alongside with _qa suffix
            output_path = results_file.parent / f"{results_file.stem}_qa{results_file.suffix}"

        agent.verify_result_file(results_file, output_path=output_path)

    else:
        # Directory mode
        results_dir = Path(args.results_dir)

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # Default: replace 'extraction_results' with 'qa_results' in dir name
            dir_name = results_dir.name
            if "extraction_results" in dir_name:
                qa_dir_name = dir_name.replace("extraction_results", "qa_results")
            else:
                qa_dir_name = dir_name + "_qa"
            output_dir = results_dir.parent / qa_dir_name

        saved_files = agent.verify_directory(
            results_dir=results_dir,
            output_dir=output_dir,
            skip_existing=not args.no_skip_existing,
        )

        # Print summary
        print(f"\n{'='*80}")
        print(f"QA VERIFICATION SUMMARY")
        print(f"{'='*80}")
        print(f"Files processed: {len(saved_files)}")
        print(f"Output directory: {output_dir}")

        # Compute aggregate statistics across all output files
        total_entries = 0
        total_verified = 0
        total_skipped = 0
        score_sum = 0.0
        verdict_totals = {"correct": 0, "incorrect": 0, "unsupported": 0, "missing": 0}

        for filename, filepath in saved_files.items():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for entry in data.values():
                    total_entries += 1
                    qa = entry.get("qa_verification", {})
                    if qa.get("skipped"):
                        total_skipped += 1
                    elif "overall_score" in qa:
                        total_verified += 1
                        score_sum += qa["overall_score"]
                        counts = qa.get("counts", {})
                        for k in verdict_totals:
                            verdict_totals[k] += counts.get(k, 0)
            except (json.JSONDecodeError, Exception) as e:
                print(f"  Warning: Could not read {filepath}: {e}")

        print(f"\nTotal entries: {total_entries}")
        print(f"Verified: {total_verified}")
        print(f"Skipped: {total_skipped}")
        if total_verified > 0:
            avg_score = score_sum / total_verified
            print(f"Average QA score: {avg_score:.2f}/10")
            print(f"Verdict totals: {verdict_totals}")
        print(f"{'='*80}")

    agent.cleanup()


if __name__ == "__main__":
    main()
