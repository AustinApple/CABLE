"""
Test script for the two-step extraction pipeline.

This script tests:
1. Step 1 (extract_paragraph) independently
2. Step 2 (fill_structured_description) on pre-extracted paragraphs
3. Full two-step pipeline (extract_assay_description with two_step=True)
4. Backward compatibility (single-step mode)
"""
import sys
import json
import argparse
from pathlib import Path

# Add the parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.assay_extraction_agent_qwen import AssayExtractionAgentQwen


def test_step1_only(agent: AssayExtractionAgentQwen, pmid: str, description: str):
    """Test Step 1: Extract paragraph only."""
    print("\n" + "="*80)
    print("TEST: Step 1 Only (extract_paragraph)")
    print("="*80)

    result = agent.extract_paragraph(
        pmid=pmid,
        assay_description=description,
        max_pages=None,
        max_new_tokens=2048
    )

    print("\nStep 1 Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def test_step2_only(agent: AssayExtractionAgentQwen, extracted_paragraph: dict,
                    description: str, protein: str = None, ligand_smiles: str = None,
                    affinity_data: dict = None):
    """Test Step 2: Fill structured_description from text."""
    print("\n" + "="*80)
    print("TEST: Step 2 Only (fill_structured_description)")
    print("="*80)

    result = agent.fill_structured_description(
        extracted_paragraph=extracted_paragraph,
        assay_description=description,
        protein=protein,
        ligand_smiles=ligand_smiles,
        affinity_data=affinity_data,
        max_new_tokens=2048
    )

    print("\nStep 2 Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def test_full_two_step(agent: AssayExtractionAgentQwen, pmid: str, description: str,
                       protein: str = None, ligand_smiles: str = None,
                       affinity_data: dict = None):
    """Test full two-step pipeline."""
    print("\n" + "="*80)
    print("TEST: Full Two-Step Pipeline")
    print("="*80)

    result = agent.extract_assay_description(
        pmid=pmid,
        assay_description=description,
        protein=protein,
        ligand_smiles=ligand_smiles,
        affinity_data=affinity_data,
        max_pages=None,
        max_new_tokens=4096,
        two_step=True  # Force two-step mode
    )

    print("\nFull Two-Step Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def test_single_step(agent: AssayExtractionAgentQwen, pmid: str, description: str,
                     protein: str = None, ligand_smiles: str = None,
                     affinity_data: dict = None):
    """Test single-step (original) mode for backward compatibility."""
    print("\n" + "="*80)
    print("TEST: Single-Step Mode (Backward Compatibility)")
    print("="*80)

    result = agent.extract_assay_description(
        pmid=pmid,
        assay_description=description,
        protein=protein,
        ligand_smiles=ligand_smiles,
        affinity_data=affinity_data,
        max_pages=None,
        max_new_tokens=4096,
        two_step=False  # Force single-step mode
    )

    print("\nSingle-Step Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def test_step2_on_existing_results(agent: AssayExtractionAgentQwen, results_file: str):
    """Test Step 2 on pre-extracted paragraphs from existing results."""
    print("\n" + "="*80)
    print(f"TEST: Step 2 on existing results from {results_file}")
    print("="*80)

    with open(results_file, 'r') as f:
        existing_results = json.load(f)

    for entry_id, entry in existing_results.items():
        print(f"\n--- Entry {entry_id} ---")
        original_paragraph = entry.get("original_paragraph", {})
        if not original_paragraph or original_paragraph == {}:
            print("  Skipping: No original_paragraph found")
            continue

        result = agent.fill_structured_description(
            extracted_paragraph=original_paragraph,
            assay_description=entry.get("DESCRIPTION", ""),
            protein=entry.get("protein"),
            ligand_smiles=entry.get("ligand", {}).get("smiles") if entry.get("ligand") else None,
            affinity_data=entry.get("affinity_data"),
            max_new_tokens=2048
        )

        print(f"  structured_description: {json.dumps(result.get('structured_description'), indent=4)[:500]}...")


def main():
    parser = argparse.ArgumentParser(description="Test two-step extraction pipeline")
    parser.add_argument("--test", choices=["step1", "step2", "full", "single", "existing", "all"],
                        default="full", help="Which test to run")
    parser.add_argument("--pmid", type=str, default="21513882",
                        help="PMID to test with")
    parser.add_argument("--description", type=str,
                        default="Binding of inhibitors and substrates to immobilized AcrB",
                        help="Assay description")
    parser.add_argument("--protein", type=str, default="AcrB",
                        help="Protein name")
    parser.add_argument("--existing-results", type=str,
                        default="spr_extraction_results/21513882.json",
                        help="Path to existing results JSON for step2 testing")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Vision model name")
    parser.add_argument("--text-model", type=str, default=None,
                        help="Separate text model for Step 2 (optional)")
    parser.add_argument("--two-step-mode", action="store_true",
                        help="Initialize agent with two_step_mode=True")
    parser.add_argument("--pdf-dir", type=str, default="./downloaded_paper_kd",
                        help="Directory containing PDFs")

    args = parser.parse_args()

    print("="*80)
    print("Initializing Two-Step Extraction Agent")
    print("="*80)
    print(f"Vision Model: {args.model}")
    print(f"Text Model: {args.text_model or 'Same as vision model'}")
    print(f"Two-Step Mode: {args.two_step_mode}")
    print(f"PDF Directory: {args.pdf_dir}")

    agent = AssayExtractionAgentQwen(
        model_name=args.model,
        text_model_name=args.text_model,
        pdf_dir=args.pdf_dir,
        two_step_mode=args.two_step_mode,
        search_supplementary=True,
        search_references=False  # Disable for faster testing
    )

    affinity_data = {"type": "Kd", "value": 110000, "relation": "=", "unit": "nM"}

    if args.test in ["step1", "all"]:
        test_step1_only(agent, args.pmid, args.description)

    if args.test in ["step2", "all"]:
        # First extract paragraph, then test step2
        step1_result = agent.extract_paragraph(args.pmid, args.description)
        if agent._is_paragraph_found(step1_result.get("original_paragraph")):
            test_step2_only(
                agent,
                step1_result.get("original_paragraph"),
                args.description,
                protein=args.protein,
                affinity_data=affinity_data
            )
        else:
            print("Step 1 didn't find paragraph, skipping Step 2 test")

    if args.test in ["full", "all"]:
        test_full_two_step(
            agent, args.pmid, args.description,
            protein=args.protein,
            affinity_data=affinity_data
        )

    if args.test in ["single", "all"]:
        test_single_step(
            agent, args.pmid, args.description,
            protein=args.protein,
            affinity_data=affinity_data
        )

    if args.test == "existing":
        test_step2_on_existing_results(agent, args.existing_results)

    # Print token usage summary
    print("\n" + "="*80)
    print("TOKEN USAGE SUMMARY")
    print("="*80)
    print(f"Total Requests: {agent.token_usage['requests']}")
    print(f"Total Input Tokens: {agent.token_usage['total_input_tokens']:,}")
    print(f"Total Output Tokens: {agent.token_usage['total_output_tokens']:,}")
    print(f"Total Tokens: {agent.token_usage['total_tokens']:,}")

    # Cleanup
    agent.cleanup()


if __name__ == "__main__":
    main()
