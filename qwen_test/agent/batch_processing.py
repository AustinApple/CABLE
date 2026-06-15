"""
Batch Processing Mixin for Assay Extraction Agents.

This module contains batch processing functionality that can be mixed into
extraction agent classes. It provides methods for processing DataFrames
and handling multiple extractions efficiently.
"""
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, List

import time

from .dataframe_utils import extract_row_data, build_pair_entry, build_metadata_entry
from .prompts import get_batched_structured_assay_extraction_prompt


class BatchProcessingMixin:
    """Mixin providing batch processing capabilities for extraction agents.

    This mixin assumes the following methods/attributes exist on the class:
    - extract_assay_description(pmid, assay_description, protein, ligand_smiles, affinity_data, max_pages)
    - _get_paper_images(pmid, max_pages)
    - _query_model(images, prompt, max_new_tokens)
    - _parse_response(response_text)
    - _is_paragraph_found(original_paragraph)
    - pubmed_fetcher
    - doc_converter
    - search_supplementary
    - token_usage
    """

    def _print_token_summary(self):
        """Print token usage summary."""
        print(f"\n{'='*80}")
        print("TOKEN USAGE SUMMARY")
        print(f"{'='*80}")
        print(f"Total Requests: {self.token_usage['requests']}")
        print(f"Total Input Tokens: {self.token_usage['total_input_tokens']:,}")
        print(f"Total Output Tokens: {self.token_usage['total_output_tokens']:,}")
        print(f"Total Tokens: {self.token_usage['total_tokens']:,}")
        if self.token_usage['requests'] > 0:
            print(f"Avg Input/Request: {self.token_usage['total_input_tokens'] / self.token_usage['requests']:,.0f}")
            print(f"Avg Output/Request: {self.token_usage['total_output_tokens'] / self.token_usage['requests']:,.0f}")
        print(f"{'='*80}")

    def process_dataframe(
        self,
        df: pd.DataFrame,
        pmid_col: str = "PMID",
        description_col: str = "DESCRIPTION",
        reactant_set_id_col: Optional[str] = None,
        protein_col: Optional[str] = None,
        ligand_smiles_col: Optional[str] = None,
        affinity_type_col: Optional[str] = None,
        affinity_value_col: Optional[str] = None,
        affinity_relation_col: Optional[str] = None,
        affinity_unit_col: Optional[str] = None,
        limit: Optional[int] = None,
        max_pages: Optional[int] = None,
        delay: float = 0.0
    ) -> pd.DataFrame:
        """
        Process a dataframe of assay descriptions.

        Args:
            df: DataFrame with PMID, DESCRIPTION, and optional context columns
            pmid_col: Name of PMID column
            description_col: Name of description column
            reactant_set_id_col: Name of reactant set ID column (optional, for logging)
            protein_col: Name of protein column (optional)
            ligand_smiles_col: Name of ligand SMILES column (optional)
            affinity_type_col: Name of affinity type column (optional, e.g., "Kd")
            affinity_value_col: Name of affinity value column (optional)
            affinity_relation_col: Name of affinity relation column (optional, e.g., "=")
            affinity_unit_col: Name of affinity unit column (optional, e.g., "nM")
            limit: Maximum number of rows to process (None for all)
            max_pages: Maximum pages per PDF to process
            delay: Delay between queries (useful for API rate limiting)

        Returns:
            DataFrame with added columns for extracted paragraphs and structured_description
        """
        results = []
        df_to_process = df.head(limit) if limit else df

        row_count = 0
        for _, row in df_to_process.iterrows():
            row_data = extract_row_data(
                row,
                pmid_col=pmid_col,
                description_col=description_col,
                reactant_set_id_col=reactant_set_id_col,
                protein_col=protein_col,
                ligand_smiles_col=ligand_smiles_col,
                affinity_type_col=affinity_type_col,
                affinity_value_col=affinity_value_col,
                affinity_relation_col=affinity_relation_col,
                affinity_unit_col=affinity_unit_col
            )

            row_count += 1
            print(f"\n{'='*80}")
            print(f"Processing row {row_count}/{len(df_to_process)}")
            if row_data["reactant_set_id"] is not None:
                print(f"Reactant Set ID: {row_data['reactant_set_id']}")
            print(f"PMID: {row_data['pmid']}")
            print(f"Description: {row_data['description'][:100]}...")
            if row_data["protein"]:
                print(f"Protein: {row_data['protein']}")
            if row_data["ligand_smiles"]:
                print(f"Ligand SMILES: {row_data['ligand_smiles'][:80]}...")

            result = self.extract_assay_description(
                row_data["pmid"],
                row_data["description"],
                protein=row_data["protein"],
                ligand_smiles=row_data["ligand_smiles"],
                affinity_data=row_data["affinity_data"],
                max_pages=max_pages
            )
            if row_data["reactant_set_id"] is not None:
                result["reactant_set_id"] = row_data["reactant_set_id"]
            results.append(result)

            if delay > 0 and row_count < len(df_to_process):
                time.sleep(delay)

        self._print_token_summary()

        df_result = df_to_process.copy()
        df_result["original_paragraph"] = [
            json.dumps(result.get("original_paragraph")) if result.get("original_paragraph") else "{}"
            for result in results
        ]
        df_result["extraction_confidence"] = [result.get("confidence", "N/A") for result in results]
        df_result["source"] = [result.get("source", "N/A") for result in results]
        df_result["search_path"] = [result.get("search_path", []) for result in results]
        df_result["supplementary_source"] = [result.get("supplementary_source", []) for result in results]
        df_result["references_previous"] = [result.get("references_previous", "N/A") for result in results]
        df_result["structured_description"] = [
            json.dumps(result.get("structured_description")) if result.get("structured_description") else "N/A"
            for result in results
        ]

        return df_result

    def process_dataframe_to_json(
        self,
        df: pd.DataFrame,
        output_dir: str,
        pmid_col: str = "PMID",
        description_col: str = "DESCRIPTION",
        reactant_set_id_col: Optional[str] = None,
        protein_col: Optional[str] = None,
        ligand_name_col: Optional[str] = None,
        ligand_smiles_col: Optional[str] = None,
        affinity_type_col: Optional[str] = None,
        affinity_value_col: Optional[str] = None,
        affinity_relation_col: Optional[str] = None,
        affinity_unit_col: Optional[str] = None,
        limit: Optional[int] = None,
        max_pages: Optional[int] = None,
        delay: float = 0.0
    ) -> Dict[str, Path]:
        """
        Process a dataframe and save results as JSON files per PMID.

        Returns:
            Dict mapping PMID -> Path of saved JSON file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        df_to_process = df.head(limit) if limit else df
        pmid_groups = df_to_process.groupby(pmid_col)

        saved_files = {}
        row_count = 0
        total_rows = len(df_to_process)

        for pmid, group in pmid_groups:
            pmid_str = str(int(pmid))
            pmid_results = {}

            for _, row in group.iterrows():
                row_count += 1
                row_data = extract_row_data(
                    row,
                    pmid_col=pmid_col,
                    description_col=description_col,
                    reactant_set_id_col=reactant_set_id_col,
                    protein_col=protein_col,
                    ligand_name_col=ligand_name_col,
                    ligand_smiles_col=ligand_smiles_col,
                    affinity_type_col=affinity_type_col,
                    affinity_value_col=affinity_value_col,
                    affinity_relation_col=affinity_relation_col,
                    affinity_unit_col=affinity_unit_col
                )

                print(f"\n{'='*80}")
                print(f"Processing row {row_count}/{total_rows}")
                if row_data["reactant_set_id"] is not None:
                    print(f"Reactant Set ID: {row_data['reactant_set_id']}")
                print(f"PMID: {pmid_str}")
                print(f"Description: {row_data['description'][:100]}...")

                result = self.extract_assay_description(
                    pmid_str,
                    row_data["description"],
                    protein=row_data["protein"],
                    ligand_smiles=row_data["ligand_smiles"],
                    affinity_data=row_data["affinity_data"],
                    max_pages=max_pages
                )

                entry = {
                    **build_metadata_entry(row_data, int(pmid)),
                    "search_path": result.get("search_path", "N/A"),
                    "supplementary_source": result.get("supplementary_source"),
                    "references_previous": result.get("references_previous"),
                    "original_paragraph": result.get("original_paragraph", {}),
                    "structured_description": result.get("structured_description")
                }

                key = str(row_data["reactant_set_id"]) if row_data["reactant_set_id"] else f"entry_{row_count}"
                pmid_results[key] = entry

                if delay > 0:
                    time.sleep(delay)

            json_path = output_path / f"{pmid_str}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(pmid_results, f, indent=4, ensure_ascii=False)
            print(f"\nSaved: {json_path}")
            saved_files[pmid_str] = json_path

        self._print_token_summary()
        print(f"\nSaved {len(saved_files)} JSON files to {output_dir}")

        return saved_files

    def extract_assay_descriptions_batched(
        self,
        pmid: str,
        pairs: List[Dict],
        max_pages: Optional[int] = None,
        max_new_tokens: int = 8192,
        _depth: int = 0
    ) -> Dict[str, Dict]:
        """
        Extract structured SPR assay info for multiple protein-ligand pairs from a single paper.
        Reads the paper ONCE and extracts info for all pairs in one model call.

        Args:
            pmid: PubMed ID of the paper
            pairs: List of dicts, each containing:
                - id: reactant_set_id as identifier
                - assay_description: Brief assay description
                - protein: Name of the protein target
                - ligand_smiles: SMILES string for the ligand (optional)
                - affinity_data: Dict with keys: type, value, relation, unit (optional)
            max_pages: Maximum number of pages to process
            max_new_tokens: Maximum tokens to generate (larger for batch output)
            _depth: Internal parameter for tracking reference search depth

        Returns:
            Dict mapping reactant_set_id -> extraction result
        """
        print(f"\n{'='*60}")
        print(f"Batched extraction for PMID {pmid} ({len(pairs)} pairs, depth={_depth})")
        print(f"{'='*60}")

        # Initialize default results
        results = {}
        for pair in pairs:
            pair_id = str(pair.get("id", "unknown"))
            results[pair_id] = {
                "pmid": pmid,
                "original_paragraph": {},
                "confidence": "N/A",
                "source": "N/A",
                "search_path": [],
                "supplementary_source": [],
                "structured_description": None
            }

        images, pdf_path = self._get_paper_images(pmid, max_pages)

        if pdf_path is None:
            for pair_id in results:
                results[pair_id]["original_paragraph"] = {"error": "PDF not found and could not be fetched"}
            return results

        if images is None or len(images) == 0:
            for pair_id in results:
                results[pair_id]["original_paragraph"] = {"error": "PDF conversion failed"}
            return results

        prompt = get_batched_structured_assay_extraction_prompt(pairs)

        try:
            print(f"[Step 1] Searching main paper for {len(pairs)} pairs...")
            response_text, _, _ = self._query_model(images, prompt, max_new_tokens)

            print(f"\n[DEBUG] Raw response (first 500 chars): {response_text[:500]}...")

            parsed_results = self._parse_response(response_text)

            if parsed_results and isinstance(parsed_results, dict):
                found_pairs = []
                missing_pairs = []

                for pair in pairs:
                    pair_id = str(pair.get("id", "unknown"))
                    if pair_id in parsed_results:
                        pair_result = parsed_results[pair_id]
                        if self._is_paragraph_found(pair_result.get("original_paragraph")):
                            pair_result["pmid"] = pmid
                            pair_result["source"] = f"main_paper_{pmid}"
                            pair_result["search_path"] = ["main"]
                            pair_result["supplementary_source"] = []
                            results[pair_id] = pair_result
                            found_pairs.append(pair_id)
                        else:
                            missing_pairs.append(pair)
                    else:
                        missing_pairs.append(pair)

                print(f"  Found {len(found_pairs)}/{len(pairs)} pairs in main paper")

                # Search supplementary for missing pairs
                if missing_pairs and self.search_supplementary:
                    print(f"\n[Step 2] Searching supplementary for {len(missing_pairs)} missing pairs...")
                    supp_files = self.pubmed_fetcher.fetch_supplementary_from_pmc(pmid)

                    for supp_path in supp_files:
                        if not missing_pairs:
                            break

                        supp_images = self.doc_converter.file_to_images(supp_path, max_pages=max_pages, label="supplementary")
                        if supp_images and len(supp_images) > 0:
                            print(f"  Searching supplementary: {supp_path.name} for {len(missing_pairs)} pairs...")
                            supp_prompt = get_batched_structured_assay_extraction_prompt(missing_pairs)
                            supp_response, _, _ = self._query_model(supp_images, supp_prompt, max_new_tokens)
                            supp_parsed = self._parse_response(supp_response)

                            if supp_parsed and isinstance(supp_parsed, dict):
                                newly_found = []
                                for pair in missing_pairs[:]:
                                    pair_id = str(pair.get("id", "unknown"))
                                    if pair_id in supp_parsed:
                                        pair_result = supp_parsed[pair_id]
                                        if self._is_paragraph_found(pair_result.get("original_paragraph")):
                                            pair_result["pmid"] = pmid
                                            pair_result["source"] = f"supplementary_{supp_path.name}"
                                            pair_result["search_path"] = ["main", "supplementary"]
                                            pair_result["supplementary_source"] = ["main"]
                                            results[pair_id] = pair_result
                                            newly_found.append(pair)

                                for pair in newly_found:
                                    missing_pairs.remove(pair)

                                if newly_found:
                                    print(f"    Found {len(newly_found)} pairs in {supp_path.name}")

                for pair in missing_pairs:
                    pair_id = str(pair.get("id", "unknown"))
                    results[pair_id]["source"] = "not_found"

            return results

        except Exception as e:
            print(f"Error in batched extraction for PMID {pmid}: {e}")
            import traceback
            traceback.print_exc()
            for pair_id in results:
                results[pair_id]["original_paragraph"] = {"error": str(e)}
                results[pair_id]["source"] = "error"
            return results

    def process_pmids_batched(
        self,
        df: pd.DataFrame,
        output_dir: str,
        pmid_col: str = "PMID",
        reactant_set_id_col: str = "reactant_set_id",
        description_col: str = "DESCRIPTION",
        protein_col: Optional[str] = None,
        ligand_name_col: Optional[str] = None,
        ligand_smiles_col: Optional[str] = None,
        affinity_type_col: Optional[str] = None,
        affinity_value_col: Optional[str] = None,
        affinity_relation_col: Optional[str] = None,
        affinity_unit_col: Optional[str] = None,
        limit: Optional[int] = None,
        max_pages: Optional[int] = None,
        max_new_tokens: int = 8192,
        delay: float = 0.0
    ) -> Dict[str, Path]:
        """
        Process dataframe with batched extraction - one model call per PMID.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        df_to_process = df.head(limit) if limit else df
        pmid_groups = df_to_process.groupby(pmid_col)

        saved_files = {}
        pmid_count = 0
        total_pmids = len(pmid_groups)
        total_pairs = len(df_to_process)

        print(f"\n{'='*80}")
        print(f"BATCHED EXTRACTION: {total_pairs} pairs across {total_pmids} PMIDs")
        print(f"{'='*80}")

        for pmid, group in pmid_groups:
            pmid_count += 1
            pmid_str = str(int(pmid))

            pairs = []
            pair_metadata = {}

            for _, row in group.iterrows():
                row_data = extract_row_data(
                    row,
                    pmid_col=pmid_col,
                    description_col=description_col,
                    reactant_set_id_col=reactant_set_id_col,
                    protein_col=protein_col,
                    ligand_name_col=ligand_name_col,
                    ligand_smiles_col=ligand_smiles_col,
                    affinity_type_col=affinity_type_col,
                    affinity_value_col=affinity_value_col,
                    affinity_relation_col=affinity_relation_col,
                    affinity_unit_col=affinity_unit_col
                )

                pair_id = str(int(row_data["reactant_set_id"])) if row_data["reactant_set_id"] else f"row_{len(pairs)}"

                pairs.append(build_pair_entry(pair_id, row_data, int(pmid)))
                pair_metadata[pair_id] = build_metadata_entry(row_data, int(pmid))

            print(f"\n{'='*80}")
            print(f"Processing PMID {pmid_count}/{total_pmids}: {pmid_str} ({len(pairs)} pairs)")
            print(f"{'='*80}")

            extraction_results = self.extract_assay_descriptions_batched(
                pmid_str,
                pairs,
                max_pages=max_pages,
                max_new_tokens=max_new_tokens
            )

            pmid_output = {}
            for pair_id, extraction in extraction_results.items():
                meta = pair_metadata.get(pair_id, {})
                entry = {
                    **meta,
                    "search_path": extraction.get("search_path", []),
                    "supplementary_source": extraction.get("supplementary_source", []),
                    "references_previous": extraction.get("references_previous"),
                    "original_paragraph": extraction.get("original_paragraph", {}),
                    "structured_description": extraction.get("structured_description")
                }
                pmid_output[pair_id] = entry

            json_path = output_path / f"{pmid_str}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(pmid_output, f, indent=4, ensure_ascii=False)
            print(f"\nSaved: {json_path}")
            saved_files[pmid_str] = json_path

            if delay > 0 and pmid_count < total_pmids:
                time.sleep(delay)

        self._print_token_summary()
        print(f"\nProcessed {total_pairs} pairs across {total_pmids} PMIDs")
        print(f"Saved {len(saved_files)} JSON files to {output_dir}")

        return saved_files
