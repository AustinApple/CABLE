"""
Assay Extraction Agent - Qwen3-VL Version (Single-Step)

Extracts structured SPR assay descriptions from papers in a single model call.
The model sees the paper images and extracts both original_paragraph and
structured_description at once.

Features:
- Automatically fetches and searches supplementary information from the internet
- Detects references to previous literature and can fetch those papers via PubMed
- Tracks token usage for analysis
- Batched extraction for multiple pairs from a single paper
"""
from typing import Optional, Dict

from .base_extraction_agent import BaseAssayExtractionAgent
from .batch_processing import BatchProcessingMixin
from .prompts import get_structured_assay_extraction_prompt


class AssayExtractionAgentQwen(BatchProcessingMixin, BaseAssayExtractionAgent):
    """Agent to extract assay descriptions from scientific papers using Qwen3-VL.

    This is the single-step agent that extracts both original_paragraph and
    structured_description in one model call.

    Features:
    - Automatic supplementary material fetching from PubMed Central
    - Referenced literature fetching via PMID extraction
    - Multi-document search (main paper + supplementary + references)
    - Batched extraction for efficiency
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        pdf_dir: str = "./downloaded_paper_kd",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        temperature: float = 0.0,
        search_supplementary: bool = True,
        search_references: bool = True,
        max_reference_depth: int = 1,
        ncbi_api_key: Optional[str] = None
    ):
        """
        Initialize the single-step extraction agent.

        Args:
            model_name: Hugging Face model name for Qwen3-VL
            pdf_dir: Directory containing PDF files
            device: Device to run the model on ('cuda' or 'cpu')
            torch_dtype: Data type for model weights ('bfloat16', 'float16', or 'float32')
            temperature: Sampling temperature (0.0 = greedy/deterministic, >0 = more random)
            search_supplementary: Whether to automatically fetch and search supplementary materials
            search_references: Whether to fetch referenced papers if assay not found
            max_reference_depth: Maximum depth for recursive reference search (1 = only direct refs)
            ncbi_api_key: NCBI API key for higher rate limits (optional)
        """
        super().__init__(
            model_name=model_name,
            pdf_dir=pdf_dir,
            device=device,
            torch_dtype=torch_dtype,
            temperature=temperature,
            search_supplementary=search_supplementary,
            search_references=search_references,
            max_reference_depth=max_reference_depth,
            ncbi_api_key=ncbi_api_key
        )

    # ==================== Reference Handling ====================

    def _check_and_fetch_references(
        self,
        result: Dict,
        pmid: str,
        assay_description: str,
        max_pages: Optional[int],
        max_new_tokens: int,
        _depth: int,
        protein: Optional[str] = None,
        ligand_smiles: Optional[str] = None,
        affinity_data: Optional[Dict] = None
    ) -> Dict:
        """
        Check if a found result references previous work and fetch/combine if needed.
        """
        if not self.search_references or _depth >= self.max_reference_depth:
            return result

        references_previous = result.get("references_previous", "none")
        if references_previous == "none" or not references_previous:
            return result

        print(f"  Found reference to previous work: {references_previous}...")

        ref_pmids = self.citation_searcher.extract_referenced_pmids(references_previous)
        if result.get("original_paragraph"):
            ref_pmids.extend(self.citation_searcher.extract_referenced_pmids(result.get("original_paragraph", "")))
        ref_pmids = list(set(ref_pmids))

        if not ref_pmids:
            print(f"  No direct PMIDs found, trying citation-based search...")
            all_text = f"{references_previous} {result.get('original_paragraph', '')}"
            citations = self.citation_searcher.extract_citations_from_text(all_text)

            if references_previous and len(references_previous) > 20:
                citations.append(references_previous)

            for citation in citations[:3]:
                ref_pmid = self.citation_searcher.search_pmid_by_citation(citation)
                if ref_pmid and ref_pmid != pmid:
                    ref_pmids.append(ref_pmid)

            if not ref_pmids:
                print(f"  Could not find PMIDs from citations either")
                return result

        for ref_pmid in ref_pmids[:3]:
            if ref_pmid == pmid:
                continue

            print(f"  Fetching referenced paper PMID {ref_pmid}...")
            ref_result = self.extract_assay_description(
                ref_pmid,
                assay_description,
                protein=protein,
                ligand_smiles=ligand_smiles,
                affinity_data=affinity_data,
                max_pages=max_pages,
                max_new_tokens=max_new_tokens,
                _depth=_depth + 1
            )

            ref_paragraph = ref_result.get("original_paragraph")
            if self._is_paragraph_found(ref_paragraph):
                original_paragraph = result.get("original_paragraph", {})
                combined_paragraph = {}
                if isinstance(original_paragraph, dict):
                    for loc, content in original_paragraph.items():
                        combined_paragraph[f"[PMID {pmid}] {loc}"] = content
                elif original_paragraph:
                    combined_paragraph[f"[PMID {pmid}] extracted"] = original_paragraph
                if isinstance(ref_paragraph, dict):
                    for loc, content in ref_paragraph.items():
                        combined_paragraph[f"[PMID {ref_pmid}] {loc}"] = content
                elif ref_paragraph:
                    combined_paragraph[f"[PMID {ref_pmid}] extracted"] = ref_paragraph

                result["original_paragraph"] = combined_paragraph
                result["source"] = f"{result.get('source', '')} + referenced_paper_{ref_pmid}"

                result["structured_description"] = self._merge_structured_descriptions(
                    result.get("structured_description"),
                    ref_result.get("structured_description")
                )

                current_path = result.get("search_path", ["main"])
                ref_search_path = ref_result.get("search_path", ["main"])
                combined_path = list(current_path) if isinstance(current_path, list) else ["main"]
                combined_path.append("reference")
                if isinstance(ref_search_path, list) and len(ref_search_path) > 1:
                    combined_path.extend(ref_search_path[1:])
                result["search_path"] = combined_path

                supp_sources = []
                if isinstance(current_path, list) and "supplementary" in current_path:
                    supp_sources.append("main")
                if isinstance(ref_search_path, list) and "supplementary" in ref_search_path:
                    supp_sources.append("reference")
                result["supplementary_source"] = supp_sources

                print(f"  Combined with referenced paper PMID {ref_pmid}!")
                return result

        return result

    # ==================== Main Extraction Method ====================

    def extract_assay_description(
        self,
        pmid: str,
        assay_description: str,
        protein: Optional[str] = None,
        ligand_smiles: Optional[str] = None,
        affinity_data: Optional[Dict] = None,
        max_pages: Optional[int] = None,
        max_new_tokens: int = 4096,
        _depth: int = 0
    ) -> Dict:
        """
        Extract structured SPR assay information from a paper (single-step).

        This method extracts both original_paragraph and structured_description
        in a single model call.

        Args:
            pmid: PubMed ID of the paper
            assay_description: Brief assay description from BindingDB
            protein: Name of the protein target
            ligand_smiles: SMILES string for the ligand
            affinity_data: Dict with keys: type, value, relation, unit
            max_pages: Maximum number of pages to process (None for all)
            max_new_tokens: Maximum tokens to generate
            _depth: Internal parameter for tracking reference search depth

        Returns:
            Dictionary with 'original_paragraph', 'structured_description',
            'confidence', and 'source' keys
        """
        print(f"\n{'='*60}")
        print(f"Searching for assay in PMID {pmid} (depth={_depth})")
        print(f"{'='*60}")

        images, pdf_path = self._get_paper_images(pmid, max_pages)

        if pdf_path is None:
            return self._error_result(pmid, assay_description, "PDF not found and could not be fetched")

        if images is None or len(images) == 0:
            return self._error_result(pmid, assay_description, "PDF conversion failed")

        prompt = get_structured_assay_extraction_prompt(
            assay_description,
            protein=protein,
            ligand_smiles=ligand_smiles,
            affinity_data=affinity_data
        )
        supp_files = []

        try:
            # ========== STEP 1: Search main paper ==========
            print(f"[Step 1] Searching main paper for PMID {pmid}...")
            response_text, _, _ = self._query_model(images, prompt, max_new_tokens)

            print(f"\n[DEBUG] Raw response: {response_text[:500]}...")

            result = self._parse_response(response_text)

            if result and self._is_paragraph_found(result.get("original_paragraph")):
                result["pmid"] = pmid
                result["assay_description"] = assay_description
                result["source"] = f"main_paper_{pmid}"
                result["search_path"] = ["main"]
                result["supplementary_source"] = []
                print(f"  Found in main paper!")

                result = self._check_and_fetch_references(
                    result, pmid, assay_description, max_pages, max_new_tokens, _depth,
                    protein=protein, ligand_smiles=ligand_smiles, affinity_data=affinity_data
                )
                return result

            # ========== STEP 2: Search supplementary materials ==========
            if self.search_supplementary:
                print(f"\n[Step 2] Not found in main paper. Fetching supplementary materials...")
                supp_files = self.pubmed_fetcher.fetch_supplementary_from_pmc(pmid)

                for supp_path in supp_files:
                    supp_images = self.doc_converter.file_to_images(supp_path, max_pages=max_pages, label="supplementary")
                    if supp_images and len(supp_images) > 0:
                        print(f"  Searching supplementary: {supp_path.name}...")
                        supp_response, _, _ = self._query_model(supp_images, prompt, max_new_tokens)
                        supp_result = self._parse_response(supp_response)

                        if supp_result and self._is_paragraph_found(supp_result.get("original_paragraph")):
                            supp_result["pmid"] = pmid
                            supp_result["assay_description"] = assay_description
                            supp_result["source"] = f"supplementary_{supp_path.name}"
                            supp_result["location"] = f"Supplementary: {supp_result.get('location', 'N/A')}"
                            supp_result["search_path"] = ["main", "supplementary"]
                            supp_result["supplementary_source"] = ["main"]
                            print(f"  Found in supplementary: {supp_path.name}!")

                            supp_result = self._check_and_fetch_references(
                                supp_result, pmid, assay_description, max_pages, max_new_tokens, _depth,
                                protein=protein, ligand_smiles=ligand_smiles, affinity_data=affinity_data
                            )
                            return supp_result

            # ========== STEP 3: Check for referenced literature ==========
            if self.search_references and _depth < self.max_reference_depth:
                print(f"\n[Step 3] Not found in main paper or supplementary. Checking for referenced literature...")

                if result and result.get("references_previous") and result.get("references_previous") != "none":
                    ref_info = result.get("references_previous", "")
                    print(f"  Paper references previous literature: {ref_info[:200]}...")

                    ref_pmids = self.citation_searcher.extract_referenced_pmids(ref_info)

                    if result.get("original_paragraph"):
                        ref_pmids.extend(self.citation_searcher.extract_referenced_pmids(result.get("original_paragraph", "")))

                    ref_pmids = list(set(ref_pmids))

                    for ref_pmid in ref_pmids[:3]:
                        if ref_pmid != pmid:
                            print(f"  Searching referenced paper PMID {ref_pmid}...")
                            ref_result = self.extract_assay_description(
                                ref_pmid,
                                assay_description,
                                protein=protein,
                                ligand_smiles=ligand_smiles,
                                affinity_data=affinity_data,
                                max_pages=max_pages,
                                max_new_tokens=max_new_tokens,
                                _depth=_depth + 1
                            )

                            if self._is_paragraph_found(ref_result.get("original_paragraph")):
                                ref_result["source"] = f"referenced_paper_{ref_pmid}_from_{pmid}"
                                if result:
                                    ref_result["structured_description"] = self._merge_structured_descriptions(
                                        result.get("structured_description"),
                                        ref_result.get("structured_description")
                                    )
                                ref_search_path = ref_result.get("search_path", ["main"])
                                combined_path = ["main", "reference"]
                                if isinstance(ref_search_path, list) and len(ref_search_path) > 1:
                                    combined_path.extend(ref_search_path[1:])
                                ref_result["search_path"] = combined_path
                                if "supplementary" in combined_path:
                                    ref_result["supplementary_source"] = ["reference"]
                                else:
                                    ref_result["supplementary_source"] = []
                                print(f"  Found in referenced paper PMID {ref_pmid}!")
                                return ref_result

            # ========== STEP 4: Return NOT FOUND ==========
            print(f"  Assay description not found in any searched documents.")
            return self._not_found_result(pmid, assay_description, len(supp_files))

        except Exception as e:
            print(f"Error extracting assay description for PMID {pmid}: {e}")
            import traceback
            traceback.print_exc()
            return self._error_result(pmid, assay_description, str(e))

    # ==================== Result Helpers ====================

    @staticmethod
    def _error_result(pmid: str, assay_description: str, error_msg: str) -> Dict:
        """Create an error result dict."""
        return {
            "pmid": pmid,
            "assay_description": assay_description,
            "original_paragraph": {"error": error_msg},
            "confidence": "N/A",
            "source": "error" if "error" in error_msg.lower() or "failed" in error_msg.lower() else "N/A",
            "search_path": [],
            "supplementary_source": [],
            "structured_description": None
        }

    @staticmethod
    def _not_found_result(pmid: str, assay_description: str, supp_count: int) -> Dict:
        """Create a not-found result dict."""
        return {
            "pmid": pmid,
            "assay_description": assay_description,
            "original_paragraph": {},
            "confidence": "N/A",
            "source": "not_found",
            "search_path": [],
            "supplementary_source": [],
            "searched_locations": f"main_paper, supplementary({supp_count} files)",
            "structured_description": None
        }
