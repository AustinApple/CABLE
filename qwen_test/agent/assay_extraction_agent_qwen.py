"""
Assay Extraction Agent - Qwen3-VL Version
Using Qwen3-VL to extract complete assay descriptions from papers

Features:
- Automatically fetches and searches supplementary information from the internet
- Detects references to previous literature and can fetch those papers via PubMed
- Tracks token usage for analysis
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import time
from PIL import Image
import os

from .pubmed_utils import PubMedFetcher, CitationSearcher
from .pdf_utils import DocumentConverter
from .prompts import get_structured_assay_extraction_prompt, get_batched_structured_assay_extraction_prompt


class AssayExtractionAgentQwen:
    """Agent to extract assay descriptions from scientific papers using Qwen3-VL

    Features:
    - Automatic supplementary material fetching from PubMed Central
    - Referenced literature fetching via PMID extraction
    - Multi-document search (main paper + supplementary + references)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        pdf_dir: str = "./downloaded_paper_kd",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: str = "bfloat16",
        temperature: float = 0.0,
        search_supplementary: bool = True,
        search_references: bool = True,
        max_reference_depth: int = 1,
        ncbi_api_key: Optional[str] = None
    ):
        """
        Initialize the agent

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
        print(f"Loading Qwen3-VL model: {model_name}")
        print(f"Device: {device}")
        print(f"Torch dtype: {torch_dtype}")
        print(f"Temperature: {temperature}")
        print(f"Search supplementary: {search_supplementary}")
        print(f"Search references: {search_references}")

        self.temperature = temperature

        # Map string dtype to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        torch_dtype_obj = dtype_map.get(torch_dtype, torch.bfloat16)

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch_dtype_obj,
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.device = device
        self.pdf_dir = Path(pdf_dir)
        self.search_supplementary = search_supplementary
        self.search_references = search_references
        self.max_reference_depth = max_reference_depth
        self.ncbi_api_key = ncbi_api_key or os.environ.get("NCBI_API_KEY")

        # Create supplementary and reference cache directories
        self.supp_dir = self.pdf_dir / "supplementary"
        self.ref_dir = self.pdf_dir / "references"
        self.supp_dir.mkdir(exist_ok=True)
        self.ref_dir.mkdir(exist_ok=True)

        # Initialize utility classes
        self.pubmed_fetcher = PubMedFetcher(
            supp_dir=self.supp_dir,
            ref_dir=self.ref_dir,
            pdf_dir=self.pdf_dir,
            ncbi_api_key=self.ncbi_api_key
        )
        self.citation_searcher = CitationSearcher(ncbi_api_key=self.ncbi_api_key, 
                                                  model=self.model, 
                                                  processor=self.processor, 
                                                  device=self.device)
        self.doc_converter = DocumentConverter()

        # Cache converted images per PMID to avoid re-converting PDFs
        self._image_cache: Dict[str, List[Image.Image]] = {}

        self.token_usage = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "requests": 0
        }

        print("Model loaded successfully!")

    # ==================== Model Query Methods ====================

    def _query_model(
        self,
        images: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 2048
    ) -> Tuple[str, int, int]:
        """
        Send a query to the model with images.

        Args:
            images: List of PIL images
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        content_parts = []

        for img in images:
            content_parts.append({"type": "image", "image": img})
        content_parts.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content_parts}]

        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt],
            images=[img for img in images],
            return_tensors="pt",
            padding=True
        )
        inputs = inputs.to(self.device)

        input_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            # Use greedy decoding if temperature is 0, otherwise sample with temperature
            if self.temperature == 0.0:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            else:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature
                )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        output_tokens = generated_ids_trimmed[0].shape[0]

        self.token_usage["total_input_tokens"] += input_tokens
        self.token_usage["total_output_tokens"] += output_tokens
        self.token_usage["total_tokens"] += input_tokens + output_tokens
        self.token_usage["requests"] += 1

        print(f"  [This Query] Input: {input_tokens:,} | Output: {output_tokens:,} | Total: {input_tokens + output_tokens:,}")
        print(f"  [Accumulated] Input: {self.token_usage['total_input_tokens']:,} | Output: {self.token_usage['total_output_tokens']:,} | Total: {self.token_usage['total_tokens']:,} | Requests: {self.token_usage['requests']}")

        return response_text, input_tokens, output_tokens

    def _parse_response(self, response_text: str) -> Optional[Dict]:
        """Parse JSON response from model output."""
        response_text = response_text.strip()

        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        try:
            result = json.loads(response_text)
            # Clean newlines from string values recursively
            result = self._clean_string_values(result)
            return result
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _clean_string_values(obj):
        """Recursively clean newlines from string values in nested dicts/lists."""
        if isinstance(obj, dict):
            return {k: AssayExtractionAgentQwen._clean_string_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [AssayExtractionAgentQwen._clean_string_values(item) for item in obj]
        elif isinstance(obj, str):
            return ' '.join(obj.split())
        return obj

    @staticmethod
    def _is_paragraph_found(original_paragraph) -> bool:
        """Check if original_paragraph contains valid extracted content.

        original_paragraph can be:
        - A dict mapping location -> content (new format)
        - A string (legacy format, for backward compatibility)
        - None or empty
        """
        if original_paragraph is None:
            return False
        if isinstance(original_paragraph, dict):
            # Check if dict has any non-empty values
            return any(v and str(v).strip() and str(v).strip() != "NOT FOUND" for v in original_paragraph.values())
        if isinstance(original_paragraph, str):
            # Legacy string format
            return original_paragraph not in ["NOT FOUND", "", None] and not original_paragraph.startswith("ERROR")
        return False

    # ==================== Reference Handling ====================

    @staticmethod
    def _merge_structured_descriptions(main_desc: Optional[Dict], ref_desc: Optional[Dict]) -> Optional[Dict]:
        """Merge structured_description from main and referenced papers.

        Fills in null values from the main paper with values from the reference.
        """
        if not main_desc and not ref_desc:
            return None
        if not main_desc:
            return ref_desc
        if not ref_desc:
            return main_desc

        merged = {}
        all_keys = set(list(main_desc.keys()) + list(ref_desc.keys()))
        for key in all_keys:
            main_val = main_desc.get(key)
            ref_val = ref_desc.get(key)
            if isinstance(main_val, dict) and isinstance(ref_val, dict):
                # Merge nested dicts, preferring non-null main values
                merged_sub = {}
                sub_keys = set(list(main_val.keys()) + list(ref_val.keys()))
                for sk in sub_keys:
                    mv = main_val.get(sk)
                    rv = ref_val.get(sk)
                    if mv is not None:
                        merged_sub[sk] = mv
                    elif rv is not None:
                        merged_sub[sk] = rv
                    else:
                        merged_sub[sk] = None
                merged[key] = merged_sub
            elif main_val is not None:
                merged[key] = main_val
            else:
                merged[key] = ref_val
        return merged

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

        This handles scenarios where we find a paragraph but it references another paper
        for the actual protocol details (scenarios 3 and 5).

        Args:
            result: The extraction result that may contain references
            pmid: Original PMID being searched
            assay_description: The assay description being searched for
            max_pages: Maximum pages to process
            max_new_tokens: Maximum tokens to generate
            _depth: Current search depth

        Returns:
            Updated result with combined paragraphs if references were found
        """
        if not self.search_references or _depth >= self.max_reference_depth:
            return result

        references_previous = result.get("references_previous", "none")
        if references_previous == "none" or not references_previous:
            return result

        print(f"  Found reference to previous work: {references_previous}...")

        # Extract PMIDs from the reference info and the paragraph itself
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

        # Try to fetch and extract from referenced papers
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
                # Merge original_paragraph dicts, prefixing keys with PMID source
                combined_paragraph = {}
                if isinstance(original_paragraph, dict):
                    for loc, content in original_paragraph.items():
                        combined_paragraph[f"[PMID {pmid}] {loc}"] = content
                elif original_paragraph:  # Legacy string format
                    combined_paragraph[f"[PMID {pmid}] extracted"] = original_paragraph
                if isinstance(ref_paragraph, dict):
                    for loc, content in ref_paragraph.items():
                        combined_paragraph[f"[PMID {ref_pmid}] {loc}"] = content
                elif ref_paragraph:  # Legacy string format
                    combined_paragraph[f"[PMID {ref_pmid}] extracted"] = ref_paragraph

                result["original_paragraph"] = combined_paragraph
                result["source"] = f"{result.get('source', '')} + referenced_paper_{ref_pmid}"

                # Merge structured_description from reference into main
                result["structured_description"] = self._merge_structured_descriptions(
                    result.get("structured_description"),
                    ref_result.get("structured_description")
                )

                # Update search_path to reflect we went through reference
                current_path = result.get("search_path", ["main"])
                ref_search_path = ref_result.get("search_path", ["main"])
                # Build combined path: current path + "reference" + any additional steps from ref
                combined_path = list(current_path) if isinstance(current_path, list) else ["main"]
                combined_path.append("reference")
                # Add any steps beyond "main" from the reference's search path
                if isinstance(ref_search_path, list) and len(ref_search_path) > 1:
                    combined_path.extend(ref_search_path[1:])
                result["search_path"] = combined_path
                # Update supplementary_source based on where supplementary was used
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
        Extract structured SPR assay information from a paper.

        This method will:
        1. First search the main paper
        2. If not found and search_supplementary=True, fetch and search supplementary materials
        3. If still not found and search_references=True, look for referenced papers and search those

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
            Dictionary with 'original_paragraph', 'structured_description', 'location',
            'confidence', and 'source' keys
        """
        print(f"\n{'='*60}")
        print(f"Searching for assay in PMID {pmid} (depth={_depth})")
        print(f"{'='*60}")

        # Get main PDF path
        pdf_path = self.pubmed_fetcher.get_pdf_path(pmid)
        if pdf_path is None:
            if self.search_references:
                print(f"  Main paper not found locally, trying to fetch from PMC...")
                pdf_path = self.pubmed_fetcher.fetch_paper_by_pmid(pmid)

            if pdf_path is None:
                return {
                    "pmid": pmid,
                    "assay_description": assay_description,
                    "original_paragraph": {"error": "PDF not found and could not be fetched"},
                    "confidence": "N/A",
                    "source": "N/A",
                    "search_path": [],
                    "supplementary_source": [],
                    "structured_description": None
                }

        # Convert main PDF to images (use cache if available)
        cache_key = f"{pmid}_{max_pages}"
        if cache_key in self._image_cache:
            images = self._image_cache[cache_key]
            print(f"  Using cached images for PMID {pmid} ({len(images)} pages)")
        else:
            images = self.doc_converter.pdf_to_images(pdf_path, max_pages=max_pages, label="main")
            if images and len(images) > 0:
                self._image_cache[cache_key] = images
        if images is None or len(images) == 0:
            return {
                "pmid": pmid,
                "assay_description": assay_description,
                "original_paragraph": {"error": "PDF conversion failed"},
                "confidence": "N/A",
                "source": "N/A",
                "search_path": [],
                "supplementary_source": [],
                "structured_description": None
            }

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
                                # Merge structured_description from main attempt and reference
                                if result:
                                    ref_result["structured_description"] = self._merge_structured_descriptions(
                                        result.get("structured_description"),
                                        ref_result.get("structured_description")
                                    )
                                # Update search_path based on where it was found in the reference
                                ref_search_path = ref_result.get("search_path", ["main"])
                                # Build path: ["main", "reference"] + any additional steps from ref
                                combined_path = ["main", "reference"]
                                if isinstance(ref_search_path, list) and len(ref_search_path) > 1:
                                    combined_path.extend(ref_search_path[1:])
                                ref_result["search_path"] = combined_path
                                # Update supplementary_source if supplementary was involved
                                if "supplementary" in combined_path:
                                    ref_result["supplementary_source"] = ["reference"]
                                else:
                                    ref_result["supplementary_source"] = []
                                print(f"  Found in referenced paper PMID {ref_pmid}!")
                                return ref_result

            # ========== STEP 4: Return NOT FOUND ==========
            print(f"  Assay description not found in any searched documents.")
            return {
                "pmid": pmid,
                "assay_description": assay_description,
                "original_paragraph": {},
                "confidence": "N/A",
                "source": "not_found",
                "search_path": [],
                "supplementary_source": [],
                "searched_locations": f"main_paper, supplementary({len(supp_files)} files)",
                "structured_description": None
            }

        except Exception as e:
            print(f"Error extracting assay description for PMID {pmid}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "pmid": pmid,
                "assay_description": assay_description,
                "original_paragraph": {"error": str(e)},
                "confidence": "N/A",
                "source": "error",
                "search_path": [],
                "supplementary_source": [],
                "structured_description": None
            }

    # ==================== Batch Processing ====================

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
            pmid = str(int(row[pmid_col]))
            description = row[description_col]
            reactant_set_id = row[reactant_set_id_col] if reactant_set_id_col and reactant_set_id_col in row and pd.notna(row[reactant_set_id_col]) else None

            # Extract optional context
            protein = str(row[protein_col]) if protein_col and protein_col in row and pd.notna(row[protein_col]) else None
            ligand_smiles = str(row[ligand_smiles_col]) if ligand_smiles_col and ligand_smiles_col in row and pd.notna(row[ligand_smiles_col]) else None

            affinity_data = None
            if affinity_value_col and affinity_value_col in row and pd.notna(row[affinity_value_col]):
                affinity_data = {
                    "type": str(row[affinity_type_col]) if affinity_type_col and affinity_type_col in row and pd.notna(row[affinity_type_col]) else "Kd",
                    "value": row[affinity_value_col],
                    "relation": str(row[affinity_relation_col]) if affinity_relation_col and affinity_relation_col in row and pd.notna(row[affinity_relation_col]) else "=",
                    "unit": str(row[affinity_unit_col]) if affinity_unit_col and affinity_unit_col in row and pd.notna(row[affinity_unit_col]) else "nM"
                }

            row_count += 1
            print(f"\n{'='*80}")
            print(f"Processing row {row_count}/{len(df_to_process)}")
            if reactant_set_id is not None:
                print(f"Reactant Set ID: {reactant_set_id}")
            print(f"PMID: {pmid}")
            print(f"Description: {description[:100]}...")
            if protein:
                print(f"Protein: {protein}")
            if ligand_smiles:
                print(f"Ligand SMILES: {ligand_smiles[:80]}...")

            result = self.extract_assay_description(
                pmid, description,
                protein=protein,
                ligand_smiles=ligand_smiles,
                affinity_data=affinity_data,
                max_pages=max_pages
            )
            if reactant_set_id is not None:
                result["reactant_set_id"] = reactant_set_id
            results.append(result)

            if delay > 0 and row_count < len(df_to_process):
                time.sleep(delay)

        # Print token usage summary
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
        Process a dataframe and save results as JSON files per PMID (matching ground truth format).

        Args:
            df: DataFrame with PMID, DESCRIPTION, and context columns
            output_dir: Directory to save JSON files (one per PMID)
            pmid_col: Name of PMID column
            description_col: Name of description column
            reactant_set_id_col: Name of reactant set ID column
            protein_col: Name of protein column
            ligand_name_col: Name of ligand reference name column
            ligand_smiles_col: Name of ligand SMILES column
            affinity_type_col: Name of affinity type column
            affinity_value_col: Name of affinity value column
            affinity_relation_col: Name of affinity relation column
            affinity_unit_col: Name of affinity unit column
            limit: Maximum number of rows to process
            max_pages: Maximum pages per PDF to process
            delay: Delay between queries

        Returns:
            Dict mapping PMID -> Path of saved JSON file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Group by PMID
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
                reactant_set_id = row[reactant_set_id_col] if reactant_set_id_col and reactant_set_id_col in row and pd.notna(row[reactant_set_id_col]) else None
                description = row[description_col]
                protein = str(row[protein_col]) if protein_col and protein_col in row and pd.notna(row[protein_col]) else None
                ligand_name = str(row[ligand_name_col]) if ligand_name_col and ligand_name_col in row and pd.notna(row[ligand_name_col]) else None
                ligand_smiles = str(row[ligand_smiles_col]) if ligand_smiles_col and ligand_smiles_col in row and pd.notna(row[ligand_smiles_col]) else None

                affinity_data = None
                if affinity_value_col and affinity_value_col in row and pd.notna(row[affinity_value_col]):
                    affinity_data = {
                        "type": str(row[affinity_type_col]) if affinity_type_col and affinity_type_col in row and pd.notna(row[affinity_type_col]) else "Kd",
                        "value": float(row[affinity_value_col]) if pd.notna(row[affinity_value_col]) else None,
                        "relation": str(row[affinity_relation_col]) if affinity_relation_col and affinity_relation_col in row and pd.notna(row[affinity_relation_col]) else "=",
                        "unit": str(row[affinity_unit_col]) if affinity_unit_col and affinity_unit_col in row and pd.notna(row[affinity_unit_col]) else "nM"
                    }

                print(f"\n{'='*80}")
                print(f"Processing row {row_count}/{total_rows}")
                if reactant_set_id is not None:
                    print(f"Reactant Set ID: {reactant_set_id}")
                print(f"PMID: {pmid_str}")
                print(f"Description: {description[:100]}...")
                if protein:
                    print(f"Protein: {protein}")
                if ligand_smiles:
                    print(f"Ligand SMILES: {ligand_smiles[:80]}...")

                # Extract assay info
                result = self.extract_assay_description(
                    pmid_str, description,
                    protein=protein,
                    ligand_smiles=ligand_smiles,
                    affinity_data=affinity_data,
                    max_pages=max_pages
                )

                # Build entry in ground truth format
                entry = {
                    "reactant_set_id": int(reactant_set_id) if reactant_set_id else None,
                    "pmid": int(pmid),
                    "protein": protein,
                    "ligand": {
                        "reference_name": ligand_name,
                        "smiles": ligand_smiles
                    },
                    "affinity_data": affinity_data,
                    "DESCRIPTION": description,
                    "search_path": result.get("search_path", "N/A"),
                    "supplementary_source": result.get("supplementary_source"),
                    "references_previous": result.get("references_previous"),
                    "original_paragraph": result.get("original_paragraph", {}),
                    "structured_description": result.get("structured_description")
                }

                # Use reactant_set_id as key (as string, matching ground truth format)
                key = str(reactant_set_id) if reactant_set_id else f"entry_{row_count}"
                pmid_results[key] = entry

                if delay > 0:
                    time.sleep(delay)

            # Save JSON file for this PMID
            json_path = output_path / f"{pmid_str}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(pmid_results, f, indent=4, ensure_ascii=False)
            print(f"\nSaved: {json_path}")
            saved_files[pmid_str] = json_path

        # Print token usage summary
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
        print(f"\nSaved {len(saved_files)} JSON files to {output_dir}")

        return saved_files

    # ==================== Batched Extraction (One Model Call Per PMID) ====================

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

        # Initialize results dict with error state for all pairs
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

        # Get main PDF path
        pdf_path = self.pubmed_fetcher.get_pdf_path(pmid)
        if pdf_path is None:
            if self.search_references:
                print(f"  Main paper not found locally, trying to fetch from PMC...")
                pdf_path = self.pubmed_fetcher.fetch_paper_by_pmid(pmid)

            if pdf_path is None:
                for pair_id in results:
                    results[pair_id]["original_paragraph"] = {"error": "PDF not found and could not be fetched"}
                return results

        # Convert main PDF to images (use cache if available)
        cache_key = f"{pmid}_{max_pages}"
        if cache_key in self._image_cache:
            images = self._image_cache[cache_key]
            print(f"  Using cached images for PMID {pmid} ({len(images)} pages)")
        else:
            images = self.doc_converter.pdf_to_images(pdf_path, max_pages=max_pages, label="main")
            if images and len(images) > 0:
                self._image_cache[cache_key] = images

        if images is None or len(images) == 0:
            for pair_id in results:
                results[pair_id]["original_paragraph"] = {"error": "PDF conversion failed"}
            return results

        # Build batched prompt
        prompt = get_batched_structured_assay_extraction_prompt(pairs)

        try:
            # ========== STEP 1: Search main paper ==========
            print(f"[Step 1] Searching main paper for {len(pairs)} pairs...")
            response_text, _, _ = self._query_model(images, prompt, max_new_tokens)

            print(f"\n[DEBUG] Raw response (first 500 chars): {response_text[:500]}...")

            parsed_results = self._parse_response(response_text)

            if parsed_results and isinstance(parsed_results, dict):
                # Check how many pairs were found
                found_pairs = []
                missing_pairs = []

                for pair in pairs:
                    pair_id = str(pair.get("id", "unknown"))
                    if pair_id in parsed_results:
                        pair_result = parsed_results[pair_id]
                        if self._is_paragraph_found(pair_result.get("original_paragraph")):
                            # Found in main paper
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

                # ========== STEP 2: Search supplementary for missing pairs ==========
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

                # ========== STEP 3: Check for referenced literature for missing pairs ==========
                if missing_pairs and self.search_references and _depth < self.max_reference_depth:
                    print(f"\n[Step 3] Checking referenced literature for {len(missing_pairs)} missing pairs...")

                    # Check if any result mentions references
                    ref_pmids_to_try = set()
                    for pair_id, result in results.items():
                        refs = result.get("references_previous", "none")
                        if refs and refs != "none":
                            extracted_pmids = self.citation_searcher.extract_referenced_pmids(refs)
                            ref_pmids_to_try.update(extracted_pmids)

                    ref_pmids_to_try = [p for p in ref_pmids_to_try if p != pmid][:3]

                    for ref_pmid in ref_pmids_to_try:
                        if not missing_pairs:
                            break

                        print(f"  Searching referenced paper PMID {ref_pmid} for {len(missing_pairs)} pairs...")
                        ref_results = self.extract_assay_descriptions_batched(
                            ref_pmid,
                            missing_pairs,
                            max_pages=max_pages,
                            max_new_tokens=max_new_tokens,
                            _depth=_depth + 1
                        )

                        newly_found = []
                        for pair in missing_pairs[:]:
                            pair_id = str(pair.get("id", "unknown"))
                            if pair_id in ref_results:
                                ref_result = ref_results[pair_id]
                                if self._is_paragraph_found(ref_result.get("original_paragraph")):
                                    ref_result["source"] = f"referenced_paper_{ref_pmid}_from_{pmid}"
                                    # Update search_path
                                    ref_search_path = ref_result.get("search_path", ["main"])
                                    combined_path = ["main", "reference"]
                                    if isinstance(ref_search_path, list) and len(ref_search_path) > 1:
                                        combined_path.extend(ref_search_path[1:])
                                    ref_result["search_path"] = combined_path
                                    if "supplementary" in combined_path:
                                        ref_result["supplementary_source"] = ["reference"]
                                    else:
                                        ref_result["supplementary_source"] = []
                                    results[pair_id] = ref_result
                                    newly_found.append(pair)

                        for pair in newly_found:
                            missing_pairs.remove(pair)

                        if newly_found:
                            print(f"    Found {len(newly_found)} pairs in referenced paper {ref_pmid}")

                # Mark remaining missing pairs
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

        Args:
            df: DataFrame with PMID, reactant_set_id, DESCRIPTION, and context columns
            output_dir: Directory to save JSON files (one per PMID)
            pmid_col: Name of PMID column
            reactant_set_id_col: Name of reactant set ID column
            description_col: Name of description column
            protein_col: Name of protein column
            ligand_name_col: Name of ligand reference name column
            ligand_smiles_col: Name of ligand SMILES column
            affinity_type_col: Name of affinity type column
            affinity_value_col: Name of affinity value column
            affinity_relation_col: Name of affinity relation column
            affinity_unit_col: Name of affinity unit column
            limit: Maximum number of rows to process
            max_pages: Maximum pages per PDF to process
            max_new_tokens: Maximum tokens for batch output
            delay: Delay between PMID processing

        Returns:
            Dict mapping PMID -> Path of saved JSON file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Group by PMID
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

            # Build list of pairs for this PMID
            pairs = []
            pair_metadata = {}  # Store metadata for building output

            for _, row in group.iterrows():
                reactant_set_id = row[reactant_set_id_col] if reactant_set_id_col in row and pd.notna(row[reactant_set_id_col]) else None
                pair_id = str(int(reactant_set_id)) if reactant_set_id else f"row_{len(pairs)}"

                description = row[description_col]
                protein = str(row[protein_col]) if protein_col and protein_col in row and pd.notna(row[protein_col]) else None
                ligand_name = str(row[ligand_name_col]) if ligand_name_col and ligand_name_col in row and pd.notna(row[ligand_name_col]) else None
                ligand_smiles = str(row[ligand_smiles_col]) if ligand_smiles_col and ligand_smiles_col in row and pd.notna(row[ligand_smiles_col]) else None

                affinity_data = None
                if affinity_value_col and affinity_value_col in row and pd.notna(row[affinity_value_col]):
                    affinity_data = {
                        "type": str(row[affinity_type_col]) if affinity_type_col and affinity_type_col in row and pd.notna(row[affinity_type_col]) else "Kd",
                        "value": float(row[affinity_value_col]) if pd.notna(row[affinity_value_col]) else None,
                        "relation": str(row[affinity_relation_col]) if affinity_relation_col and affinity_relation_col in row and pd.notna(row[affinity_relation_col]) else "=",
                        "unit": str(row[affinity_unit_col]) if affinity_unit_col and affinity_unit_col in row and pd.notna(row[affinity_unit_col]) else "nM"
                    }

                pairs.append({
                    "id": pair_id,
                    "assay_description": description,
                    "protein": protein,
                    "ligand_smiles": ligand_smiles,
                    "affinity_data": affinity_data
                })

                pair_metadata[pair_id] = {
                    "reactant_set_id": int(reactant_set_id) if reactant_set_id else None,
                    "pmid": int(pmid),
                    "protein": protein,
                    "ligand": {
                        "reference_name": ligand_name,
                        "smiles": ligand_smiles
                    },
                    "affinity_data": affinity_data,
                    "DESCRIPTION": description
                }

            print(f"\n{'='*80}")
            print(f"Processing PMID {pmid_count}/{total_pmids}: {pmid_str} ({len(pairs)} pairs)")
            print(f"{'='*80}")

            # Batched extraction
            extraction_results = self.extract_assay_descriptions_batched(
                pmid_str,
                pairs,
                max_pages=max_pages,
                max_new_tokens=max_new_tokens
            )

            # Build output JSON matching ground truth format
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

            # Save JSON file for this PMID
            json_path = output_path / f"{pmid_str}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(pmid_output, f, indent=4, ensure_ascii=False)
            print(f"\nSaved: {json_path}")
            saved_files[pmid_str] = json_path

            if delay > 0 and pmid_count < total_pmids:
                time.sleep(delay)

        # Print token usage summary
        print(f"\n{'='*80}")
        print("TOKEN USAGE SUMMARY (Batched)")
        print(f"{'='*80}")
        print(f"Total Requests: {self.token_usage['requests']}")
        print(f"Total Input Tokens: {self.token_usage['total_input_tokens']:,}")
        print(f"Total Output Tokens: {self.token_usage['total_output_tokens']:,}")
        print(f"Total Tokens: {self.token_usage['total_tokens']:,}")
        if self.token_usage['requests'] > 0:
            print(f"Avg Input/Request: {self.token_usage['total_input_tokens'] / self.token_usage['requests']:,.0f}")
            print(f"Avg Output/Request: {self.token_usage['total_output_tokens'] / self.token_usage['requests']:,.0f}")
        print(f"{'='*80}")
        print(f"\nProcessed {total_pairs} pairs across {total_pmids} PMIDs")
        print(f"Saved {len(saved_files)} JSON files to {output_dir}")

        return saved_files

    def clear_image_cache(self):
        """Clear the cached PDF-to-image conversions to free memory."""
        self._image_cache.clear()

    def cleanup(self):
        """Clean up GPU memory and caches."""
        print("\nCleaning up GPU memory...")
        self.clear_image_cache()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print("Cleanup complete!")
