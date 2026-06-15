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
from .prompts import get_assay_extraction_prompt


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
            search_supplementary: Whether to automatically fetch and search supplementary materials
            search_references: Whether to fetch referenced papers if assay not found
            max_reference_depth: Maximum depth for recursive reference search (1 = only direct refs)
            ncbi_api_key: NCBI API key for higher rate limits (optional)
        """
        print(f"Loading Qwen3-VL model: {model_name}")
        print(f"Device: {device}")
        print(f"Torch dtype: {torch_dtype}")
        print(f"Search supplementary: {search_supplementary}")
        print(f"Search references: {search_references}")

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
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
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
            # Clean newlines from string values after parsing
            for key, value in result.items():
                if isinstance(value, str):
                    result[key] = ' '.join(value.split())
            return result
        except json.JSONDecodeError:
            return None

    # ==================== Reference Handling ====================

    def _check_and_fetch_references(
        self,
        result: Dict,
        pmid: str,
        assay_description: str,
        max_pages: Optional[int],
        max_new_tokens: int,
        _depth: int
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
                max_pages=max_pages,
                max_new_tokens=max_new_tokens,
                _depth=_depth + 1
            )

            ref_paragraph = ref_result.get("original_paragraph", "")
            if ref_paragraph not in ["NOT FOUND", None, ""] and not ref_paragraph.startswith("ERROR"):
                original_paragraph = result.get("original_paragraph", "")
                combined_paragraph = (
                    f"[From PMID {pmid}]: {original_paragraph}"
                    f"[From referenced PMID {ref_pmid}]: {ref_paragraph}"
                )

                result["original_paragraph"] = combined_paragraph
                result["source"] = f"{result.get('source', '')} + referenced_paper_{ref_pmid}"
                result["location"] = f"{result.get('location', '')} + {ref_result.get('location', 'N/A')}"

                # Update search_path to reflect we went through reference
                current_path = result.get("search_path", "main")
                ref_search_path = ref_result.get("search_path", "main")
                if current_path == "main":
                    if ref_search_path == "main":
                        result["search_path"] = "main -> reference"
                    elif ref_search_path == "main -> supplementary":
                        result["search_path"] = "main -> reference -> supplementary"
                        result["supplementary_source"] = "reference"
                elif current_path == "main -> supplementary":
                    if ref_search_path == "main":
                        result["search_path"] = "main -> supplementary -> reference"
                    elif ref_search_path == "main -> supplementary":
                        result["search_path"] = "main -> supplementary -> reference -> supplementary"
                        result["supplementary_source"] = "main + reference"

                print(f"  Combined with referenced paper PMID {ref_pmid}!")
                return result

        return result

    # ==================== Main Extraction Method ====================

    def extract_assay_description(
        self,
        pmid: str,
        assay_description: str,
        max_pages: Optional[int] = None,
        max_new_tokens: int = 2048,
        _depth: int = 0
    ) -> Dict[str, str]:
        """
        Extract the complete original paragraph describing an assay from a paper.

        This method will:
        1. First search the main paper
        2. If not found and search_supplementary=True, fetch and search supplementary materials
        3. If still not found and search_references=True, look for referenced papers and search those

        Args:
            pmid: PubMed ID of the paper
            assay_description: Brief assay description from BindingDB
            max_pages: Maximum number of pages to process (None for all)
            max_new_tokens: Maximum tokens to generate
            _depth: Internal parameter for tracking reference search depth

        Returns:
            Dictionary with 'original_paragraph', 'location', 'confidence', and 'source' keys
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
                    "original_paragraph": "ERROR: PDF not found and could not be fetched",
                    "location": "N/A",
                    "confidence": "N/A",
                    "source": "N/A",
                    "search_path": "N/A",
                    "supplementary_source": "N/A"
                }

        # Convert main PDF to images
        images = self.doc_converter.pdf_to_images(pdf_path, max_pages=max_pages, label="main")
        if images is None or len(images) == 0:
            return {
                "pmid": pmid,
                "assay_description": assay_description,
                "original_paragraph": "ERROR: PDF conversion failed",
                "location": "N/A",
                "confidence": "N/A",
                "source": "N/A",
                "search_path": "N/A",
                "supplementary_source": "N/A"
            }

        prompt = get_assay_extraction_prompt(assay_description)
        supp_files = []

        try:
            # ========== STEP 1: Search main paper ==========
            print(f"[Step 1] Searching main paper for PMID {pmid}...")
            response_text, _, _ = self._query_model(images, prompt, max_new_tokens)

            print(f"\n[DEBUG] Raw response: {response_text[:500]}...")

            result = self._parse_response(response_text)

            if result and result.get("original_paragraph") not in ["NOT FOUND", None, ""]:
                result["pmid"] = pmid
                result["assay_description"] = assay_description
                result["source"] = f"main_paper_{pmid}"
                result["search_path"] = "main"
                result["supplementary_source"] = "N/A"
                print(f"  Found in main paper!")

                result = self._check_and_fetch_references(
                    result, pmid, assay_description, max_pages, max_new_tokens, _depth
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

                        if supp_result and supp_result.get("original_paragraph") not in ["NOT FOUND", None, ""]:
                            supp_result["pmid"] = pmid
                            supp_result["assay_description"] = assay_description
                            supp_result["source"] = f"supplementary_{supp_path.name}"
                            supp_result["location"] = f"Supplementary: {supp_result.get('location', 'N/A')}"
                            supp_result["search_path"] = "main -> supplementary"
                            supp_result["supplementary_source"] = "main"
                            print(f"  Found in supplementary: {supp_path.name}!")

                            supp_result = self._check_and_fetch_references(
                                supp_result, pmid, assay_description, max_pages, max_new_tokens, _depth
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
                                max_pages=max_pages,
                                max_new_tokens=max_new_tokens,
                                _depth=_depth + 1
                            )

                            if ref_result.get("original_paragraph") not in ["NOT FOUND", None, ""] and not ref_result.get("original_paragraph", "").startswith("ERROR"):
                                ref_result["source"] = f"referenced_paper_{ref_pmid}_from_{pmid}"
                                # Update search_path based on where it was found in the reference
                                ref_search_path = ref_result.get("search_path", "main")
                                if ref_search_path == "main":
                                    ref_result["search_path"] = "main -> reference"
                                elif ref_search_path == "main -> supplementary":
                                    ref_result["search_path"] = "main -> reference -> supplementary"
                                    ref_result["supplementary_source"] = "reference"
                                # If supplementary_source is N/A and search_path doesn't involve supplementary, keep it N/A
                                if "supplementary" not in ref_result.get("search_path", ""):
                                    ref_result["supplementary_source"] = "N/A"
                                print(f"  Found in referenced paper PMID {ref_pmid}!")
                                return ref_result

            # ========== STEP 4: Return NOT FOUND ==========
            print(f"  Assay description not found in any searched documents.")
            return {
                "pmid": pmid,
                "assay_description": assay_description,
                "original_paragraph": "NOT FOUND",
                "location": "N/A",
                "confidence": "N/A",
                "source": "not_found",
                "search_path": "N/A",
                "supplementary_source": "N/A",
                "searched_locations": f"main_paper, supplementary({len(supp_files)} files)"
            }

        except Exception as e:
            print(f"Error extracting assay description for PMID {pmid}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "pmid": pmid,
                "assay_description": assay_description,
                "original_paragraph": f"ERROR: {str(e)}",
                "location": "N/A",
                "confidence": "N/A",
                "source": "error",
                "search_path": "N/A",
                "supplementary_source": "N/A"
            }

    # ==================== Batch Processing ====================

    def process_dataframe(
        self,
        df: pd.DataFrame,
        pmid_col: str = "PMID",
        description_col: str = "DESCRIPTION",
        limit: Optional[int] = None,
        max_pages: Optional[int] = None,
        delay: float = 0.0
    ) -> pd.DataFrame:
        """
        Process a dataframe of assay descriptions.

        Args:
            df: DataFrame with PMID and DESCRIPTION columns
            pmid_col: Name of PMID column
            description_col: Name of description column
            limit: Maximum number of rows to process (None for all)
            max_pages: Maximum pages per PDF to process
            delay: Delay between queries (useful for API rate limiting)

        Returns:
            DataFrame with added columns for extracted paragraphs
        """
        results = []
        df_to_process = df.head(limit) if limit else df

        row_count = 0
        for _, row in df_to_process.iterrows():
            pmid = str(int(row[pmid_col]))
            description = row[description_col]

            row_count += 1
            print(f"\n{'='*80}")
            print(f"Processing row {row_count}/{len(df_to_process)}")
            print(f"PMID: {pmid}")
            print(f"Description: {description[:100]}...")

            result = self.extract_assay_description(pmid, description, max_pages=max_pages)
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
        df_result["original_paragraph"] = [result.get("original_paragraph", "N/A") for result in results]
        df_result["paragraph_location"] = [result.get("location", "N/A") for result in results]
        df_result["extraction_confidence"] = [result.get("confidence", "N/A") for result in results]
        df_result["source"] = [result.get("source", "N/A") for result in results]
        df_result["search_path"] = [result.get("search_path", "N/A") for result in results]
        df_result["supplementary_source"] = [result.get("supplementary_source", "N/A") for result in results]
        df_result["references_previous"] = [result.get("references_previous", "N/A") for result in results]

        return df_result

    def cleanup(self):
        """Clean up GPU memory."""
        print("\nCleaning up GPU memory...")
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print("Cleanup complete!")
