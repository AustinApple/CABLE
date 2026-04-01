"""
Base Assay Extraction Agent - Shared functionality for all extraction agents.

This module contains common functionality used by both single-step and two-step
extraction agents, including model loading, PDF processing, and response parsing.
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from PIL import Image
import os

from .pubmed_utils import PubMedFetcher, CitationSearcher
from .pdf_utils import DocumentConverter


class BaseAssayExtractionAgent:
    """Base class with shared functionality for assay extraction agents.

    Features:
    - Model loading and query methods
    - PDF to image conversion with caching
    - Response parsing and cleaning
    - PubMed/PMC integration for fetching papers and supplementary materials
    - Reference handling
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
        Initialize the base agent.

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
        self.model_name = model_name

        # Map string dtype to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        self.torch_dtype_obj = dtype_map.get(torch_dtype, torch.bfloat16)

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype_obj,
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
        self.citation_searcher = CitationSearcher(
            ncbi_api_key=self.ncbi_api_key,
            model=self.model,
            processor=self.processor,
            device=self.device
        )
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
            add_generation_prompt=True,
            enable_thinking=False
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
            result = self._clean_string_values(result)
            return result
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _clean_string_values(obj):
        """Recursively clean newlines from string values in nested dicts/lists."""
        if isinstance(obj, dict):
            return {k: BaseAssayExtractionAgent._clean_string_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [BaseAssayExtractionAgent._clean_string_values(item) for item in obj]
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
            return any(v and str(v).strip() and str(v).strip() != "NOT FOUND" for v in original_paragraph.values())
        if isinstance(original_paragraph, str):
            return original_paragraph not in ["NOT FOUND", "", None] and not original_paragraph.startswith("ERROR")
        return False

    # ==================== PDF/Image Handling ====================

    def _get_paper_images(
        self,
        pmid: str,
        max_pages: Optional[int] = None
    ) -> Tuple[Optional[List[Image.Image]], Optional[Path]]:
        """
        Get images for a paper, handling caching and fetching.

        Args:
            pmid: PubMed ID of the paper
            max_pages: Maximum number of pages to process

        Returns:
            Tuple of (images list or None, pdf_path or None)
        """
        pdf_path = self.pubmed_fetcher.get_pdf_path(pmid)
        if pdf_path is None:
            if self.search_references:
                print(f"  Main paper not found locally, trying to fetch from PMC...")
                pdf_path = self.pubmed_fetcher.fetch_paper_by_pmid(pmid)

            if pdf_path is None:
                return None, None

        # Convert main PDF to images (use cache if available)
        cache_key = f"{pmid}_{max_pages}"
        if cache_key in self._image_cache:
            images = self._image_cache[cache_key]
            print(f"  Using cached images for PMID {pmid} ({len(images)} pages)")
        else:
            images = self.doc_converter.pdf_to_images(pdf_path, max_pages=max_pages, label="main")
            if images and len(images) > 0:
                self._image_cache[cache_key] = images

        return images, pdf_path

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

    def _extract_for_reference(
        self,
        pmid: str,
        assay_description: str,
        max_pages: Optional[int],
        max_new_tokens: int,
        _depth: int
    ) -> Dict:
        """Extract from a referenced paper. Subclasses must override this.

        This is the recursive entry point called by _check_and_fetch_references
        and _search_references_not_found. It should run the full extraction
        pipeline (main paper + supplementary + reference search) for the given PMID.

        Args:
            pmid: PubMed ID of the referenced paper
            assay_description: Brief assay description from BindingDB
            max_pages: Maximum number of pages to process
            max_new_tokens: Maximum tokens to generate
            _depth: Current reference search depth

        Returns:
            Extraction result dict with 'original_paragraph', 'search_path', etc.
        """
        raise NotImplementedError(
            "Subclasses must implement _extract_for_reference for reference search"
        )

    @staticmethod
    def _combine_paragraphs(
        main_paragraph,
        main_pmid: str,
        ref_paragraph,
        ref_pmid: str
    ):
        """Combine original_paragraph from main and referenced papers.

        Handles both dict format (two-step agent) and string format (legacy).
        """
        if isinstance(main_paragraph, dict) and isinstance(ref_paragraph, dict):
            combined = {}
            for k, v in main_paragraph.items():
                combined[f"[PMID {main_pmid}] {k}"] = v
            for k, v in ref_paragraph.items():
                combined[f"[Ref PMID {ref_pmid}] {k}"] = v
            return combined
        if isinstance(main_paragraph, str) and isinstance(ref_paragraph, str):
            return (
                f"[From PMID {main_pmid}]: {main_paragraph}\n"
                f"[From referenced PMID {ref_pmid}]: {ref_paragraph}"
            )
        # Mixed types — normalize to dict
        combined = {}
        if isinstance(main_paragraph, dict):
            for k, v in main_paragraph.items():
                combined[f"[PMID {main_pmid}] {k}"] = v
        elif main_paragraph:
            combined[f"[PMID {main_pmid}]"] = str(main_paragraph)
        if isinstance(ref_paragraph, dict):
            for k, v in ref_paragraph.items():
                combined[f"[Ref PMID {ref_pmid}] {k}"] = v
        elif ref_paragraph:
            combined[f"[Ref PMID {ref_pmid}]"] = str(ref_paragraph)
        return combined

    def _resolve_reference_pmids(
        self,
        references_previous: str,
        original_paragraph,
        pmid: str
    ) -> List[str]:
        """Extract referenced PMIDs from references_previous and original_paragraph.

        Tries direct PMID extraction first, then falls back to citation-based search.

        Args:
            references_previous: The references_previous field from model output
            original_paragraph: The original_paragraph (dict or str)
            pmid: Current paper's PMID (excluded from results)

        Returns:
            List of unique referenced PMIDs (excluding current pmid)
        """
        ref_pmids = self.citation_searcher.extract_referenced_pmids(references_previous)

        # Also extract PMIDs mentioned in the paragraph text
        if isinstance(original_paragraph, str) and original_paragraph:
            ref_pmids.extend(
                self.citation_searcher.extract_referenced_pmids(original_paragraph)
            )
        elif isinstance(original_paragraph, dict):
            for v in original_paragraph.values():
                if v:
                    ref_pmids.extend(
                        self.citation_searcher.extract_referenced_pmids(str(v))
                    )

        ref_pmids = list(set(p for p in ref_pmids if p != pmid))

        if not ref_pmids:
            # Fallback: citation-based search (author/title/DOI → PubMed query)
            print(f"  No direct PMIDs found, trying citation-based search...")
            all_text = references_previous
            if isinstance(original_paragraph, str) and original_paragraph:
                all_text += " " + original_paragraph
            elif isinstance(original_paragraph, dict):
                all_text += " " + " ".join(
                    str(v) for v in original_paragraph.values() if v
                )

            citations = self.citation_searcher.extract_citations_from_text(all_text)
            if len(references_previous) > 20:
                citations.append(references_previous)

            for citation in citations[:3]:
                ref_pmid = self.citation_searcher.search_pmid_by_citation(citation)
                if ref_pmid and ref_pmid != pmid:
                    ref_pmids.append(ref_pmid)

            if not ref_pmids:
                print(f"  Could not find PMIDs from citations either")

        return ref_pmids

    def _check_and_fetch_references(
        self,
        result: Dict,
        pmid: str,
        assay_description: str,
        max_pages: Optional[int],
        max_new_tokens: int,
        _depth: int
    ) -> Dict:
        """Check if a found result references previous work and fetch/combine if needed.

        Called when a paragraph IS found but may reference another paper for the
        actual protocol details. If a referenced paper is found, the paragraphs
        are combined.

        Args:
            result: Extraction result that may contain references_previous
            pmid: Current paper's PMID
            assay_description: Brief assay description
            max_pages: Maximum pages to process
            max_new_tokens: Maximum tokens to generate
            _depth: Current reference search depth

        Returns:
            Updated result (combined with reference if found, unchanged otherwise)
        """
        if not self.search_references or _depth >= self.max_reference_depth:
            return result

        references_previous = result.get("references_previous", "none")
        if references_previous == "none" or not references_previous:
            return result

        print(f"  Found reference to previous work: {str(references_previous)[:200]}...")

        ref_pmids = self._resolve_reference_pmids(
            references_previous, result.get("original_paragraph"), pmid
        )
        if not ref_pmids:
            return result

        for ref_pmid in ref_pmids[:3]:
            print(f"  Fetching referenced paper PMID {ref_pmid}...")
            ref_result = self._extract_for_reference(
                ref_pmid, assay_description, max_pages, max_new_tokens, _depth + 1
            )

            ref_paragraph = ref_result.get("original_paragraph")
            if self._is_paragraph_found(ref_paragraph):
                # Combine paragraphs from both papers
                result["original_paragraph"] = self._combine_paragraphs(
                    result.get("original_paragraph"), pmid,
                    ref_paragraph, ref_pmid
                )
                result["source"] = (
                    f"{result.get('source', '')} + referenced_paper_{ref_pmid}"
                )

                # Extend search_path
                current_path = result.get("search_path", [])
                ref_path = ref_result.get("search_path", [])
                if isinstance(current_path, list) and isinstance(ref_path, list):
                    result["search_path"] = current_path + ["reference"] + ref_path
                else:
                    result["search_path"] = f"{current_path} -> reference -> {ref_path}"

                # Extend supplementary_source if reference used supplementary
                ref_supp = ref_result.get("supplementary_source", [])
                if ref_supp and isinstance(ref_supp, list) and len(ref_supp) > 0:
                    current_supp = result.get("supplementary_source", [])
                    if isinstance(current_supp, list):
                        result["supplementary_source"] = (
                            current_supp + [f"reference_{ref_pmid}"]
                        )

                print(f"  Combined with referenced paper PMID {ref_pmid}!")
                return result

        return result

    def _search_references_not_found(
        self,
        result: Dict,
        pmid: str,
        assay_description: str,
        max_pages: Optional[int],
        max_new_tokens: int,
        _depth: int
    ) -> Dict:
        """Search referenced literature when assay not found in main/supplementary.

        Called when no paragraph was found in the main paper or supplementary
        materials, but the model's response may contain references_previous
        pointing to another paper.

        Args:
            result: The not-found extraction result (may have references_previous)
            pmid: Current paper's PMID
            assay_description: Brief assay description
            max_pages: Maximum pages to process
            max_new_tokens: Maximum tokens to generate
            _depth: Current reference search depth

        Returns:
            Reference paper's result if found, original not-found result otherwise
        """
        if not self.search_references or _depth >= self.max_reference_depth:
            return result

        references_previous = result.get("references_previous", "none")
        if references_previous == "none" or not references_previous:
            return result

        print(f"\n  Not found locally. Checking referenced literature: "
              f"{str(references_previous)[:200]}...")

        ref_pmids = self._resolve_reference_pmids(
            references_previous, result.get("original_paragraph"), pmid
        )
        if not ref_pmids:
            return result

        for ref_pmid in ref_pmids[:3]:
            print(f"  Searching referenced paper PMID {ref_pmid}...")
            ref_result = self._extract_for_reference(
                ref_pmid, assay_description, max_pages, max_new_tokens, _depth + 1
            )

            if self._is_paragraph_found(ref_result.get("original_paragraph")):
                ref_result["source"] = f"referenced_paper_{ref_pmid}_from_{pmid}"

                # Prepend "reference" to search_path
                ref_path = ref_result.get("search_path", [])
                if isinstance(ref_path, list):
                    ref_result["search_path"] = ["reference"] + ref_path
                else:
                    ref_result["search_path"] = f"reference -> {ref_path}"

                # Mark supplementary_source if reference used supplementary
                ref_supp = ref_result.get("supplementary_source", [])
                if ref_supp and isinstance(ref_supp, list) and len(ref_supp) > 0:
                    ref_result["supplementary_source"] = [f"reference_{ref_pmid}"]

                print(f"  Found in referenced paper PMID {ref_pmid}!")
                return ref_result

        return result

    # ==================== Cleanup ====================

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
