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

import re

from .pubmed_utils import PubMedFetcher, CitationSearcher
from .pdf_utils import DocumentConverter, pdf_to_markdown


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

        # Create supplementary, reference and markdown cache directories
        self.supp_dir = self.pdf_dir / "supplementary"
        self.ref_dir = self.pdf_dir / "references"
        self.markdown_cache_dir = self.pdf_dir / "markdown_cache"
        self.supp_dir.mkdir(exist_ok=True)
        self.ref_dir.mkdir(exist_ok=True)
        self.markdown_cache_dir.mkdir(exist_ok=True)

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

        # Cache converted images and markdown per PMID
        self._image_cache: Dict[str, List[Image.Image]] = {}
        self._markdown_cache: Dict[str, Optional[str]] = {}
        # Mapping ref_pmid -> citation text, populated by _resolve_reference_pmids
        self._ref_pmid_to_citation: Dict[str, str] = {}

        # CUDA device index for MinerU subprocess (extract from device string)
        _dev = str(device)
        self._cuda_device_idx = _dev.replace("cuda:", "") if _dev.startswith("cuda:") else "0"

        # Path for logging missing references and inaccessible supplementary (set by caller)
        self.chase_log: Optional[Path] = None

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
            return any(
                k != "error" and v and str(v).strip() and str(v).strip() != "NOT FOUND"
                for k, v in original_paragraph.items()
            )
        if isinstance(original_paragraph, str):
            return original_paragraph not in ["NOT FOUND", "", None] and not original_paragraph.startswith("ERROR")
        return False

    @staticmethod
    def _paragraph_mentions_supplementary(original_paragraph) -> bool:
        """Check if extracted paragraph text mentions supplementary materials.

        Detects phrases like 'Supplemental Experimental Procedures',
        'Supporting Information', 'Table S1', 'Figure S2', etc.
        """
        if original_paragraph is None:
            return False
        if isinstance(original_paragraph, dict):
            text = " ".join(str(v) for v in original_paragraph.values())
        else:
            text = str(original_paragraph)

        pattern = re.compile(
            r'supplement(?:al|ary)\s+\w+'
            r'|supporting\s+(?:information|material)'
            r'|\bTable\s+S\d'
            r'|\bFigure\s+S\d'
            r'|\bFig\.\s*S\d',
            re.IGNORECASE
        )
        return bool(pattern.search(text))

    # ==================== PDF/Image Handling ====================

    MAX_SUPPLEMENTARY_PAGES = 50

    @staticmethod
    def _get_page_count(file_path: Path) -> Optional[int]:
        """Return the page count of a PDF/DOCX without converting to images."""
        import fitz
        suffix = file_path.suffix.lower()
        try:
            if suffix == '.pdf':
                doc = fitz.open(str(file_path))
                count = len(doc)
                doc.close()
                return count
            elif suffix == '.docx':
                return None  # can't cheaply count pages for docx
        except Exception:
            return None
        return None

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

    def _get_paper_markdown(self, pmid: str) -> Optional[str]:
        """Get paper content as markdown via MinerU. Results are cached in memory.

        Args:
            pmid: PubMed ID of the paper

        Returns:
            Markdown text string, or None if conversion failed
        """
        if pmid in self._markdown_cache:
            return self._markdown_cache[pmid]

        pdf_path = self.pubmed_fetcher.get_pdf_path(pmid)
        if pdf_path is None:
            pdf_path = self.pubmed_fetcher.fetch_paper_by_pmid(pmid)
        if pdf_path is None:
            self._markdown_cache[pmid] = None
            return None

        print(f"  Converting PMID {pmid} to markdown via MinerU...")
        md_path = pdf_to_markdown(
            pdf_path,
            cache_dir=self.markdown_cache_dir
        )
        if md_path is None:
            print(f"  MinerU conversion failed for PMID {pmid}")
            self._markdown_cache[pmid] = None
            return None

        markdown = md_path.read_text(encoding="utf-8")
        ref_section = self._extract_reference_section(markdown)
        self._markdown_cache[pmid] = ref_section
        print(f"  Markdown reference section ready ({len(ref_section)} chars)")
        return ref_section

    @staticmethod
    def _extract_reference_section(markdown: str) -> str:
        """Extract only the reference/bibliography section from paper markdown.

        Looks for common section headers (References, Bibliography, etc.) and
        returns everything from that header to the end of the document.
        Falls back to the last 20% of the document if no header is found.

        Args:
            markdown: Full paper markdown text

        Returns:
            Reference section text
        """
        pattern = re.compile(
            r'^#{1,3}\s*(references|bibliography|works cited|literature cited|citations)\s*$',
            re.IGNORECASE | re.MULTILINE
        )
        match = pattern.search(markdown)
        if match:
            return markdown[match.start():]
        # Fallback: last 20% of document likely contains references
        return markdown[int(len(markdown) * 0.8):]

    @staticmethod
    def _parse_references_from_markdown(markdown: str, ref_numbers: List[str]) -> Dict[str, str]:
        """Extract specific numbered references from a markdown reference list.

        Looks for patterns like "(26) Author, Title..." and extracts the full
        citation text for each requested reference number.

        Args:
            markdown: Full markdown text of the paper
            ref_numbers: List of reference numbers as strings (e.g. ["26", "27"])

        Returns:
            Dict mapping reference number -> full citation text
        """
        results = {}
        for num in ref_numbers:
            pattern = rf'\({re.escape(num)}\)\s+(.*?)(?=\n\s*\(\d+\)|\Z)'
            match = re.search(pattern, markdown, re.DOTALL)
            if match:
                citation = " ".join(match.group(1).split())
                results[num] = citation
        return results

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
                if k.startswith("[PMID ") or k.startswith("[Ref PMID "):
                    combined[k] = v
                else:
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
                if k.startswith("[PMID ") or k.startswith("[Ref PMID "):
                    combined[k] = v
                else:
                    combined[f"[PMID {main_pmid}] {k}"] = v
        elif main_paragraph:
            combined[f"[PMID {main_pmid}]"] = str(main_paragraph)
        if isinstance(ref_paragraph, dict):
            for k, v in ref_paragraph.items():
                combined[f"[Ref PMID {ref_pmid}] {k}"] = v
        elif ref_paragraph:
            combined[f"[Ref PMID {ref_pmid}]"] = str(ref_paragraph)
        return combined

    def _log_chase_miss(
        self,
        main_pmid: str,
        category: str,
        detail: str
    ) -> None:
        """Append a chase-miss entry to the log file.

        Args:
            main_pmid: PMID of the paper being processed
            category: 'reference' or 'supplementary'
            detail: Description of what was missed
        """
        if self.chase_log is None:
            return
        line = f"{main_pmid}\t{category}\t{detail.strip()}\n"
        with open(self.chase_log, "a", encoding="utf-8") as f:
            f.write(line)
        print(f"  [LOG] Chase miss logged → {self.chase_log}")

    def _resolve_reference_pmids(
        self,
        references_previous: str,
        original_paragraph,
        pmid: str,
        reference_number_in_text: Optional[str] = None
    ) -> List[str]:
        """Extract referenced PMIDs from references_previous and original_paragraph.

        Priority:
        1. If reference_number_in_text is given, parse exact citations from the
           paper's markdown (via MinerU) — deterministic, no hallucination.
        2. Direct PMID extraction from text.
        3. Citation-based PubMed search (author/title/volume/page).

        Args:
            references_previous: The references_previous field from model output
            original_paragraph: The original_paragraph (dict or str)
            pmid: Current paper's PMID (excluded from results)
            reference_number_in_text: Comma-separated reference numbers, e.g. "26, 27"

        Returns:
            List of unique referenced PMIDs (excluding current pmid)
        """
        # --- Priority 1: Markdown-based reference resolution (most reliable) ---
        if reference_number_in_text:
            ref_nums = [n.strip() for n in str(reference_number_in_text).split(",") if n.strip()]
            if ref_nums:
                markdown = self._get_paper_markdown(pmid)
                if markdown:
                    citations_from_md = self._parse_references_from_markdown(markdown, ref_nums)
                    if citations_from_md:
                        print(f"  Using markdown reference list for refs: {ref_nums}")
                        ref_pmids = []
                        resolved_nums = set()
                        for num, citation in citations_from_md.items():
                            print(f"  Searching PMID for ref ({num}): {citation[:80]}...")
                            ref_pmid = self.citation_searcher.search_pmid_by_citation(citation)
                            if ref_pmid and ref_pmid != pmid:
                                ref_pmids.append(ref_pmid)
                                resolved_nums.add(num)
                                self._ref_pmid_to_citation[ref_pmid] = f"({num}) {citation}"
                            else:
                                print(f"  Could not resolve PMID for ref ({num}) from markdown")
                                self._log_chase_miss(pmid, "reference", f"unresolved ref ({num}): {citation[:120]}")
                        # For ref nums not resolved via markdown, try references_previous text
                        unresolved_nums = [n for n in ref_nums if n not in resolved_nums]
                        if unresolved_nums and references_previous and len(str(references_previous)) > 20:
                            print(f"  Trying references_previous fallback for unresolved refs: {unresolved_nums}")
                            fallback_citations = self._parse_references_from_markdown(
                                str(references_previous), unresolved_nums
                            )
                            for num, citation in fallback_citations.items():
                                print(f"  [Fallback] Searching PMID for ref ({num}): {citation[:80]}...")
                                ref_pmid = self.citation_searcher.search_pmid_by_citation(citation)
                                if ref_pmid and ref_pmid != pmid:
                                    ref_pmids.append(ref_pmid)
                                    self._ref_pmid_to_citation[ref_pmid] = f"({num}) {citation}"
                        if ref_pmids:
                            return list(dict.fromkeys(ref_pmids))  # deduplicate, preserve order

        # --- Priority 2: Direct PMID extraction from text ---
        ref_pmids = self.citation_searcher.extract_referenced_pmids(references_previous)

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
            # --- Priority 3: Citation-based search (author/title/DOI → PubMed query) ---
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
                self._log_chase_miss(pmid, "reference", f"unresolved citation: {str(references_previous)[:150]}")

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
            references_previous, result.get("original_paragraph"), pmid,
            reference_number_in_text=result.get("reference_number_in_text")
        )
        if not ref_pmids:
            return result

        found_any = False
        seen_pmids = set()
        for ref_pmid in ref_pmids[:3]:
            if ref_pmid in seen_pmids:
                continue
            seen_pmids.add(ref_pmid)

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
                found_any = True

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
