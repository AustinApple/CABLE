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
