"""
Assay Extraction Agent - Two-Step Version

Extracts assay descriptions from papers using a two-step approach:
- Step 1: Extract original_paragraph using vision model (with paper images)
- Step 2: Fill structured_description from extracted text only (no images)

This allows Step 2 to use cheaper/faster text-only processing.

Features:
- Automatic supplementary material fetching from PubMed Central
- Tracks token usage for analysis
- Can run each step independently
- Optional separate text model for Step 2
"""
import torch
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import time

from .base_extraction_agent import BaseAssayExtractionAgent
from .prompts_two_step import get_paragraph_extraction_prompt, get_structured_description_from_text_prompt

# Type alias for prompt functions
from typing import Callable
ParagraphPromptFn = Callable[[str], str]
StructuredPromptFn = Callable[..., str]


class TwoStepAssayExtractionAgent(BaseAssayExtractionAgent):
    """Two-step agent for extracting assay information.

    Step 1: Extract original_paragraph using vision model + paper images
    Step 2: Fill structured_description using text-only input

    Benefits:
    - Step 2 is cheaper/faster (no image tokens)
    - Can use different models for each step
    - Can run steps independently
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Thinking",
        text_model_name: Optional[str] = None,
        pdf_dir: str = "./downloaded_paper_kd",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        temperature: float = 0.0,
        search_supplementary: bool = True,
        search_references: bool = True,
        max_reference_depth: int = 1,
        ncbi_api_key: Optional[str] = None,
        paragraph_prompt_fn: Optional['ParagraphPromptFn'] = None,
        structured_prompt_fn: Optional['StructuredPromptFn'] = None
    ):
        """
        Initialize the two-step extraction agent.

        Args:
            model_name: Hugging Face model name for Qwen3-VL (vision model for Step 1)
            text_model_name: Optional separate model for text-only Step 2.
                           If None, uses the vision model with text-only input.
            pdf_dir: Directory containing PDF files
            device: Device to run the model on ('cuda' or 'cpu')
            torch_dtype: Data type for model weights ('bfloat16', 'float16', or 'float32')
            temperature: Sampling temperature (0.0 = greedy/deterministic, >0 = more random)
            search_supplementary: Whether to automatically fetch and search supplementary materials
            search_references: Whether to fetch referenced papers if assay not found
            max_reference_depth: Maximum depth for recursive reference search (1 = only direct refs)
            ncbi_api_key: NCBI API key for higher rate limits (optional)
            paragraph_prompt_fn: Custom prompt function for Step 1 (paragraph extraction).
                               Signature: fn(assay_description: str) -> str.
                               Defaults to SPR prompt if None.
            structured_prompt_fn: Custom prompt function for Step 2 (structured description).
                                Signature: fn(extracted_paragraph: str, assay_description: str, ...) -> str.
                                Defaults to SPR prompt if None.
        """
        # Initialize base class
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

        # Store custom prompt functions (default to SPR prompts)
        self.paragraph_prompt_fn = paragraph_prompt_fn or get_paragraph_extraction_prompt
        self.structured_prompt_fn = structured_prompt_fn or get_structured_description_from_text_prompt

        # Load separate text model for Step 2 if specified
        self.text_model = None
        self.text_tokenizer = None
        self.use_separate_text_model = False

        if text_model_name and text_model_name != model_name:
            print(f"Loading separate text model for Step 2: {text_model_name}")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.text_model = AutoModelForCausalLM.from_pretrained(
                text_model_name,
                torch_dtype=self.torch_dtype_obj,
                device_map="auto",
                trust_remote_code=True
            )
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                text_model_name,
                trust_remote_code=True
            )
            self.use_separate_text_model = True
            print(f"Text model loaded: {text_model_name}")

    # ==================== Text-Only Query Method ====================

    def _query_text_model(
        self,
        prompt: str,
        max_new_tokens: int = 2048
    ) -> Tuple[str, int, int]:
        """
        Query model with text-only input (no images).

        This is used for Step 2 where we only need to process the extracted text.

        Args:
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        if self.use_separate_text_model and self.text_model is not None:
            # Use dedicated text model
            inputs = self.text_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            input_tokens = inputs["input_ids"].shape[1]

            with torch.no_grad():
                if self.temperature == 0.0:
                    generated_ids = self.text_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False
                    )
                else:
                    generated_ids = self.text_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature
                    )

            generated_ids_trimmed = generated_ids[0][input_tokens:]
            response_text = self.text_tokenizer.decode(
                generated_ids_trimmed,
                skip_special_tokens=True
            )
            output_tokens = len(generated_ids_trimmed)
        else:
            # Use vision model with text-only input (no images)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process without images
            inputs = self.processor(
                text=[text_prompt],
                images=None,
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

        print(f"  [Text Query] Input: {input_tokens:,} | Output: {output_tokens:,} | Total: {input_tokens + output_tokens:,}")
        print(f"  [Accumulated] Input: {self.token_usage['total_input_tokens']:,} | Output: {self.token_usage['total_output_tokens']:,} | Total: {self.token_usage['total_tokens']:,} | Requests: {self.token_usage['requests']}")

        return response_text, input_tokens, output_tokens

    # ==================== Step 1: Paragraph Extraction ====================

    def extract_paragraph(
        self,
        pmid: str,
        assay_description: str,
        max_pages: Optional[int] = None,
        max_new_tokens: int = 2048,
        _depth: int = 0
    ) -> Dict:
        """
        Step 1: Extract original_paragraph from paper using vision model.

        This method extracts the relevant paragraphs from the paper without
        filling in structured_description fields.

        Args:
            pmid: PubMed ID of the paper
            assay_description: Brief assay description from BindingDB
            max_pages: Maximum number of pages to process (None for all)
            max_new_tokens: Maximum tokens to generate
            _depth: Internal parameter for tracking reference search depth

        Returns:
            Dictionary with 'original_paragraph', 'confidence', 'references_previous',
            'source', 'search_path', 'supplementary_source' keys
        """
        print(f"\n{'='*60}")
        print(f"[Step 1] Extracting paragraph from PMID {pmid} (depth={_depth})")
        print(f"{'='*60}")

        images, pdf_path = self._get_paper_images(pmid, max_pages)

        if pdf_path is None:
            return {
                "pmid": pmid,
                "assay_description": assay_description,
                "original_paragraph": {"error": "PDF not found and could not be fetched"},
                "confidence": "N/A",
                "source": "N/A",
                "search_path": [],
                "supplementary_source": [],
                "references_previous": "none"
            }

        if images is None or len(images) == 0:
            return {
                "pmid": pmid,
                "assay_description": assay_description,
                "original_paragraph": {"error": "PDF conversion failed"},
                "confidence": "N/A",
                "source": "N/A",
                "search_path": [],
                "supplementary_source": [],
                "references_previous": "none"
            }

        # Use paragraph extraction prompt (no structured_description)
        prompt = self.paragraph_prompt_fn(assay_description)
        supp_files = []

        try:
            # Search main paper
            print(f"  Searching main paper for PMID {pmid}...")
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
                return result

            # Search supplementary materials
            if self.search_supplementary:
                print(f"\n  Not found in main paper. Fetching supplementary materials...")
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
                            supp_result["search_path"] = ["main", "supplementary"]
                            supp_result["supplementary_source"] = ["main"]
                            print(f"  Found in supplementary: {supp_path.name}!")
                            return supp_result

            # Return NOT FOUND
            print(f"  Paragraph not found in any searched documents.")
            return {
                "pmid": pmid,
                "assay_description": assay_description,
                "original_paragraph": {},
                "confidence": "N/A",
                "source": "not_found",
                "search_path": [],
                "supplementary_source": [],
                "references_previous": "none",
                "searched_locations": f"main_paper, supplementary({len(supp_files)} files)"
            }

        except Exception as e:
            print(f"Error extracting paragraph for PMID {pmid}: {e}")
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
                "references_previous": "none"
            }

    # ==================== Step 2: Structured Description ====================

    def fill_structured_description(
        self,
        extracted_paragraph: Dict[str, str],
        assay_description: str,
        protein: Optional[str] = None,
        ligand_smiles: Optional[str] = None,
        affinity_data: Optional[Dict] = None,
        max_new_tokens: int = 2048
    ) -> Dict:
        """
        Step 2: Fill structured_description from extracted paragraph text.

        This method takes text only (no images needed), making it more efficient.
        Can be used independently on pre-extracted paragraphs.

        Note: Since one DESCRIPTION maps to one original_paragraph, the structured
        description is about the assay methodology and is the same for all
        protein-ligand pairs. The protein, ligand_smiles, and affinity_data
        parameters are kept for backward compatibility but are not used.

        Args:
            extracted_paragraph: Dict mapping location -> text (from Step 1)
            assay_description: Brief assay description
            protein: Deprecated - not used
            ligand_smiles: Deprecated - not used
            affinity_data: Deprecated - not used
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dict with 'structured_description' key
        """
        print(f"\n{'='*60}")
        print(f"[Step 2] Filling structured_description from text")
        print(f"{'='*60}")

        # Combine paragraph text from dict
        if isinstance(extracted_paragraph, dict):
            combined_text = "\n\n".join(
                f"[{loc}]: {text}" for loc, text in extracted_paragraph.items()
                if text and str(text).strip()
            )
        else:
            # Handle legacy string format
            combined_text = str(extracted_paragraph)

        if not combined_text.strip():
            print("  No text to process, returning null structured_description")
            return {"structured_description": None}

        # Build prompt for text-only model (protein/ligand/affinity not used)
        prompt = self.structured_prompt_fn(
            extracted_paragraph=combined_text,
            assay_description=assay_description
        )

        try:
            # Query text model (no images!)
            response_text, _, _ = self._query_text_model(prompt, max_new_tokens)

            print(f"\n[DEBUG] Raw response: {response_text[:500]}...")

            result = self._parse_response(response_text)
            if result:
                return result
            else:
                return {"structured_description": None}

        except Exception as e:
            print(f"Error filling structured_description: {e}")
            import traceback
            traceback.print_exc()
            return {"structured_description": None}

    # ==================== Combined Two-Step Pipeline ====================

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
        Full two-step extraction pipeline.

        Step 1: Extract paragraphs from paper images
        Step 2: Fill structured_description from extracted text

        Note: Since one DESCRIPTION maps to one original_paragraph, the structured
        description is the same for all protein-ligand pairs. The protein,
        ligand_smiles, and affinity_data parameters are kept for backward
        compatibility but are not used.

        Args:
            pmid: PubMed ID of the paper
            assay_description: Brief assay description
            protein: Deprecated - not used
            ligand_smiles: Deprecated - not used
            affinity_data: Deprecated - not used
            max_pages: Maximum pages to process
            max_new_tokens: Maximum tokens to generate
            _depth: Internal depth for reference search

        Returns:
            Dict with 'original_paragraph', 'structured_description', and metadata
        """
        print(f"\n{'='*60}")
        print(f"[TWO-STEP MODE] Extracting assay from PMID {pmid}")
        print(f"{'='*60}")

        # Step 1: Extract paragraphs
        step1_result = self.extract_paragraph(
            pmid=pmid,
            assay_description=assay_description,
            max_pages=max_pages,
            max_new_tokens=max_new_tokens,
            _depth=_depth
        )

        # Check if Step 1 found anything
        if not self._is_paragraph_found(step1_result.get("original_paragraph")):
            print(f"  Step 1 did not find relevant paragraphs")
            return {
                **step1_result,
                "structured_description": None
            }

        print(f"  Step 1 completed, proceeding to Step 2...")

        # Step 2: Fill structured_description from extracted text
        # Note: protein/ligand/affinity not used - assay methodology is the same for all pairs
        step2_result = self.fill_structured_description(
            extracted_paragraph=step1_result.get("original_paragraph", {}),
            assay_description=assay_description,
            max_new_tokens=max_new_tokens
        )

        # Combine results
        final_result = {
            **step1_result,
            "structured_description": step2_result.get("structured_description")
        }

        print(f"  Two-step extraction completed for PMID {pmid}")
        return final_result

    # ==================== DataFrame Processing ====================

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
            json_path = output_path / f"{pmid_str}.json"

            # Skip if PMID already processed (JSON file exists)
            if json_path.exists():
                print(f"\n{'='*80}")
                print(f"[SKIP] PMID {pmid_str} already exists: {json_path}")
                saved_files[pmid_str] = json_path
                row_count += len(group)  # Update row count for progress tracking
                continue

            pmid_results = {}

            # Cache extraction results by DESCRIPTION to avoid redundant extractions
            # Since one DESCRIPTION maps to one original_paragraph, we can reuse results
            description_cache: Dict[str, Dict] = {}

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

                # Check if we already extracted for this DESCRIPTION
                if description in description_cache:
                    print(f"  [CACHE HIT] Reusing extraction result for same DESCRIPTION")
                    result = description_cache[description]
                else:
                    # First time seeing this DESCRIPTION, run full extraction
                    result = self.extract_assay_description(
                        pmid_str, description,
                        protein=protein,
                        ligand_smiles=ligand_smiles,
                        affinity_data=affinity_data,
                        max_pages=max_pages
                    )
                    # Cache the result for this DESCRIPTION
                    description_cache[description] = result

                    if delay > 0:
                        time.sleep(delay)

                entry = {
                    "reactant_set_id": int(reactant_set_id) if reactant_set_id else None,
                    "pmid": int(pmid),
                    "protein": protein,
                    "ligand": {"reference_name": ligand_name, "smiles": ligand_smiles},
                    "affinity_data": affinity_data,
                    "DESCRIPTION": description,
                    "search_path": result.get("search_path", "N/A"),
                    "supplementary_source": result.get("supplementary_source"),
                    "references_previous": result.get("references_previous"),
                    "original_paragraph": result.get("original_paragraph", {}),
                    "structured_description": result.get("structured_description")
                }

                key = str(reactant_set_id) if reactant_set_id else f"entry_{row_count}"
                pmid_results[key] = entry

            # json_path already defined at the start of the loop
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(pmid_results, f, indent=4, ensure_ascii=False)
            print(f"\nSaved: {json_path}")
            saved_files[pmid_str] = json_path

        self._print_token_summary()
        print(f"\nSaved {len(saved_files)} JSON files to {output_dir}")

        return saved_files

    def _print_token_summary(self):
        """Print token usage summary."""
        print(f"\n{'='*80}")
        print("TOKEN USAGE SUMMARY (Two-Step)")
        print(f"{'='*80}")
        print(f"Total Requests: {self.token_usage['requests']}")
        print(f"Total Input Tokens: {self.token_usage['total_input_tokens']:,}")
        print(f"Total Output Tokens: {self.token_usage['total_output_tokens']:,}")
        print(f"Total Tokens: {self.token_usage['total_tokens']:,}")
        if self.token_usage['requests'] > 0:
            print(f"Avg Input/Request: {self.token_usage['total_input_tokens'] / self.token_usage['requests']:,.0f}")
            print(f"Avg Output/Request: {self.token_usage['total_output_tokens'] / self.token_usage['requests']:,.0f}")
        print(f"{'='*80}")
