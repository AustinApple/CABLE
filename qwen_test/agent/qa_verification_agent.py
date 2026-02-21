"""
QA Verification Agent for checking structured_description against original_paragraph.

This is a text-only agent (no vision model needed) that uses an instruct model
to systematically verify whether each field in structured_description is
supported by the original_paragraph text.

Usage:
    agent = QAVerificationAgent(
        model_name="Qwen/Qwen3-30B-A3B",
        assay_type="spr"
    )
    result = agent.verify_entry(original_paragraph, structured_description, description)
    agent.cleanup()
"""

import torch
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple

from .prompts_qa import (
    get_field_descriptions_from_schema,
    get_qa_verification_prompt,
    compute_overall_score,
    count_verdicts,
)
from .base_extraction_agent import BaseAssayExtractionAgent


class QAVerificationAgent:
    """Text-only agent that verifies structured_description against original_paragraph.

    Does NOT inherit from BaseAssayExtractionAgent because that loads a vision model.
    Instead, loads a text-only CausalLM model directly.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-30B-A3B",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        temperature: float = 0.6,
        assay_type: str = "spr",
    ):
        """
        Initialize the QA verification agent.

        Args:
            model_name: HuggingFace model name for a text-only instruct model
            device: Device to run on ('cuda', 'cuda:0', etc.)
            torch_dtype: Data type for model weights
            temperature: Sampling temperature
            assay_type: "spr" or "itc"
        """
        print(f"Loading QA verification model: {model_name}")
        print(f"Device: {device}, dtype: {torch_dtype}, temperature: {temperature}")

        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.assay_type = assay_type
        self.max_new_tokens = 4096

        # Map string dtype to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype_obj = dtype_map.get(torch_dtype, torch.bfloat16)

        # Load text-only model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype_obj,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        print(f"Model loaded: {model_name}")

        # Load schema
        schema_filename = "spr_schema.json" if assay_type == "spr" else "itc_schema.json"
        schema_path = Path(__file__).parent.parent.parent / schema_filename
        with open(schema_path, "r") as f:
            self.schema = json.load(f)
        self.schema_fields = get_field_descriptions_from_schema(self.schema)
        print(f"Schema loaded: {schema_path} ({len(self.schema_fields)} fields)")

        # Token usage tracking
        self.token_usage = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
        }

    def _query_model(self, prompt: str, max_new_tokens: int = 8192) -> Tuple[str, int, int]:
        """
        Query the text model with raw tokenizer input (no chat template).

        Args:
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True
        ).to(self.device)

        input_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            if self.temperature == 0.0:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            else:
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                )

        generated_ids_trimmed = generated_ids[0][input_tokens:]
        response_text = self.tokenizer.decode(
            generated_ids_trimmed, skip_special_tokens=True
        )
        output_tokens = len(generated_ids_trimmed)

        self.token_usage["total_input_tokens"] += input_tokens
        self.token_usage["total_output_tokens"] += output_tokens
        self.token_usage["total_tokens"] += input_tokens + output_tokens
        self.token_usage["requests"] += 1

        print(f"  [QA Query] Input: {input_tokens:,} | Output: {output_tokens:,} | Total: {input_tokens + output_tokens:,}")
        print(f"  [Accumulated] Input: {self.token_usage['total_input_tokens']:,} | Output: {self.token_usage['total_output_tokens']:,} | Requests: {self.token_usage['requests']}")

        return response_text, input_tokens, output_tokens

    @staticmethod
    def _parse_json_response(text: str) -> Optional[Dict]:
        """Parse JSON from model output, stripping markdown fences."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            result = json.loads(text)
            result = BaseAssayExtractionAgent._clean_string_values(result)
            return result
        except json.JSONDecodeError:
            return None

    def _combine_paragraph_text(self, original_paragraph: Dict) -> str:
        """Combine original_paragraph dict into a single text string."""
        if isinstance(original_paragraph, dict):
            return "\n\n".join(
                f"[{loc}]: {text}"
                for loc, text in original_paragraph.items()
                if text and str(text).strip()
            )
        return str(original_paragraph)

    def verify_entry(
        self,
        original_paragraph: Dict,
        structured_description: Dict,
        assay_description: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict:
        """
        Verify a single (original_paragraph, structured_description) pair.

        Args:
            original_paragraph: Dict mapping section -> text
            structured_description: The structured JSON to verify
            assay_description: The DESCRIPTION field
            max_new_tokens: Maximum tokens for model generation

        Returns:
            QA verification result dict
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        combined_text = self._combine_paragraph_text(original_paragraph)

        prompt = get_qa_verification_prompt(
            original_paragraph=combined_text,
            structured_description=structured_description,
            assay_description=assay_description,
            schema_fields=self.schema_fields,
            assay_type=self.assay_type,
        )

        # Query model
        response_text, _, _ = self._query_model(prompt, max_new_tokens)

        # Parse response
        parsed = self._parse_json_response(response_text)

        if parsed is None:
            # Retry once
            print("  [QA] First parse failed, retrying...")
            response_text, _, _ = self._query_model(prompt, max_new_tokens)
            parsed = self._parse_json_response(response_text)

        if parsed is None:
            return {
                "error": "Failed to parse model response",
                "raw_response": response_text[:500],
            }

        # Compute score and counts in code (not by model)
        field_verdicts = parsed.get("field_verdicts", {})
        missing_information = parsed.get("missing_information", [])

        overall_score = compute_overall_score(field_verdicts, missing_information)
        counts = count_verdicts(field_verdicts)

        return {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "overall_score": overall_score,
            "overall_summary": parsed.get("overall_summary", ""),
            "field_verdicts": field_verdicts,
            "missing_information": missing_information,
            "counts": counts,
        }

    def verify_result_file(
        self,
        json_path: Path,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Load a result JSON file, verify each unique (DESCRIPTION) entry,
        and save QA-annotated results.

        Args:
            json_path: Path to the extraction result JSON file
            output_path: Path to save QA-annotated output (if None, derived from json_path)

        Returns:
            Path to saved output file, or None on error
        """
        print(f"\n{'='*60}")
        print(f"[QA] Processing: {json_path.name}")
        print(f"{'='*60}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Group entries by DESCRIPTION to verify once per unique description
        desc_to_keys: Dict[str, list] = {}
        for key, entry in data.items():
            desc = entry.get("DESCRIPTION", "")
            if desc not in desc_to_keys:
                desc_to_keys[desc] = []
            desc_to_keys[desc].append(key)

        print(f"  {len(data)} entries, {len(desc_to_keys)} unique DESCRIPTIONs")

        # Verify each unique DESCRIPTION
        desc_qa_cache: Dict[str, Dict] = {}
        desc_count = 0

        for description, keys in desc_to_keys.items():
            desc_count += 1
            first_key = keys[0]
            entry = data[first_key]

            original_paragraph = entry.get("original_paragraph")
            structured_description = entry.get("structured_description")

            print(f"\n  --- DESCRIPTION {desc_count}/{len(desc_to_keys)} ({len(keys)} entries) ---")
            print(f"  {description[:100]}...")

            # Skip if no data to verify
            if not original_paragraph or (isinstance(original_paragraph, dict) and not original_paragraph):
                qa_result = {"skipped": True, "reason": "empty original_paragraph"}
                print(f"  [SKIP] Empty original_paragraph")
            elif structured_description is None:
                qa_result = {"skipped": True, "reason": "null structured_description"}
                print(f"  [SKIP] Null structured_description")
            elif isinstance(original_paragraph, dict) and "error" in original_paragraph:
                qa_result = {"skipped": True, "reason": "extraction_error"}
                print(f"  [SKIP] Extraction error: {original_paragraph.get('error', '')[:80]}")
            else:
                # Run QA verification
                qa_result = self.verify_entry(
                    original_paragraph=original_paragraph,
                    structured_description=structured_description,
                    assay_description=description,
                )
                if "overall_score" in qa_result:
                    print(f"  [QA] Score: {qa_result['overall_score']}/10 | {qa_result.get('counts', {})}")

            desc_qa_cache[description] = qa_result

        # Attach QA results to all entries
        for key, entry in data.items():
            desc = entry.get("DESCRIPTION", "")
            entry["qa_verification"] = desc_qa_cache.get(desc, {"skipped": True, "reason": "unknown"})

        # Save output
        if output_path is None:
            output_path = json_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"\n  Saved: {output_path}")

        return output_path

    def verify_directory(
        self,
        results_dir: Path,
        output_dir: Optional[Path] = None,
        skip_existing: bool = True,
    ) -> Dict[str, Path]:
        """
        Verify all JSON files in a results directory.

        Args:
            results_dir: Directory containing extraction result JSON files
            output_dir: Directory to save QA-annotated results (if None, overwrites originals)
            skip_existing: Skip files that already have QA results in output_dir

        Returns:
            Dict mapping filename -> output path
        """
        results_dir = Path(results_dir)
        json_files = sorted(results_dir.glob("*.json"))

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"[QA] Verifying {len(json_files)} files from: {results_dir}")
        if output_dir:
            print(f"[QA] Output to: {output_dir}")
        print(f"{'='*80}")

        saved_files = {}
        for i, json_path in enumerate(json_files, 1):
            out_path = (output_dir / json_path.name) if output_dir else json_path

            if skip_existing and out_path.exists():
                # Check if QA already done
                try:
                    with open(out_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    first_entry = next(iter(existing.values()), {})
                    if "qa_verification" in first_entry:
                        print(f"\n[{i}/{len(json_files)}] {json_path.name}: QA already exists, skipping")
                        saved_files[json_path.name] = out_path
                        continue
                except (json.JSONDecodeError, StopIteration):
                    pass

            print(f"\n[{i}/{len(json_files)}] Processing {json_path.name}")
            result_path = self.verify_result_file(json_path, output_path=out_path)
            if result_path:
                saved_files[json_path.name] = result_path

        self._print_token_summary()
        return saved_files

    def _print_token_summary(self):
        """Print token usage summary."""
        print(f"\n{'='*80}")
        print("TOKEN USAGE SUMMARY (QA Verification)")
        print(f"{'='*80}")
        print(f"Total Requests: {self.token_usage['requests']}")
        print(f"Total Input Tokens: {self.token_usage['total_input_tokens']:,}")
        print(f"Total Output Tokens: {self.token_usage['total_output_tokens']:,}")
        print(f"Total Tokens: {self.token_usage['total_tokens']:,}")
        if self.token_usage["requests"] > 0:
            avg_in = self.token_usage["total_input_tokens"] / self.token_usage["requests"]
            avg_out = self.token_usage["total_output_tokens"] / self.token_usage["requests"]
            print(f"Avg Input/Request: {avg_in:,.0f}")
            print(f"Avg Output/Request: {avg_out:,.0f}")
        print(f"{'='*80}")

    def cleanup(self):
        """Free GPU memory."""
        print("\nCleaning up QA agent GPU memory...")
        del self.model
        del self.tokenizer
        if "cuda" in self.device:
            torch.cuda.empty_cache()
        print("Cleanup complete!")
