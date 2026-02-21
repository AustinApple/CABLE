"""
Prompt templates for QA verification of structured_description against original_paragraph.

The QA agent checks whether Step 2's structured output faithfully represents
the source text from Step 1, detecting:
- Incorrect values (contradicts the text)
- Hallucinated values (not supported by text)
- Missing information (present in text but not captured)
"""

import json
from typing import Dict


def get_field_descriptions_from_schema(schema: Dict) -> Dict[str, str]:
    """
    Flatten a JSON schema into a mapping of dotted field paths to descriptions.

    E.g., {"instrument.manufacturer": "e.g., GE Healthcare, Cytiva", ...}

    Args:
        schema: The JSON schema dict (spr_schema.json or itc_schema.json)

    Returns:
        Dict mapping "section.field" -> description string
    """
    field_descriptions = {}
    properties = schema.get("properties", {})

    for section_name, section_def in properties.items():
        # Skip top-level non-object fields like "pmid"
        if section_def.get("type") != "object":
            continue

        section_props = section_def.get("properties", {})
        for field_name, field_def in section_props.items():
            path = f"{section_name}.{field_name}"

            # Build description from schema metadata
            desc_parts = []
            if "description" in field_def:
                desc_parts.append(field_def["description"])
            if "examples" in field_def:
                examples = field_def["examples"]
                if isinstance(examples, list):
                    desc_parts.append(f"Examples: {', '.join(str(e) for e in examples[:3])}")
            if "enum" in field_def:
                desc_parts.append(f"Allowed values: {', '.join(str(e) for e in field_def['enum'])}")

            # Handle nested objects (e.g., assay_conditions.assay_flow_rate, injection_parameters)
            if field_def.get("type") == "object":
                nested_props = field_def.get("properties", {})
                for nested_name, nested_def in nested_props.items():
                    nested_path = f"{section_name}.{field_name}.{nested_name}"
                    nested_desc = nested_def.get("description", "")
                    field_descriptions[nested_path] = nested_desc
                # Also add the parent object description
                if desc_parts:
                    field_descriptions[path] = " | ".join(desc_parts)
            else:
                field_descriptions[path] = " | ".join(desc_parts) if desc_parts else ""

    return field_descriptions


def _flatten_structured_description(structured: Dict, prefix: str = "") -> Dict[str, object]:
    """
    Flatten a nested structured_description dict into dotted-path keys.

    E.g., {"instrument": {"manufacturer": "X"}} -> {"instrument.manufacturer": "X"}
    """
    flat = {}
    for key, value in structured.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_structured_description(value, path))
        else:
            flat[path] = value
    return flat


def compute_overall_score(field_verdicts: Dict, missing_information: list) -> float:
    """
    Compute overall quality score from field verdicts.

    Scoring:
    - Start at 10
    - Subtract 1.0 for each "incorrect" field
    - Subtract 0.5 for each "unsupported" field
    - Subtract 0.5 for each "missing" field
    - Minimum score is 0

    Args:
        field_verdicts: Dict of field_path -> {verdict: str, ...}
        missing_information: List of missing information items

    Returns:
        Float score 0-10
    """
    score = 10.0
    for field_path, verdict_info in field_verdicts.items():
        verdict = verdict_info.get("verdict", "")
        if verdict == "incorrect":
            score -= 1.0
        elif verdict == "unsupported":
            score -= 0.5
        elif verdict == "missing":
            score -= 0.5

    # Also penalize for missing information not captured in any field
    score -= 0.25 * len(missing_information)

    return max(0.0, round(score, 2))


def count_verdicts(field_verdicts: Dict) -> Dict[str, int]:
    """
    Count verdict types from field_verdicts.

    Args:
        field_verdicts: Dict of field_path -> {verdict: str, ...}

    Returns:
        Dict with counts for each verdict type and total
    """
    counts = {"correct": 0, "incorrect": 0, "unsupported": 0, "missing": 0}
    for verdict_info in field_verdicts.values():
        verdict = verdict_info.get("verdict", "")
        if verdict in counts:
            counts[verdict] += 1
    counts["total_fields"] = sum(counts.values())
    return counts


def get_qa_verification_prompt(
    original_paragraph: str,
    structured_description: Dict,
    assay_description: str,
    schema_fields: Dict[str, str],
    assay_type: str = "spr"
) -> str:
    """
    Build the QA verification prompt for a thinking model.

    Args:
        original_paragraph: Combined text from original_paragraph dict
        structured_description: The JSON structured_description to verify
        assay_description: The DESCRIPTION field from the entry
        schema_fields: Field path -> description mapping from schema
        assay_type: "spr" or "itc"

    Returns:
        Formatted prompt string
    """
    assay_type_full = (
        "Surface Plasmon Resonance (SPR)" if assay_type == "spr"
        else "Isothermal Titration Calorimetry (ITC)"
    )

    # Flatten the structured description for display
    flat_structured = _flatten_structured_description(structured_description)

    # Build field list with descriptions and extracted values
    field_list_lines = []
    for path, desc in sorted(schema_fields.items()):
        extracted_val = flat_structured.get(path, "<not in extraction>")
        field_list_lines.append(f"  - {path}: {desc}")
        field_list_lines.append(f"    Extracted value: {json.dumps(extracted_val)}")
    field_list_str = "\n".join(field_list_lines)

    return f"""You are a scientific QA reviewer specializing in {assay_type_full} experiments. Your task is to verify whether a structured JSON extraction accurately and completely represents information from the original text.

=== ASSAY DESCRIPTION ===
{assay_description}

=== ORIGINAL TEXT FROM PAPER ===
{original_paragraph}

=== STRUCTURED EXTRACTION TO VERIFY ===
{json.dumps(structured_description, indent=2)}

=== FIELDS AND THEIR EXTRACTED VALUES ===
{field_list_str}

Instructions:
1. For EACH field listed above, determine its verdict:
   - "correct": The extracted value is clearly supported by the original text
   - "incorrect": The extracted value contradicts or misrepresents the text
   - "unsupported": The extracted value cannot be verified from the text (may be hallucinated or inferred without evidence)
   - "missing": The field is null/empty in the extraction but the information IS present in the original text

2. For each verdict, provide a brief "evidence" quote from the original text that supports your judgment (or state "not found in text" for unsupported values).

3. After checking all extracted fields, scan the original text for any important experimental details that were NOT captured in any structured field. Report these as "missing_information".

Output ONLY valid JSON in this exact format:
{{
    "overall_summary": "<1-2 sentence summary of quality>",
    "field_verdicts": {{
        "<field.path>": {{
            "extracted_value": <the value from the extraction>,
            "verdict": "correct|incorrect|unsupported|missing",
            "evidence": "<quote from text or 'not found in text'>"
        }}
    }},
    "missing_information": [
        {{
            "description": "<what information is missing>",
            "evidence": "<quote from text>",
            "relevant_field": "<which field should have captured this>"
        }}
    ]
}}

Respond ONLY with valid JSON, no other text."""
