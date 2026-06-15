"""
Prompt templates for the assay extraction agent.
"""

from typing import Optional, Dict, List


def get_structured_assay_extraction_prompt(
    assay_description: str,
    protein: Optional[str] = None,
    ligand_smiles: Optional[str] = None,
    affinity_data: Optional[Dict] = None
) -> str:
    """
    Get the prompt for extracting structured assay information from a paper.

    Args:
        assay_description: Brief assay description from BindingDB
        protein: Name of the protein target
        ligand_smiles: SMILES string for the ligand
        affinity_data: Dict with keys: type, value, relation, unit

    Returns:
        Formatted prompt string
    """
    # Build context block from available metadata
    context_parts = [f"Assay Description: {assay_description}"]
    if protein:
        context_parts.append(f"Protein Target: {protein}")
    if ligand_smiles:
        context_parts.append(f"Ligand SMILES: {ligand_smiles}")
    if affinity_data:
        aff_str = (
            f"{affinity_data.get('type', 'Kd')} "
            f"{affinity_data.get('relation', '=')} "
            f"{affinity_data.get('value', '')} "
            f"{affinity_data.get('unit', 'nM')}"
        )
        context_parts.append(f"Reported Affinity: {aff_str}")
    context_block = "\n".join(context_parts)

    return f"""You are an expert scientific reader analyzing a research paper about Surface Plasmon Resonance (SPR) experiments.

Task: Find the experimental section describing the SPR binding assay for the following experiment, extract the COMPLETE ORIGINAL text, AND extract structured assay parameters.

{context_block}

Instructions:
1. Search through the paper to find where this SPR assay is described (focus on Experimental Methods or Methods section first, then Results, then figure captions).
2. Extract the COMPLETE ORIGINAL text that contains the full experimental protocol. Include:
   - Methods/Experimental section paragraphs
   - Data visible in figures: extract numerical values shown in sensorgrams or plots (e.g., association/dissociation times from x-axis, concentration values, pH values labeled on curves)
   - Relevant figure captions (e.g., sensorgram descriptions, concentration ranges)
   - Relevant table contents if applicable
   - Do NOT summarize or paraphrase - provide the exact text from the paper
3. Extract structured assay parameters from the text into the fields described below.
4. If a parameter is not mentioned in the paper, use null.
5. Extract conditions SPECIFIC to the compound/ligand identified above, not general paper-level conditions (e.g., if different compounds are tested at different pH or concentration ranges, extract the ones for this specific compound).

6. If the paper references a previous publication for methodology details (e.g., "as described previously", "following [ref]", "according to [author]"):

   CRITICAL STEPS FOR REFERENCE EXTRACTION:
   a) Note the EXACT reference number mentioned in the methods text
   b) Go to the References/Bibliography section at the END of the paper
   c) Find the reference entry that starts with EXACTLY that number
   d) VERIFY the number matches before copying
   e) Copy the FULL citation including: authors, title, journal name, year, volume, and page numbers

   IMPORTANT: If you cannot find the exact numbered reference, write "Reference [X] not found in bibliography".

STRUCTURED PARAMETERS TO EXTRACT:

instrument:
  - manufacturer: Instrument manufacturer (e.g., "GE Healthcare", "Cytiva", "BIAcore AB")
  - model: Instrument model (e.g., "Biacore T200", "Biacore 8K", "BIAcore 3000")

sensor_chip:
  - type: Chip surface type (e.g., "CM5", "CM7", "SA", "NTA", "Series S Sensor Chip NTA")
  - manufacturer: Chip manufacturer (e.g., "GE Healthcare", "Cytiva", "Biacore")

immobilization:
  - ligand_name: Name of the molecule immobilized on the chip (e.g., "his-tagged HDAC8", "Biotinylated AcrB")
  - strategy: Immobilization method (e.g., "Amine coupling", "SA-Biotin capture", "NTA-His capture")
  - density_ru: Immobilization level in Response Units (number or range, e.g., 2040, "6000-8000")
  - concentration_for_immobilization: Concentration used for immobilization with units (e.g., "50 µg/mL"), null if not stated

analyte:
  - description: Description of the analyte being tested (e.g., "MC-207,110 (efflux pump inhibitor)")
  - concentration_range: Concentration range tested with units (e.g., "12.5 µM – 200 µM (two-fold dilutions)")

assay_conditions:
  - running_buffer_composition: Full buffer composition including pH and additives
  - pH: pH value as a number (e.g., 7.5, 6.0)
  - assay_type: One of: "Single Cycle Kinetics", "Multi Cycle Kinetics", "Single Cycle Steady-State Affinity", "Multi Cycle Steady-State Affinity", "Screening", "Unknown"
  - assay_flow_rate: Object with "value" (number) and "unit" (one of: "µL/min", "ml/min", "µL/s")
  - temperature_c: Temperature in Celsius (number or null)
  - association_time_s: Association/contact time in seconds (number or null)
  - dissociation_time_s: Dissociation time in seconds (number or null)
  - regeneration_solution: Composition of regeneration solution

data_analysis:
  - reference_description: Description of the reference surface/channel
  - subtraction_method: Data normalization method (e.g., "Double referenced", "Reference-subtracted")
  - fitting_model: Kinetic/affinity model used (e.g., "1:1 kinetic model", "Steady-state affinity")
  - software: Analysis software (e.g., "BiaEvaluation 3.0", "Biacore 8K Insight Evaluation Software")

Output format (JSON):
{{{{
    "original_paragraph": {{{{
        "<descriptive_key>": "<extracted_text>",
        ...
    }}}},
    "confidence": "high/medium/low",
    "reference_number_in_text": "Reference number(s) or 'none'",
    "references_previous": "Complete citation from References section or 'none'",
    "structured_description": {{{{
        "instrument": {{{{"manufacturer": "...", "model": "..."}}}},
        "sensor_chip": {{{{"type": "...", "manufacturer": "..."}}}},
        "immobilization": {{{{
            "ligand_name": "...",
            "strategy": "...",
            "density_ru": ...,
            "concentration_for_immobilization": "..." or null
        }}}},
        "analyte": {{{{
            "description": "...",
            "concentration_range": "..."
        }}}},
        "assay_conditions": {{{{
            "running_buffer_composition": "...",
            "pH": ...,
            "assay_type": "...",
            "assay_flow_rate": {{{{"value": ..., "unit": "..."}}}},
            "temperature_c": ... or null,
            "association_time_s": ... or null,
            "dissociation_time_s": ... or null,
            "regeneration_solution": "..."
        }}}},
        "data_analysis": {{{{
            "reference_description": "...",
            "subtraction_method": "...",
            "fitting_model": "...",
            "software": "..."
        }}}}
    }}}}
}}}}

Note: For original_paragraph, use descriptive keys that identify where the text came from (e.g., "Materials and Methods", "SPR Analysis", "Figure 2 caption", "Supporting Information S1"). The keys are flexible - use whatever accurately describes the source location.

If you cannot find a matching SPR assay description, return:
{{{{
    "original_paragraph": {{{{}}}},
    "confidence": "N/A",
    "reference_number_in_text": "none",
    "references_previous": "none",
    "structured_description": null
}}}}

Please respond ONLY with valid JSON, no other text."""


def get_assay_extraction_prompt(assay_description: str) -> str:
    """
    Get the prompt for extracting assay descriptions from a paper.

    Args:
        assay_description: Brief assay description from BindingDB

    Returns:
        Formatted prompt string
    """
    return f"""You are an expert scientific reader analyzing a research paper.

Task: Find and extract the COMPLETE ORIGINAL PARAGRAPH that describes the following assay:

Assay Description: {assay_description}

Instructions:
1. Search through the paper to find where this assay is described (focus on Experimental Methods or Methods section first if available)
2. Extract the COMPLETE ORIGINAL paragraph(s) that contain the full experimental protocol
3. Include ALL details: concentrations, conditions, temperatures, times, buffers, equipment, etc.
4. Do NOT summarize or paraphrase - provide the exact text from the paper
5. If the description spans multiple paragraphs, include all relevant paragraphs
6. Indicate the location in the paper (e.g., "Methods section, page X" or "Results section")
7. If the paper references a previous publication for methodology details (e.g., "as described previously", "following [ref]", "according to [author]"):

   CRITICAL STEPS FOR REFERENCE EXTRACTION:
   a) First, note the EXACT reference number mentioned in the methods text (e.g., if text says "as previously described [53]", the number is 53)
   b) Go to the References/Bibliography section at the END of the paper
   c) Find the reference entry that starts with EXACTLY that number (e.g., "53." or "(53)")
   d) VERIFY the number matches before copying - do NOT copy a different numbered reference
   e) Copy the FULL citation including: authors, title, journal name, year, volume, and page numbers

   Example: If methods say "following the protocol in [53]", find entry "53. Smith, J.; Jones, A. Title of Paper. J. Med. Chem. 2020, 63, 1234-1245." and copy that EXACT entry.

   IMPORTANT: Reference numbers in the text MUST match the number you extract from the reference list. If you cannot find the exact numbered reference, write "Reference [X] not found in bibliography" where X is the number.

Output format (JSON):
{{
    "original_paragraph": "The complete extracted text from the paper...",
    "location": "Where this text was found in the paper (section name, page, etc.)",
    "confidence": "high/medium/low - your confidence that this is the correct passage",
    "reference_number_in_text": "The exact reference number found in the methods text (e.g., '53' or '12,13'). Use 'none' if no reference is cited.",
    "references_previous": "The COMPLETE citation from the References section that matches the reference_number_in_text. MUST start with the same number. If no reference, use 'none'"
}}

If you cannot find a matching description, return:
{{
    "original_paragraph": "NOT FOUND",
    "location": "N/A",
    "confidence": "N/A",
    "reference_number_in_text": "none",
    "references_previous": "none"
}}

Please respond ONLY with valid JSON, no other text."""


def get_batched_structured_assay_extraction_prompt(pairs: List[Dict]) -> str:
    """
    Get the prompt for extracting structured assay information for multiple protein-ligand pairs
    from a single paper in one model call.

    Args:
        pairs: List of dicts, each containing:
            - id: reactant_set_id as identifier
            - assay_description: Brief assay description from BindingDB
            - protein: Name of the protein target
            - ligand_smiles: SMILES string for the ligand (optional)
            - affinity_data: Dict with keys: type, value, relation, unit (optional)

    Returns:
        Formatted prompt string
    """
    # Build the pairs listing block
    pairs_listing = []
    for i, pair in enumerate(pairs, 1):
        pair_id = pair.get("id", f"pair_{i}")
        desc = pair.get("assay_description", "N/A")
        protein = pair.get("protein", "N/A")
        ligand = pair.get("ligand_smiles", "N/A")
        aff = pair.get("affinity_data")

        pair_block = f"""Pair ID: {pair_id}
  Assay Description: {desc}
  Protein: {protein}
  Ligand SMILES: {ligand if ligand else "N/A"}"""

        if aff:
            aff_str = f"{aff.get('type', 'Kd')} {aff.get('relation', '=')} {aff.get('value', '')} {aff.get('unit', 'nM')}"
            pair_block += f"\n  Reported Affinity: {aff_str}"

        pairs_listing.append(pair_block)

    pairs_block = "\n\n".join(pairs_listing)

    # Generate example output keys based on first few pair IDs
    example_ids = [str(pair.get("id", f"pair_{i}")) for i, pair in enumerate(pairs[:2], 1)]
    if len(example_ids) < 2:
        example_ids = ["51234567", "51234568"]

    return f"""You are an expert scientific reader analyzing a research paper about Surface Plasmon Resonance (SPR) experiments.

Task: Extract structured SPR assay information for MULTIPLE protein-ligand pairs from this paper. Each pair may have been tested under the same or different conditions.

=== PROTEIN-LIGAND PAIRS TO EXTRACT ===
{pairs_block}

=== INSTRUCTIONS ===
1. Search the paper to find where SPR assays are described (Methods/Experimental section, Results, figure captions).
2. For EACH protein-ligand pair listed above, extract the complete original text AND structured parameters.
3. Include in original_paragraph:
   - Methods/Experimental section paragraphs
   - Data visible in figures (e.g., association/dissociation times from x-axis, concentration values)
   - Relevant figure captions
   - Relevant table contents
4. Extract structured parameters specific to each pair where applicable.
5. If a parameter is not mentioned, use null.
6. If the paper references a previous publication for methodology, note the reference.

=== STRUCTURED PARAMETERS TO EXTRACT (for each pair) ===

instrument:
  - manufacturer: Instrument manufacturer (e.g., "GE Healthcare", "Cytiva")
  - model: Instrument model (e.g., "Biacore T200", "Biacore 8K")

sensor_chip:
  - type: Chip surface type (e.g., "CM5", "CM7", "SA", "NTA")
  - manufacturer: Chip manufacturer

immobilization:
  - ligand_name: Name of molecule immobilized on chip
  - strategy: Immobilization method (e.g., "Amine coupling", "SA-Biotin capture")
  - density_ru: Immobilization level in Response Units
  - concentration_for_immobilization: Concentration used for immobilization

analyte:
  - description: Description of the analyte being tested
  - concentration_range: Concentration range tested with units

assay_conditions:
  - running_buffer_composition: Full buffer composition including pH and additives
  - pH: pH value as number
  - assay_type: One of: "Single Cycle Kinetics", "Multi Cycle Kinetics", "Single Cycle Steady-State Affinity", "Multi Cycle Steady-State Affinity", "Screening", "Unknown"
  - assay_flow_rate: Object with "value" and "unit"
  - temperature_c: Temperature in Celsius
  - association_time_s: Association/contact time in seconds
  - dissociation_time_s: Dissociation time in seconds
  - regeneration_solution: Composition of regeneration solution

data_analysis:
  - reference_description: Description of reference surface/channel
  - subtraction_method: Data normalization method
  - fitting_model: Kinetic/affinity model used
  - software: Analysis software

=== OUTPUT FORMAT (JSON) ===
Return a JSON object where each key is a Pair ID and the value contains that pair's extraction results:

{{{{
    "{example_ids[0]}": {{{{
        "original_paragraph": {{{{
            "<descriptive_key>": "<extracted_text>",
            ...
        }}}},
        "confidence": "high/medium/low",
        "references_previous": "Complete citation or 'none'",
        "structured_description": {{{{
            "instrument": {{{{"manufacturer": "...", "model": "..."}}}},
            "sensor_chip": {{{{"type": "...", "manufacturer": "..."}}}},
            "immobilization": {{{{...}}}},
            "analyte": {{{{...}}}},
            "assay_conditions": {{{{...}}}},
            "data_analysis": {{{{...}}}}
        }}}}
    }}}},
    "{example_ids[1] if len(example_ids) > 1 else 'next_pair_id'}": {{{{
        ...
    }}}}
}}}}

IMPORTANT:
- Include an entry for EVERY Pair ID listed above
- Use the exact Pair ID as the key
- For original_paragraph: use descriptive keys that identify where the text came from (e.g., "Materials and Methods", "SPR Analysis", "Figure 2 caption", "Supporting Information S1"). The keys are flexible - use whatever accurately describes the source location.
- If information for a specific pair cannot be found, set confidence to "N/A" and structured_description to null

Please respond ONLY with valid JSON, no other text."""
