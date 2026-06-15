"""
Prompt templates for the two-step assay extraction pipeline.

Step 1: Extract original_paragraph from paper (uses vision model + images)
Step 2: Fill structured_description from extracted text (text-only)
"""

from typing import Optional, Dict


def get_paragraph_extraction_prompt(assay_description: str) -> str:
    """
    Step 1 prompt: Extract original_paragraph only (no structured_description).

    This prompt asks the model to find and extract the relevant experimental
    paragraphs from the paper without filling in structured fields.

    Args:
        assay_description: Brief assay description from BindingDB

    Returns:
        Formatted prompt string for vision model
    """
    return f"""You are an expert scientific reader analyzing a research paper about Surface Plasmon Resonance (SPR) experiments.

Task: Find and extract the COMPLETE ORIGINAL text that describes the following SPR assay experiment.

Assay Description: {assay_description}

Instructions:
1. Search through the paper to find where this SPR assay is described (focus on Experimental Methods or Methods section first, then Results, then figure captions).
2. Extract the COMPLETE ORIGINAL text that contains the full experimental protocol. Include:
   - Methods/Experimental section paragraphs describing the SPR experiment
   - Data visible in figures: extract numerical values shown in sensorgrams or plots (e.g., association/dissociation times from x-axis, concentration values, pH values labeled on curves)
   - Relevant figure captions (e.g., sensorgram descriptions, concentration ranges)
   - Relevant table contents if applicable
3. Do NOT summarize or paraphrase - provide the exact text from the paper.
4. If the description spans multiple sections, include all relevant text.

5. If the paper references a previous publication for methodology details (e.g., "as described previously", "following [ref]", "according to [author]"):

   CRITICAL STEPS FOR REFERENCE EXTRACTION:
   a) Note the EXACT reference number mentioned in the methods text
   b) Go to the References/Bibliography section at the END of the paper
   c) Find the reference entry that starts with EXACTLY that number
   d) VERIFY the number matches before copying
   e) Copy the FULL citation including: authors, title, journal name, year, volume, and page numbers

   IMPORTANT: If you cannot find the exact numbered reference, write "Reference [X] not found in bibliography".

Output format (JSON):
{{
    "original_paragraph": {{
        "<descriptive_key>": "<extracted_text>",
        ...
    }},
    "confidence": "high/medium/low",
    "reference_number_in_text": "Reference number(s) or 'none'",
    "references_previous": "Complete citation from References section or 'none'"
}}

Note: For original_paragraph, use descriptive keys that identify where the text came from (e.g., "Materials and Methods", "SPR Analysis", "Figure 2 caption", "Supporting Information S1"). The keys are flexible - use whatever accurately describes the source location.

If you cannot find a matching SPR assay description, return:
{{
    "original_paragraph": {{}},
    "confidence": "N/A",
    "reference_number_in_text": "none",
    "references_previous": "none"
}}

Please respond ONLY with valid JSON, no other text."""


def get_structured_description_from_text_prompt(
    extracted_paragraph: str,
    assay_description: str,
    protein: Optional[str] = None,
    ligand_smiles: Optional[str] = None,
    affinity_data: Optional[Dict] = None
) -> str:
    """
    Step 2 prompt: Fill structured_description from extracted text only.

    This prompt takes pre-extracted paragraph text (no images needed) and
    asks the model to fill in the structured fields.

    Args:
        extracted_paragraph: The text extracted in Step 1 (combined from all sources)
        assay_description: Brief assay description from BindingDB
        protein: Deprecated - not used (kept for backward compatibility)
        ligand_smiles: Deprecated - not used (kept for backward compatibility)
        affinity_data: Deprecated - not used (kept for backward compatibility)

    Returns:
        Formatted prompt string for text-only model input
    """
    return f"""You are an expert scientific reader analyzing extracted text from a research paper about Surface Plasmon Resonance (SPR) experiments.

Task: Extract structured assay parameters from the following text.

Assay Description: {assay_description}

=== EXTRACTED TEXT FROM PAPER ===
{extracted_paragraph}
=== END OF EXTRACTED TEXT ===

Instructions:
1. Analyze the extracted text carefully to identify SPR assay parameters.
2. Extract structured assay parameters into the fields described below.
3. If a parameter is not mentioned in the text, use null.

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
  - concentration_for_immobilization: Ligand concentration during immobilization as a number in micromolar (uM). Convert from other molar units: mM × 1000, nM / 1000. Return null if the paper reports mass units (e.g., ug/mL, mg/mL) without providing a molecular weight for conversion, or if not stated. Use ASCII 'u', not Unicode micro (µ/μ).

analyte:
  - description: Description of the analyte being tested (e.g., "MC-207,110 (efflux pump inhibitor)")
  - concentration_range: Concentration range tested as a string, with all concentrations normalized to micromolar (uM). Convert from other molar units: mM × 1000, nM / 1000. Use ASCII 'u', not Unicode micro (µ/μ). Examples: "12.5-200 uM", "0.0195-5 uM", "9-step, 2-fold, top 0.02 uM".

assay_conditions:
  - running_buffer_composition: Full buffer composition including additives (exclude pH; use the separate pH field)
  - pH: pH value as a number (e.g., 7.5, 6.0)
  - assay_type: One of: "Single Cycle Kinetics", "Multi Cycle Kinetics", "Single Cycle Steady-State Affinity", "Multi Cycle Steady-State Affinity", "Screening", "Unknown"
  - assay_flow_rate: Flow rate during analyte injection as a number in microliters per minute (uL/min). Convert from other units: mL/min × 1000, uL/s × 60. Use ASCII 'u', not Unicode micro (µ/μ).
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
{{
    "structured_description": {{
        "instrument": {{"manufacturer": "...", "model": "..."}},
        "sensor_chip": {{"type": "...", "manufacturer": "..."}},
        "immobilization": {{
            "ligand_name": "...",
            "strategy": "...",
            "density_ru": ...,
            "concentration_for_immobilization": <number in uM> or null
        }},
        "analyte": {{
            "description": "...",
            "concentration_range": "<range in uM, e.g., '12.5-200 uM'>"
        }},
        "assay_conditions": {{
            "running_buffer_composition": "...",
            "pH": ...,
            "assay_type": "...",
            "assay_flow_rate": <number in uL/min>,
            "temperature_c": ... or null,
            "association_time_s": ... or null,
            "dissociation_time_s": ... or null,
            "regeneration_solution": "..."
        }},
        "data_analysis": {{
            "reference_description": "...",
            "subtraction_method": "...",
            "fitting_model": "...",
            "software": "..."
        }}
    }}
}}

If you cannot extract any structured information from the text, return:
{{
    "structured_description": null
}}

Please respond ONLY with valid JSON, no other text."""
