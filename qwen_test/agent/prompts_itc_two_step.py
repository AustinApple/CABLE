"""
Prompt templates for the two-step ITC (Isothermal Titration Calorimetry) extraction pipeline.

Step 1: Extract original_paragraph from paper (uses vision model + images)
Step 2: Fill structured_description from extracted text (text-only)
"""

from typing import Optional, Dict


def get_itc_paragraph_extraction_prompt(assay_description: str) -> str:
    """
    Step 1 prompt: Extract original_paragraph only (no structured_description).

    This prompt asks the model to find and extract the relevant experimental
    paragraphs describing ITC experiments from the paper.

    Args:
        assay_description: Brief assay description from BindingDB

    Returns:
        Formatted prompt string for vision model
    """
    return f"""You are an expert scientific reader analyzing a research paper about Isothermal Titration Calorimetry (ITC) experiments.

Task: Find and extract the COMPLETE ORIGINAL text that describes the following ITC assay experiment.

Assay Description: {assay_description}

Instructions:
1. Search through the paper to find where this ITC assay is described (focus on Experimental Methods or Methods section first, then Results, then figure captions).
2. Extract the COMPLETE ORIGINAL text that contains the full experimental protocol. Include:
   - Methods/Experimental section paragraphs describing the ITC experiment
   - Instrument details (manufacturer, model)
   - Sample preparation details (protein/ligand concentrations, cell/syringe loading)
   - Buffer composition and pH
   - Temperature and stirring speed
   - Injection parameters (volumes, number, spacing, duration)
   - Control/blank experiments
   - Data visible in figures: extract numerical values shown in thermograms or binding isotherms
   - Relevant figure captions (e.g., ITC thermogram descriptions, binding curve details)
   - Relevant table contents if applicable
   - Data analysis details (binding model, software used)
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

Note: For original_paragraph, use descriptive keys that identify where the text came from (e.g., "Materials and Methods", "ITC Measurements", "Figure 3 caption", "Supporting Information S1"). The keys are flexible - use whatever accurately describes the source location.

If you cannot find a matching ITC assay description, return:
{{
    "original_paragraph": {{}},
    "confidence": "N/A",
    "reference_number_in_text": "none",
    "references_previous": "none"
}}

Please respond ONLY with valid JSON, no other text."""


def get_itc_structured_description_from_text_prompt(
    extracted_paragraph: str,
    assay_description: str,
    protein: Optional[str] = None,
    ligand_smiles: Optional[str] = None,
    affinity_data: Optional[Dict] = None
) -> str:
    """
    Step 2 prompt: Fill structured_description from extracted text only.

    This prompt takes pre-extracted paragraph text (no images needed) and
    asks the model to fill in the structured ITC fields.

    Args:
        extracted_paragraph: The text extracted in Step 1 (combined from all sources)
        assay_description: Brief assay description from BindingDB
        protein: Deprecated - not used (kept for backward compatibility)
        ligand_smiles: Deprecated - not used (kept for backward compatibility)
        affinity_data: Deprecated - not used (kept for backward compatibility)

    Returns:
        Formatted prompt string for text-only model input
    """
    return f"""You are an expert scientific reader analyzing extracted text from a research paper about Isothermal Titration Calorimetry (ITC) experiments.

Task: Extract structured assay parameters from the following text.

Assay Description: {assay_description}

=== EXTRACTED TEXT FROM PAPER ===
{extracted_paragraph}
=== END OF EXTRACTED TEXT ===

Instructions:
1. Analyze the extracted text carefully to identify ITC assay parameters.
2. Extract structured assay parameters into the fields described below.
3. If a parameter is not mentioned in the text, use null.

STRUCTURED PARAMETERS TO EXTRACT:

instrument:
  - model: Full instrument name including manufacturer (e.g., "MicroCal VP-ITC", "MicroCal iTC200", "MicroCal PEAQ-ITC", "MicroCal Auto-iTC200", "MSC system (MicroCal Inc., MA)")

protein:
  - name: Name of the protein/macromolecule (e.g., "HSP90", "LpxC", "TTR")
  - location: Where the protein is loaded - "cell" or "syringe"
  - concentration: Protein concentration as a string in micromolar (uM), numeric value ONLY (no units in the value). Convert from other molar units: mM × 1000, nM / 1000. Use a string to allow ranges (e.g., "15", "75-225"). Return null if the paper reports mass units (e.g., mg/mL) without a molecular weight for conversion. Use ASCII 'u', not Unicode micro (µ/μ).
  - solution_volume: Volume of protein solution as a string in microliters (uL), numeric value ONLY. Convert from other units: mL × 1000. Examples: "3000", "270". Use ASCII 'u', not Unicode micro (µ/μ).

ligand:
  - name: Name of the small molecule/ligand (e.g., "compound 1", "radicicol", "tolcapone"), or null if not named
  - location: Where the ligand is loaded - "cell" or "syringe"
  - concentration: Ligand concentration as a string in micromolar (uM), numeric value ONLY (no units in the value). Convert from other molar units: mM × 1000, nM / 1000. Use a string to allow ranges (e.g., "30", "10-30"). Return null if the paper reports mass units (e.g., mg/mL) without a molecular weight for conversion. Use ASCII 'u', not Unicode micro (µ/μ).

assay_conditions:
  - buffer_composition: Full buffer composition excluding pH (e.g., "20 mM Tris-HCl, 1 mM EDTA"). Put concentration before chemical name.
  - pH: pH value as a number (e.g., 7.5, 7.4)
  - temperature_c: Temperature in Celsius as a string, numeric value ONLY (e.g., "25", "37")
  - stirring_speed: Stirring speed in rpm as a string, numeric value ONLY (e.g., "750", "1000"), or null if not stated
  - reference_cell_content: Content of the reference cell (e.g., "water", "buffer"), or null if not stated
  - control_method: Description of control/blank experiments (e.g., "Titration of ligand into buffer", "Heats of dilution subtracted"), or null if not stated
  - injection_parameters: Object with injection details (all values as strings with numeric value ONLY, no units):
    - first_volume: Volume of the first injection in microliters (uL). Convert from other units: mL × 1000. Examples: "0.4", "2". Use ASCII 'u', not Unicode micro (µ/μ).
    - subsequent_volume: Volume of subsequent injections in microliters (uL). Convert from other units: mL × 1000. Examples: "2", "10". Use ASCII 'u', not Unicode micro (µ/μ).
    - duration: Duration of each injection in seconds. Convert from other units: min × 60. Examples: "4", "10". Use null if not stated.
    - spacing: Time between injections in seconds. Convert from other units: min × 60. Examples: "120", "180". Use null if not stated.
    - total_count: Total number of injections as a string (e.g., "19", "25"), or null if not stated

data_analysis:
  - binding_model: The binding model used to fit the data (e.g., "one-site", "two-sites", "sequential binding", "nonlinear least-squares curve-fitting")

Output format (JSON):
{{
    "structured_description": {{
        "instrument": {{"model": "..."}},
        "protein": {{
            "name": "...",
            "location": "cell" or "syringe",
            "concentration": "<number in uM as string>" or null,
            "solution_volume": "<number in uL as string>" or null
        }},
        "ligand": {{
            "name": "..." or null,
            "location": "cell" or "syringe",
            "concentration": "<number in uM as string>" or null
        }},
        "assay_conditions": {{
            "buffer_composition": "...",
            "pH": <number>,
            "temperature_c": "<number in Celsius as string>",
            "stirring_speed": "<number in rpm as string>" or null,
            "reference_cell_content": "..." or null,
            "control_method": "..." or null,
            "injection_parameters": {{
                "first_volume": "<number in uL as string>" or null,
                "subsequent_volume": "<number in uL as string>" or null,
                "duration": "<number in seconds as string>" or null,
                "spacing": "<number in seconds as string>" or null,
                "total_count": "..." or null
            }}
        }},
        "data_analysis": {{
            "binding_model": "..."
        }}
    }}
}}

If you cannot extract any structured information from the text, return:
{{
    "structured_description": null
}}

Please respond ONLY with valid JSON, no other text."""