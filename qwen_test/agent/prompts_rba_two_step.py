"""
Prompt templates for the two-step RBA (Radioligand Binding Assay) extraction pipeline.

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
    return f"""You are an expert scientific reader analyzing a research paper about Radioligand Binding Assay (RBA) experiments.

Task: Find and extract the COMPLETE ORIGINAL text that describes the following RBA experiment.

Assay Description: {assay_description}

Instructions:
1. Search through the paper to find where this RBA experiment is described (focus on Experimental Methods or Methods section first, then Results, then figure captions).
2. Extract the COMPLETE ORIGINAL text that contains the full experimental protocol. Include:
   - Methods/Experimental section paragraphs describing the radioligand binding experiment
   - Details about the biological preparation (membranes, cells, tissue)
   - Radioligand identity, concentration, and specific activity
   - Nonspecific binding (NSB) definition compound and concentration
   - Incubation conditions (buffer, temperature, time, volume)
   - Separation and detection methods (filtration, washing, scintillation counting)
   - Data analysis details (software, fitting model, Ki/IC50 calculation)
   - Relevant table contents or figure captions if applicable
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

Note: For original_paragraph, use descriptive keys that identify where the text came from (e.g., "Materials and Methods", "Radioligand Binding Assay", "Data Analysis", "Table 1"). The keys are flexible - use whatever accurately describes the source location.

If you cannot find a matching RBA description, return:
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
    asks the model to fill in the structured fields per the RBA schema.

    Args:
        extracted_paragraph: The text extracted in Step 1 (combined from all sources)
        assay_description: Brief assay description from BindingDB
        protein: Deprecated - not used (kept for backward compatibility)
        ligand_smiles: Deprecated - not used (kept for backward compatibility)
        affinity_data: Deprecated - not used (kept for backward compatibility)

    Returns:
        Formatted prompt string for text-only model input
    """
    return f"""You are an expert scientific reader analyzing extracted text from a research paper about Radioligand Binding Assay (RBA) experiments.

Task: Extract structured assay parameters from the following text.

Assay Description: {assay_description}

=== EXTRACTED TEXT FROM PAPER ===
{extracted_paragraph}
=== END OF EXTRACTED TEXT ===

Instructions:
1. Analyze the extracted text carefully to identify RBA assay parameters.
2. Extract structured assay parameters into the fields described below.
3. If a parameter is not mentioned in the text, use null.

STRUCTURED PARAMETERS TO EXTRACT:

biological_preparation:
  - source_type: Type of biological material. One of: "Membrane homogenate", "Whole cells", "Tissue section", "Purified protein", "Cell lysate", "Tissue homogenate", "Other"
  - tissue_or_cell_line: Tissue, cell line, or expression system (e.g., "Rat striatal membranes", "CHO-K1 cells stably expressing hD2R")
  - species: Species origin (e.g., "Human", "Rat", "Mouse", "Guinea pig")
  - target_protein: Receptor or protein target, including isoform/subunit if specified (e.g., "Dopamine D2 receptor", "GABAA (α1β2γ2)")
  - protein_concentration: Concentration of membrane protein or cells per well/tube with units (e.g., "50 µg/well", "100,000 cells/well")

radioligand:
  - name: Name of the radioligand including isotope label (e.g., "[3H]-SCH 23390", "[125I]-RTI-55")
  - isotope: Radioisotope. One of: "3H", "125I", "35S", "14C", "33P", "Other"
  - specific_activity: Object with:
      - value: Numeric value or range as string (e.g., 85.5, "81-86")
      - unit: One of: "Ci/mmol", "mCi/mmol", "GBq/mmol", "TBq/mmol"
  - concentration_used: Concentration used in assay with units (e.g., "1 nM", "0.5 nM")
  - kd_value: Object with:
      - value: Numeric Kd value
      - unit: One of: "nM", "µM", "pM"
      - source: One of: "Determined in this study", "Literature value", "Not specified"

test_compound:
  - description: General description of the test compound(s) (e.g., "Small molecules", "Novel D2 antagonists")
  - concentration_range: Range of concentrations tested (e.g., "0.1 nM – 10 µM", "10-point, half-log dilution")
  - vehicle_solvent: Solvent used and its final assay concentration (e.g., "DMSO, final 1%")

nsb_definition:
  - compound_name: Unlabeled compound used to define nonspecific binding (e.g., "Haloperidol", "Naloxone")
  - compound_concentration: Concentration of the NSB-defining compound with units (e.g., "10 µM")
  - selectivity_rationale: One of: "Target-selective", "Non-selective", "Not specified"

assay_conditions:
  - assay_format: Physical format. (e.g., "Filtration (wells)", "Filtration (tubes)", "Scintillation Proximity Assay (SPA)", "Centrifugation", "Microplate-based (FlashPlate)", "Microplate-based (Cytostar-T)", "Other")
  - buffer_composition: Full incubation buffer composition including salts and additives, excluding pH (e.g., "50 mM Tris-HCl, 120 mM NaCl, 5 mM MgCl2, 1 mM EDTA")
  - pH: pH of incubation buffer as a number (e.g., 7.4)
  - incubation_temperature: Temperature as string to allow "room temperature" (e.g., "25", "37", "room temperature")
  - incubation_time: Object with:
      - value: Duration as number or string (e.g., 60, "60-120")
      - unit: One of: "min", "h"
  - equilibrium_confirmed: One of: "Yes", "No", "Not specified"
  - total_assay_volume: Total assay volume with units (e.g., "200 µL", "1 mL")

separation_and_detection:
  - separation_method: One of: "Rapid vacuum filtration", "Harvester filtration", "Centrifugation", "SPA (no separation)", "FlashPlate (no separation)", "Other"
  - filter_type: Filter type used if filtration-based, null otherwise (e.g., "GF/B", "Whatman GF/B pretreated with 0.3% PEI")
  - wash_protocol: Wash conditions after filtration, null if not applicable (e.g., "3 × 3 mL ice-cold 50 mM Tris-HCl pH 7.4")
  - detection_instrument: Instrument used to measure radioactivity (e.g., "Perkin Elmer MicroBeta2", "Packard TopCount NXT")
  - scintillation_cocktail: Scintillation cocktail for liquid scintillation counting, null if gamma counting or proximity-based (e.g., "Ultima Gold", "MicroScint-20")

data_analysis:
  - endpoint_reported: One of: "Ki", "IC50", "Kd", "Bmax", "Percent inhibition", "pKi", "pIC50", "EC50", "Other"
  - ic50_to_ki_conversion: One of: "Cheng-Prusoff equation", "Munson-Rodbard correction", "Exact solution (Cer & Bhatt)", "Not applicable", "Not specified", or null
  - fitting_model: One of: "One-site competition", "Two-site competition", "Four-parameter logistic", "Saturation binding (one-site)", "Saturation binding (two-site)", "Other"
  - hill_slope_reported: Object with:
      - reported: One of: "Yes, fixed to 1", "Yes, variable", "Not reported"
      - value: Hill slope value if variable, null otherwise
  - software: Software used for curve fitting (e.g., "GraphPad Prism 9", "SigmaPlot")
  - replicates: Object with:
      - technical_replicates: Number of replicates per concentration (e.g., "Duplicate", "Triplicate", "Not specified")
      - independent_experiments: Number of independent experiments as number or string (e.g., 3, "2-4", "Not specified")

Output format (JSON):
{{
    "structured_description": {{
        "biological_preparation": {{
            "source_type": "...",
            "tissue_or_cell_line": "...",
            "species": "...",
            "target_protein": "...",
            "protein_concentration": "..."
        }},
        "radioligand": {{
            "name": "...",
            "isotope": "...",
            "specific_activity": {{"value": ..., "unit": "..."}},
            "concentration_used": "...",
            "kd_value": {{"value": ..., "unit": "...", "source": "..."}}
        }},
        "test_compound": {{
            "description": "...",
            "concentration_range": "...",
            "vehicle_solvent": "..."
        }},
        "nsb_definition": {{
            "compound_name": "...",
            "compound_concentration": "...",
            "selectivity_rationale": "..."
        }},
        "assay_conditions": {{
            "assay_format": "...",
            "buffer_composition": "...",
            "pH": ...,
            "incubation_temperature": "...",
            "incubation_time": {{"value": ..., "unit": "..."}},
            "equilibrium_confirmed": "...",
            "total_assay_volume": "..."
        }},
        "separation_and_detection": {{
            "separation_method": "...",
            "filter_type": "..." or null,
            "wash_protocol": "..." or null,
            "detection_instrument": "...",
            "scintillation_cocktail": "..." or null
        }},
        "data_analysis": {{
            "endpoint_reported": "...",
            "ic50_to_ki_conversion": "..." or null,
            "fitting_model": "...",
            "hill_slope_reported": {{"reported": "...", "value": ... or null}},
            "software": "...",
            "replicates": {{"technical_replicates": "...", "independent_experiments": ...}}
        }}
    }}
}}

If you cannot extract any structured information from the text, return:
{{
    "structured_description": null
}}

Please respond ONLY with valid JSON, no other text."""
