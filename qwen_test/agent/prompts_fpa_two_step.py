"""
Prompt templates for the two-step FPA (Fluorescence Polarization/Anisotropy) extraction pipeline.

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
    return f"""You are an expert scientific reader analyzing a research paper about Fluorescence Polarization (FP) or Fluorescence Anisotropy (FA) binding assay experiments.

Task: Find and extract the COMPLETE ORIGINAL text that describes the following FP/FA experiment.

Assay Description: {assay_description}

Instructions:
1. Search through the paper to find where this FP/FA experiment is described (focus on Experimental Methods or Methods section first, then Results, then figure captions).
2. Extract the COMPLETE ORIGINAL text that contains the full experimental protocol. Include:
   - Methods/Experimental section paragraphs describing the fluorescence polarization/anisotropy binding experiment
   - Details about the target protein (purification, expression system, construct boundaries)
   - Fluorescent tracer identity, fluorophore, concentration, and Kd
   - Assay controls (high signal / low signal definitions, positive control compound)
   - Assay conditions (plate format, buffer, detergent, temperature, incubation time, volume, order of addition)
   - Detection details (instrument, readout mode FP vs FA, excitation/emission wavelengths, G-factor correction)
   - Artifact controls (compound fluorescence interference, aggregation checks, inner filter effect)
   - Data analysis details (software, fitting model, Ki/IC50 calculation, normalization)
   - Relevant table contents or figure captions if applicable
3. Do NOT summarize or paraphrase - provide the exact text from the paper.
4. If the description spans multiple sections, include all relevant text.

5. If the paper uses language like "as described previously [X]", "following the protocol in [X]", "according to [author] [X]", or "as reported in [X]" specifically within the FP/FA methodology description:

   CRITICAL STEPS FOR REFERENCE EXTRACTION:
   a) Copy the EXACT sentence(s) from the methods text that contain this language
   b) Identify the EXACT reference number(s) cited in those sentences
   c) Go to the References/Bibliography section at the END of the paper
   d) For EACH reference number, find the entry that starts with EXACTLY that number
   e) VERIFY each number matches before copying
   f) Copy the FULL citation for each, including: authors, title, journal name, year, volume, and page numbers

   IMPORTANT:
   - Only capture references cited with explicit "previously described / protocol from" language in the methods section
   - Do NOT include references cited for background, compound origins, or result comparisons
   - If you cannot find an exact numbered reference, write "Reference [X] not found in bibliography"

Output format (JSON):
{{
    "original_paragraph": {{
        "<descriptive_key>": "<extracted_text>",
        ...
    }},
    "confidence": "high/medium/low",
    "reference_number_in_text": "Reference number(s) cited with 'previously described' language (e.g. '26, 27') or 'none'",
    "reference_sentence_in_text": "The exact sentence(s) from the methods section that cite the reference(s), or 'none'",
    "references_previous": "Complete citation(s) from References section or 'none'"
}}

Note: For original_paragraph, use descriptive keys that identify where the text came from (e.g., "Materials and Methods", "Fluorescence Polarization Assay", "Data Analysis", "Table 1"). The keys are flexible - use whatever accurately describes the source location.

If you cannot find a matching FP/FA assay description, return:
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
    asks the model to fill in the structured fields per the FPA schema.

    Args:
        extracted_paragraph: The text extracted in Step 1 (combined from all sources)
        assay_description: Brief assay description from BindingDB
        protein: Deprecated - not used (kept for backward compatibility)
        ligand_smiles: Deprecated - not used (kept for backward compatibility)
        affinity_data: Deprecated - not used (kept for backward compatibility)

    Returns:
        Formatted prompt string for text-only model input
    """
    return f"""You are an expert scientific reader analyzing extracted text from a research paper about Fluorescence Polarization (FP) or Fluorescence Anisotropy (FA) binding assay experiments.

Task: Extract structured assay parameters from the following text.

Assay Description: {assay_description}

=== EXTRACTED TEXT FROM PAPER ===
{extracted_paragraph}
=== END OF EXTRACTED TEXT ===

Instructions:
1. Analyze the extracted text carefully to identify FP/FA assay parameters.
2. Extract structured assay parameters into the fields described below.
3. If a parameter is not mentioned in the text, use null.
4. The Assay Description specifies the exact target protein for this data point. For protein-specific fields (target_protein, protein_concentration), report only the values relevant to that specific protein, not all proteins mentioned in the text.

STRUCTURED PARAMETERS TO EXTRACT:

biological_preparation:
  - source_type: Type of biological material. One of: "Purified protein", "Membrane homogenate", "Whole cells", "Cell lysate", "Tissue homogenate", "Other"
  - tissue_or_cell_line: Expression system or cell line used to produce the protein (e.g., "E. coli BL21(DE3) expressing His-tagged hBRD4-BD1", "Sf9 insect cells expressing hHsp90α")
  - species: Species origin of the target protein (e.g., "Human", "Rat", "Mouse", "E. coli (recombinant human)")
  - target_protein: Protein target including domain or construct boundaries if specified (e.g., "BRD4 bromodomain 1 (BD1, residues 44–168)", "MDM2 (residues 1–118)")
  - protein_concentration: Concentration of protein used in the assay with units (e.g., "50 nM", "100 nM")
  - protein_purity_method: Purification method if reported (e.g., "Ni-NTA affinity chromatography followed by SEC"), or null

fluorescent_tracer:
  - name: Name or identifier of the fluorescent tracer including fluorophore label (e.g., "FITC-labeled p53 peptide (residues 15–29)", "TAMRA-JQ1", "FAM-SMAC peptide (AVPIAQK)")
  - tracer_type: Chemical nature of the tracer. One of: "Labeled peptide", "Labeled small molecule", "Labeled nucleic acid", "Labeled protein fragment", "Other"
  - fluorophore: Fluorophore conjugated to the tracer (e.g., "Fluorescein (FITC)", "FAM (6-carboxyfluorescein)", "TAMRA", "BODIPY", "Alexa Fluor 488")
  - excitation_wavelength_nm: Excitation wavelength in nanometers (e.g., 485), or null
  - emission_wavelength_nm: Emission wavelength in nanometers (e.g., 530), or null
  - concentration_used: Concentration of tracer used in the assay with units (e.g., "5 nM", "10 nM")
  - kd_value: Object with:
      - value: Numeric Kd value
      - unit: One of: "nM", "µM", "pM"
      - source: One of: "Determined in this study", "Literature value", "Not specified"

test_compound:
  - description: General description of the test compound(s) (e.g., "Small molecules", "Novel BRD4 inhibitors", "Stapled peptides targeting MDM2")
  - concentration_range: Range of concentrations tested (e.g., "0.1 nM – 100 µM", "10-point, half-log dilution, top 50 µM")
  - vehicle_solvent: Solvent used and its final assay concentration (e.g., "DMSO, final 1%")
  - dmso_tolerance_tested: One of: "Yes", "No", "Not specified"

assay_controls:
  - high_signal_control: Condition representing maximum polarization/anisotropy (fully bound tracer) (e.g., "Protein + tracer, no competitor")
  - low_signal_control: Condition representing minimum polarization/anisotropy (fully displaced or free tracer) (e.g., "Tracer only, no protein", "Tracer + 100 µM unlabeled JQ1")
  - positive_control_compound: Reference compound of known affinity used to validate the assay (e.g., "(+)-JQ1", "Nutlin-3a"), or null
  - positive_control_expected_value: Expected Ki or IC50 for the positive control with units (e.g., "Ki = 50 nM"), or null
  - z_prime_reported: Object with:
      - reported: One of: "Yes", "No"
      - value: Z'-factor value if reported, null otherwise

assay_conditions:
  - assay_format: Plate format. One of: "384-well black plate", "96-well black plate", "384-well low-volume black plate", "1536-well black plate", "Other"
  - plate_type: Specific plate information if reported (e.g., "Corning 3575 (384-well, low-volume, black, flat-bottom)"), or null
  - buffer_composition: Full assay buffer composition excluding pH (e.g., "50 mM HEPES, 150 mM NaCl, 0.01% Triton X-100, 1 mM DTT")
  - ph: pH of assay buffer as a number (e.g., 7.4)
  - detergent_used: Detergent included in buffer (e.g., "0.01% Triton X-100", "0.05% CHAPS"), or null
  - incubation_temperature: Temperature as string (e.g., "25", "37", "room temperature")
  - incubation_time: Object with:
      - value: Duration as number or string (e.g., 30, 60, "30-60", "overnight")
      - unit: One of: "min", "h"
  - equilibrium_confirmed: One of: "Yes", "No", "Not specified"
  - total_assay_volume: Total volume per well with units (e.g., "20 µL", "50 µL")
  - order_of_addition: Order of component addition (e.g., "Protein + compound pre-incubated 15 min, then tracer added", "All components added simultaneously"), or null

detection:
  - readout_mode: One of: "Fluorescence Polarization (FP)", "Fluorescence Anisotropy (FA)", "Not specified"
  - readout_unit: One of: "mP (millipolarization)", "Anisotropy (r)", "Not specified"
  - instrument: Plate reader used (e.g., "BMG PHERAstar FSX", "Tecan Infinite M1000 Pro", "PerkinElmer EnVision")
  - g_factor_correction: One of: "Yes", "No", "Not specified"
  - assay_window: Object with:
      - bound_signal: FP/FA value for fully bound tracer, or null (e.g., "220 mP", "0.18 r")
      - free_signal: FP/FA value for free tracer, or null (e.g., "80 mP", "0.05 r")
      - delta: Difference between bound and free signal, or null (e.g., "140 mP", "0.13 r")

data_analysis:
  - endpoint_reported: One of: "Ki", "IC50", "Kd", "Percent inhibition", "pKi", "pIC50", "EC50", "Other"
  - ic50_to_ki_conversion: One of: "Cheng-Prusoff equation", "Nikolovska-Coleska equation", "Exact solution", "Not applicable", "Not specified", or null. Use "Nikolovska-Coleska equation" when Ki values are obtained by nonlinear regression fitting of competition curves using the Kd of the probe and concentrations of protein and probe (the exact competitive binding equation for FP assays). Use "Not applicable" only when Ki values are reported directly from binding experiments without IC50 conversion.
  - fitting_model: One of: "One-site competition", "Two-site competition", "Four-parameter logistic (4PL)", "Three-parameter logistic (3PL, Hill slope fixed to 1)", "Dose-response (variable slope)", "Other"
  - hill_slope_reported: Object with:
      - reported: One of: "Yes, fixed to 1", "Yes, variable", "Not reported"
      - value: Hill slope value if variable, null otherwise
  - data_normalization: One of: "Percent inhibition relative to controls", "Fraction bound", "Raw mP values fitted directly", "Raw anisotropy values fitted directly", "Not specified", or null
  - software: Software used for curve fitting (e.g., "GraphPad Prism 9", "GeneData Screener")
  - replicates: Object with:
      - technical_replicates: One of: "Duplicate", "Triplicate", "Singlicate", "Not specified"
      - independent_experiments: Number of independent experiments as number or string (e.g., 3, "2-4", "≥3", "Not specified")

artifact_controls:
  - compound_fluorescence_interference: One of: "Tested and excluded interfering compounds", "Tested, no interference observed", "Counter-screen performed", "Not specified"
  - compound_aggregation_check: One of: "Yes, detergent sensitivity tested", "Yes, DLS performed", "Yes, Hill slope monitored", "No", "Not specified"
  - inner_filter_effect: One of: "Tested", "Not tested", "Not specified"
  - orthogonal_assay_confirmation: Orthogonal assay used to confirm hits (e.g., "ITC", "SPR", "TR-FRET", "AlphaScreen", "NMR", "None reported"), or null

Output format (JSON):
{{
    "structured_description": {{
        "biological_preparation": {{
            "source_type": "...",
            "tissue_or_cell_line": "...",
            "species": "...",
            "target_protein": "...",
            "protein_concentration": "...",
            "protein_purity_method": "..." or null
        }},
        "fluorescent_tracer": {{
            "name": "...",
            "tracer_type": "...",
            "fluorophore": "...",
            "excitation_wavelength_nm": ... or null,
            "emission_wavelength_nm": ... or null,
            "concentration_used": "...",
            "kd_value": {{"value": ..., "unit": "...", "source": "..."}}
        }},
        "test_compound": {{
            "description": "...",
            "concentration_range": "...",
            "vehicle_solvent": "...",
            "dmso_tolerance_tested": "..."
        }},
        "assay_controls": {{
            "high_signal_control": "...",
            "low_signal_control": "...",
            "positive_control_compound": "..." or null,
            "positive_control_expected_value": "..." or null,
            "z_prime_reported": {{"reported": "...", "value": ... or null}}
        }},
        "assay_conditions": {{
            "assay_format": "...",
            "plate_type": "..." or null,
            "buffer_composition": "...",
            "ph": ...,
            "detergent_used": "..." or null,
            "incubation_temperature": "...",
            "incubation_time": {{"value": ..., "unit": "..."}},
            "equilibrium_confirmed": "...",
            "total_assay_volume": "...",
            "order_of_addition": "..." or null
        }},
        "detection": {{
            "readout_mode": "...",
            "readout_unit": "...",
            "instrument": "...",
            "g_factor_correction": "...",
            "assay_window": {{
                "bound_signal": "..." or null,
                "free_signal": "..." or null,
                "delta": "..." or null
            }}
        }},
        "data_analysis": {{
            "endpoint_reported": "...",
            "ic50_to_ki_conversion": "..." or null,
            "fitting_model": "...",
            "hill_slope_reported": {{"reported": "...", "value": ... or null}},
            "data_normalization": "..." or null,
            "software": "...",
            "replicates": {{"technical_replicates": "...", "independent_experiments": ...}}
        }},
        "artifact_controls": {{
            "compound_fluorescence_interference": "...",
            "compound_aggregation_check": "...",
            "inner_filter_effect": "...",
            "orthogonal_assay_confirmation": "..." or null
        }}
    }}
}}

If you cannot extract any structured information from the text, return:
{{
    "structured_description": null
}}

Please respond ONLY with valid JSON, no other text."""
