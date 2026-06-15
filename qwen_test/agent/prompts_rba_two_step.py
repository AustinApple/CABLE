"""
Prompt templates for the two-step RBA (Radioligand Binding Assay) extraction pipeline.

Step 0: Classify assay from BindingDB description (text-only, no images)
Step 1: Extract original_paragraph from paper (uses vision model + images)
Step 2: Fill structured_description from extracted text (text-only)
"""

from typing import Optional, Dict


def get_assay_classification_prompt(assay_description: str) -> str:
    """
    Step 0 prompt: Classify whether an assay description refers to a true
    radioligand binding assay or a different assay type (enzyme activity,
    functional assay, etc.).

    This is a cheap text-only call using the BindingDB description string —
    no paper images needed. Runs before Step 1 to gate extraction.

    Args:
        assay_description: Brief assay description from BindingDB

    Returns:
        Formatted prompt string for text-only model input
    """
    return f"""You are an expert pharmacologist classifying assay types.

Task: Determine whether the following assay description refers to a TRUE radioligand binding assay (RBA) or a different type of assay.

Assay Description:
{assay_description}

=== CLASSIFICATION CRITERIA ===

A TRUE radioligand binding assay has these characteristics:
- A radiolabeled LIGAND (drug, agonist, antagonist, or specific binding probe) binds directly to a RECEPTOR or protein target (e.g., GPCR, ion channel, transporter, nuclear receptor, or other binding protein)
- The assay measures the BINDING INTERACTION itself — how much radioligand occupies the binding site on the target
- In competition/displacement assays, test compounds compete with the radioligand for the same binding site
- In saturation binding assays, increasing radioligand concentrations are used to determine Kd and Bmax
- Biological preparations are typically membrane homogenates, whole cells, or tissue sections
- Nonspecific binding (NSB) is defined using an excess of an unlabeled compound that binds the SAME target site
- Bound radioligand is separated from free radioligand by filtration (GF/B, GF/C filters), centrifugation, or Scintillation Proximity Assay (SPA)
- Reported endpoints include Ki, IC50, Kd, Bmax, pKi, pIC50, or percent inhibition of specific binding
- Common radioligands: [3H]-labeled receptor ligands, [125I]-labeled receptor ligands

The following are NOT radioligand binding assays:

1. ENZYME ACTIVITY / ENZYME INHIBITION ASSAYS:
   - The radiolabel is on a SUBSTRATE (e.g., [γ-32P]ATP, [γ-33P]ATP, [14C]-labeled substrate) that the enzyme converts into a radiolabeled PRODUCT
   - The assay measures CATALYTIC ACTIVITY — phosphorylation of a peptide/protein substrate, hydrolysis, acetylation, methylation, or other chemical transformation
   - Identifying keywords: kinase assay, phosphorylation, phosphotransfer, substrate phosphorylation, catalytic activity, enzymatic activity, enzyme inhibition, histone acetyltransferase, methyltransferase, protease assay
   - Product is separated from substrate by TCA precipitation, phosphocellulose paper (P81), SDS-PAGE/autoradiography, thin-layer chromatography, or DEAE filter binding
   - The IC50 reflects inhibition of enzyme catalytic activity, NOT displacement of a ligand from a binding site
   - Examples: [γ-33P]ATP kinase assay, [14C]acetyl-CoA acetyltransferase assay, [3H]SAM methyltransferase assay

2. GTPγS FUNCTIONAL ASSAYS:
   - [35S]GTPγS binding assays measure G-protein ACTIVATION (GDP-to-GTP exchange on Gα subunits) downstream of receptor signaling
   - This is a FUNCTIONAL readout of receptor activation, NOT a direct measurement of ligand-receptor binding affinity
   - Identifying keywords: GTPγS, [35S]GTPγS, G-protein activation, GDP/GTP exchange, agonist-stimulated GTPγS binding, basal GTPγS binding
   - Although they use filtration and membranes (similar to RBA), the measured parameter is G-protein activation, not ligand occupancy at the receptor

3. CELL PROLIFERATION / INCORPORATION ASSAYS:
   - [3H]thymidine incorporation measures DNA synthesis / cell proliferation
   - [3H]uridine incorporation measures RNA synthesis
   - [3H]leucine incorporation measures protein synthesis
   - These are NOT binding assays — they measure metabolic incorporation of radiolabeled precursors

4. UPTAKE / TRANSPORT ASSAYS:
   - Radiolabeled compound uptake into cells measures transporter FUNCTION (how much substrate is transported), not binding affinity to the transporter protein
   - Examples: [3H]dopamine uptake, [3H]serotonin uptake, [3H]GABA uptake
   - These report IC50 of uptake inhibition, which reflects transporter function, not direct binding

5. METABOLIC ENZYME ASSAYS:
   - [14C]CO2 release from [14C]-labeled substrates measures metabolic enzyme activity
   - Radiolabeled metabolite formation assays

EDGE CASE — this IS a true RBA:
- A radiolabeled enzyme INHIBITOR used to measure DIRECT BINDING to the enzyme active site (no catalysis measured, just binding occupancy) — this is a legitimate binding assay even though the target is an enzyme rather than a receptor
- Example: [3H]-labeled kinase inhibitor displacement assay measuring binding to the ATP site

=== OUTPUT FORMAT ===

Respond with ONLY valid JSON:
{{
    "is_radioligand_binding_assay": true or false,
    "assay_category": "Radioligand Binding Assay" or "Enzyme Activity Assay" or "GTPγS Functional Assay" or "Cell Proliferation Assay" or "Uptake/Transport Assay" or "Metabolic Enzyme Assay" or "Other" or "Uncertain",
    "confidence": "high" or "medium" or "low",
    "reasoning": "Brief explanation of the key evidence in the description that led to this classification"
}}

Please respond ONLY with valid JSON, no other text."""


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
  - protein_concentration: Concentration of membrane protein or cells per well/tube with units. Heterogeneous field — may be mass-based (ug/well, ug/tube, ug/mL) or cell-count-based (cells/well). Use ASCII 'u', not Unicode micro (µ/μ). Examples: "50 ug/well", "100,000 cells/well".

radioligand:
  - name: Name of the radioligand including isotope label (e.g., "[3H]-SCH 23390", "[125I]-RTI-55")
  - isotope: Radioisotope. One of: "3H", "125I", "35S", "14C", "33P", "Other"
  - specific_activity: Object with:
      - value: Numeric value or range as string (e.g., 85.5, "81-86")
      - unit: One of: "Ci/mmol", "mCi/mmol", "GBq/mmol", "TBq/mmol"
  - concentration_used: Radioligand concentration in the assay as a number in nanomolar (nM). Convert from other molar units: pM / 1000, uM × 1000, mM × 1000000. Return null if the paper reports only a qualitative value (e.g., "~Kd concentration") or uses units that cannot be converted to nM. Use ASCII 'u', not Unicode micro (µ/μ).
  - kd_value: Object with:
      - value: Kd value as a number in nanomolar (nM). Convert from other units: uM × 1000, pM / 1000. Use ASCII 'u', not Unicode micro (µ/μ).
      - source: One of: "Determined in this study", "Literature value", "Not specified"

test_compound:
  - description: General description of the test compound(s) (e.g., "Small molecules", "Novel D2 antagonists")
  - concentration_range: Range of concentrations tested as a string, with all concentrations normalized to micromolar (uM). Convert from other molar units: nM / 1000, pM / 1000000, mM × 1000. Use ASCII 'u', not Unicode micro (µ/μ). Examples: "0.0001-10 uM", "10-point, half-log dilution, top 10 uM".
  - vehicle_solvent: Solvent used and its final assay concentration (e.g., "DMSO, final 1%")

nsb_definition:
  - compound_name: Unlabeled compound used to define nonspecific binding (e.g., "Haloperidol", "Naloxone")
  - compound_concentration: NSB-defining compound concentration as a number in micromolar (uM). Convert from other molar units: nM / 1000, mM × 1000. Return null if units cannot be converted to uM. Use ASCII 'u', not Unicode micro (µ/μ).
  - selectivity_rationale: One of: "Target-selective", "Non-selective", "Not specified"

assay_conditions:
  - assay_format: Physical format. (e.g., "Filtration (wells)", "Filtration (tubes)", "Scintillation Proximity Assay (SPA)", "Centrifugation", "Microplate-based (FlashPlate)", "Microplate-based (Cytostar-T)", "Other")
  - buffer_composition: Full incubation buffer composition including salts and additives, excluding pH (e.g., "50 mM Tris-HCl, 120 mM NaCl, 5 mM MgCl2, 1 mM EDTA")
  - pH: pH of incubation buffer as a number (e.g., 7.4)
  - incubation_temperature: Temperature as string to allow "room temperature" (e.g., "25", "37", "room temperature")
  - incubation_time: Duration of incubation as a number in minutes. Convert from other units: h × 60, s / 60. Use a string to capture ranges (e.g., "60-120"). Examples: 60, 90, "60-120".
  - equilibrium_confirmed: One of: "Yes", "No", "Not specified"
  - total_assay_volume: Total assay volume as a number in microliters (uL). Convert from other units: mL × 1000. Use ASCII 'u', not Unicode micro (µ/μ). Examples: 200, 500, 1000.

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
            "concentration_used": <number in nM> or null,
            "kd_value": {{"value": <number in nM>, "source": "..."}}
        }},
        "test_compound": {{
            "description": "...",
            "concentration_range": "<range in uM, e.g., '0.0001-10 uM'>",
            "vehicle_solvent": "..."
        }},
        "nsb_definition": {{
            "compound_name": "...",
            "compound_concentration": <number in uM> or null,
            "selectivity_rationale": "..."
        }},
        "assay_conditions": {{
            "assay_format": "...",
            "buffer_composition": "...",
            "pH": ...,
            "incubation_temperature": "...",
            "incubation_time": <number in minutes, or string for ranges like "60-120">,
            "equilibrium_confirmed": "...",
            "total_assay_volume": <number in uL>
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
