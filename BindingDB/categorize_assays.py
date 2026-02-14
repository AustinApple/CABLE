"""
Categorize descriptions from BindingDB into assay types for measuring
protein-ligand binding affinity.
"""

import pandas as pd
import re
from collections import OrderedDict, Counter

# Define assay categories with keyword patterns (case-insensitive)
# Order matters: more specific patterns should come before generic ones
ASSAY_CATEGORIES = OrderedDict([
    # --- Biophysical / Label-free methods ---
    ("Surface Plasmon Resonance (SPR)", [
        r'\bSPR\b', r'surface plasmon resonance', r'\bBiacore\b',
    ]),
    ("Isothermal Titration Calorimetry (ITC)", [
        r'\bITC\b', r'isothermal titration calorimetry',
    ]),
    ("Microscale Thermophoresis (MST)", [
        r'\bMST\b', r'microscale thermophoresis',
    ]),
    ("Bio-Layer Interferometry (BLI)", [
        r'\bBLI\b', r'bio-?layer interferometry', r'\bOctet\b',
    ]),
    ("Thermal Shift Assay (TSA/DSF)", [
        r'thermal shift', r'differential scanning fluorimetry', r'\bDSF\b',
        r'\bTSA\b', r'\bThermoFluor\b', r'thermal denaturation',
        r'melting temperature',
    ]),
    ("Dual Polarization Interferometry (DPI)", [
        r'\bDPI\b', r'dual polarization interferometry',
    ]),
    ("NMR", [
        r'\bNMR\b', r'nuclear magnetic resonance', r'\bSTD[- ]NMR\b',
        r'waterLOGSY', r'\bHSQC\b', r'\bNOESY\b',
    ]),

    # --- Fluorescence-based methods ---
    ("TR-FRET / HTRF / LanthaScreen", [
        r'\bTR-FRET\b', r'\bHTRF\b', r'\bLanthaScreen\b', r'\bLANCE\b',
        r'time-resolved fluorescence', r'time resolved fluorescence',
        r'homogeneous time-resolved',
    ]),
    ("FRET", [
        r'\bFRET\b', r'fluorescence resonance energy transfer',
        r'F(?:o|ö)rster resonance',
    ]),
    ("Fluorescence Polarization / Anisotropy (FP)", [
        r'fluorescence polarization', r'fluorescence anisotropy',
        r'\bFP assay\b', r'\bFP\b(?=.*assay)',
    ]),
    ("AlphaScreen / AlphaLISA", [
        r'\bAlphaScreen\b', r'\bAlphaLISA\b', r'\bAlpha[- ]?Screen\b',
    ]),
    ("NanoBRET", [
        r'\bNanoBRET\b', r'\bBRET\b',
    ]),
    ("Fluorometric / Fluorescence Assay", [
        r'fluorometric', r'fluorimetric', r'fluorescence',
        r'fluorescent', r'fluorogenic', r'fluori(?:metric|metry)',
    ]),

    # --- Radioligand / Radiometric methods ---
    ("Scintillation Proximity Assay (SPA)", [
        r'scintillation proximity', r'\bSPA\b(?=.*assay)',
    ]),
    ("Radioligand Binding Assay", [
        r'\[3H\]', r'\[125I\]', r'\[35S\]', r'\[14C\]',
        r'\[1-14C\]', r'\[gamma-33P\]', r'33P-ATP', r'\[32P\]',
        r'radioligand', r'radiolabel', r'radio-?label',
        r'scintillation count', r'autoradiograph',
        r'tritiated', r'gamma counting',
    ]),

    # --- Immunoassays ---
    ("ELISA", [
        r'\bELISA\b', r'enzyme.linked immunosorbent',
    ]),
    ("Immunoprecipitation", [
        r'immunoprecipitation', r'\bIP\b(?=.*assay)',
        r'co-immunoprecipitation',
    ]),

    # --- Chromatography / Separation-based ---
    ("HPLC-based Assay", [
        r'\bHPLC\b', r'high.performance liquid chromatography',
        r'high.pressure liquid chromatography',
    ]),
    ("LC-MS / Mass Spectrometry", [
        r'\bLC-MS\b', r'\bLC/MS\b', r'mass spectrom',
        r'\bMS/MS\b', r'\bMALDI\b', r'\bESI-MS\b',
    ]),

    # --- Electrophysiology ---
    ("Electrophysiology / Patch Clamp", [
        r'patch.clamp', r'electrophysiolog', r'voltage.clamp',
        r'whole.cell', r'\btevc\b', r'two.electrode',
        r'membrane potential', r'\bFlipR\b',
    ]),

    # --- Kinase-specific assays ---
    ("KINOMEscan / KinomeScan", [
        r'KINOME\s*scan', r'Kinome\s*scan', r'DiscoverX',
    ]),
    ("ADP-Glo Kinase Assay", [
        r'ADP-Glo', r'ADP.Glo',
    ]),
    ("KinaseGlo / Kinase-Glo", [
        r'Kinase.?Glo',
    ]),
    ("Mobility Shift Assay", [
        r'mobility shift',
    ]),

    # --- Reporter Gene / Cell-based ---
    ("Luciferase Reporter Gene Assay", [
        r'luciferase', r'reporter gene',
    ]),
    ("Cell Viability / Proliferation Assay (MTT/MTS/CTG/WST/Alamar)", [
        r'\bMTT\b', r'\bMTS\b', r'CellTiter', r'\bWST\b',
        r'Alamar\s*Blue', r'resazurin', r'cell viability',
        r'cell proliferation', r'antiproliferative',
        r'cytotoxicity', r'cell growth inhibition',
    ]),
    ("Calcium Flux / FLIPR Assay", [
        r'calcium flux', r'calcium mobilization', r'calcium signal',
        r'intracellular calcium', r'\bFLIPR\b',
        r'calcium assay', r'Ca2\+',
    ]),
    ("cAMP Assay", [
        r'\bcAMP\b(?!.*hydrolysis)', r'cyclic AMP', r'adenylyl cyclase',
    ]),
    ("GTPgammaS Binding Assay", [
        r'GTP.?gamma.?S', r'GTPgammaS', r'\[35S\]GTP',
    ]),

    # --- Colorimetric / Spectrophotometric ---
    ("Malachite Green Assay", [
        r'malachite green',
    ]),
    ("Ellman's Method", [
        r"Ellman", r'DTNB',
    ]),
    ("Colorimetric / Spectrophotometric Assay", [
        r'colorimetric', r'spectrophotometr', r'absorbance',
        r'optical density', r'\bOD\b(?=.*nm)', r'UV-Vis',
        r'chromogenic', r'p-nitrophenol', r'pNPP',
    ]),

    # --- Pull-down / Affinity-based ---
    ("Pull-down / Affinity Capture Assay", [
        r'pull.down', r'pulldown', r'affinity capture',
        r'Kinobead', r'chemoproteomic',
    ]),

    # --- Calorimetry (non-ITC) ---
    ("Differential Scanning Calorimetry (DSC)", [
        r'\bDSC\b', r'differential scanning calorimetry',
    ]),

    # --- Filter binding ---
    ("Filter Binding Assay", [
        r'filter binding',
    ]),

    # --- Flow cytometry ---
    ("Flow Cytometry", [
        r'flow cytometry', r'\bFACS\b',
    ]),

    # --- Western blot ---
    ("Western Blot", [
        r'western blot',
    ]),

    # --- Enzyme inhibition (generic, catch-all) ---
    ("Enzyme Inhibition Assay (generic)", [
        r'inhibition of\b', r'inhibitory activity', r'inhibitor',
        r'\bIC50\b', r'\bKi\b', r'\bEC50\b', r'\bKd\b',
        r'enzyme assay', r'enzymatic assay', r'substrate cleavage',
        r'enzyme.substrate', r'catalytic activity',
    ]),

    # --- Binding (generic, catch-all) ---
    ("Binding Assay (generic)", [
        r'binding affinity', r'displacement', r'competition',
        r'competitive', r'\bKD\b', r'dissociation constant',
        r'binding assay', r'ligand binding',
    ]),
])


def categorize_description(desc):
    """Categorize a single description into assay type(s)."""
    if pd.isna(desc):
        return "Uncategorized"

    desc_str = str(desc)
    matched = []

    for category, patterns in ASSAY_CATEGORIES.items():
        for pattern in patterns:
            if re.search(pattern, desc_str, re.IGNORECASE):
                matched.append(category)
                break  # Move to next category once matched

    if not matched:
        return "Uncategorized"

    # Return the first (most specific) match as primary category
    return matched[0]


def categorize_all(desc):
    """Return ALL matching categories for a description."""
    if pd.isna(desc):
        return ["Uncategorized"]

    desc_str = str(desc)
    matched = []

    for category, patterns in ASSAY_CATEGORIES.items():
        for pattern in patterns:
            if re.search(pattern, desc_str, re.IGNORECASE):
                matched.append(category)
                break

    return matched if matched else ["Uncategorized"]


def main():
    print("Loading data...")
    df = pd.read_csv('/data/mwu11/LLM_affinity/BindingDB/all_description.csv')
    print(f"Total rows: {len(df)}")
    print(f"Unique descriptions: {df['DESCRIPTION'].nunique()}")

    # Work with unique descriptions for efficiency
    unique_descs = df['DESCRIPTION'].drop_duplicates().reset_index(drop=True)
    print(f"\nCategorizing {len(unique_descs)} unique descriptions...")

    # Categorize each unique description (primary category)
    unique_descs_df = pd.DataFrame({'DESCRIPTION': unique_descs})
    unique_descs_df['PRIMARY_ASSAY'] = unique_descs_df['DESCRIPTION'].apply(categorize_description)
    unique_descs_df['ALL_ASSAYS'] = unique_descs_df['DESCRIPTION'].apply(categorize_all)

    # Merge back to full dataframe
    df = df.merge(unique_descs_df, on='DESCRIPTION', how='left')

    # === RESULTS ===

    # 1. Counts by PRIMARY assay category (each description counted once)
    print("\n" + "=" * 70)
    print("PRIMARY ASSAY CATEGORY COUNTS (per unique description)")
    print("=" * 70)
    primary_counts_unique = unique_descs_df['PRIMARY_ASSAY'].value_counts()
    for assay, count in primary_counts_unique.items():
        print(f"  {assay:55s} {count:>8,}")
    print(f"  {'TOTAL':55s} {primary_counts_unique.sum():>8,}")

    # 2. Counts by PRIMARY assay category (per row)
    print("\n" + "=" * 70)
    print("PRIMARY ASSAY CATEGORY COUNTS (per row, total entries)")
    print("=" * 70)
    primary_counts_rows = df['PRIMARY_ASSAY'].value_counts()
    for assay, count in primary_counts_rows.items():
        print(f"  {assay:55s} {count:>8,}")
    print(f"  {'TOTAL':55s} {primary_counts_rows.sum():>8,}")

    # 3. Multi-label counts (each description can match multiple assays)
    print("\n" + "=" * 70)
    print("ALL ASSAY MATCHES (multi-label, per unique description)")
    print("=" * 70)
    all_assay_counter = Counter()
    for assays in unique_descs_df['ALL_ASSAYS']:
        for a in assays:
            all_assay_counter[a] += 1
    for assay, count in all_assay_counter.most_common():
        print(f"  {assay:55s} {count:>8,}")

    # 4. Save results
    output_path = '/data/mwu11/LLM_affinity/BindingDB/assay_categorization_results.csv'
    df[['DESCRIPTION', 'PRIMARY_ASSAY']].to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # 5. Show some uncategorized examples for review
    uncategorized = unique_descs_df[unique_descs_df['PRIMARY_ASSAY'] == 'Uncategorized']['DESCRIPTION']
    if len(uncategorized) > 0:
        print(f"\n--- Sample Uncategorized Descriptions ({len(uncategorized)} total) ---")
        for d in uncategorized.head(20):
            print(f"  {str(d)[:200]}")


if __name__ == '__main__':
    main()