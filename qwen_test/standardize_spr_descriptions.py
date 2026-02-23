#!/usr/bin/env python3
"""
Standardize structured_description fields in SPR extraction results.

Normalizes units, categorical fields, and formatting to match the
updated spr_schema.json enums and conventions.

Usage:
    python standardize_spr_descriptions.py
"""

import json
import re
import collections
from pathlib import Path


# =============================================================================
# Flow rate unit standardization
# =============================================================================

# Map various unit strings to the canonical µL/min (Unicode MICRO SIGN U+00B5)
FLOW_RATE_UNIT_MAP = {
    "uL/min": "µL/min",
    "ul/min": "µL/min",
    "\u03bcL/min": "µL/min",   # Greek mu
    "\u03bcl/min": "µL/min",   # Greek mu lowercase l
    "\u00b5L/min": "µL/min",   # Already correct (micro sign)
    "\u00b5l/min": "µL/min",   # Micro sign lowercase l
    "µL/min": "µL/min",        # Already correct
}

# Units that require value conversion (multiply factor to get µL/min)
FLOW_RATE_CONVERT = {
    "uL/s": 60,
    "\u03bcL/s": 60,
    "\u00b5L/s": 60,
    "µL/s": 60,
    "mL/min": 1000,
    "ml/min": 1000,
    "ML/min": 1000,
    "L/min": 1_000_000,
}

# Invalid units — set flow rate to null
FLOW_RATE_INVALID = {
    "\u03bcL", "uL", "\u00b5L", "µL",  # incomplete (missing /min)
    "\u03bcg/min", "ug/min",            # wrong dimension
    "rpm",                               # wrong type entirely
}


def standardize_flow_rate(flow_rate):
    """Standardize assay_flow_rate object. Returns modified dict or None."""
    if flow_rate is None:
        return None
    unit = flow_rate.get("unit")
    value = flow_rate.get("value")
    if unit is None:
        return flow_rate

    unit_stripped = unit.strip()

    # Check invalid
    if unit_stripped in FLOW_RATE_INVALID:
        return None

    # Check direct mapping (no conversion needed)
    if unit_stripped in FLOW_RATE_UNIT_MAP:
        flow_rate["unit"] = "µL/min"
        return flow_rate

    # Check conversion needed
    if unit_stripped in FLOW_RATE_CONVERT:
        factor = FLOW_RATE_CONVERT[unit_stripped]
        if value is not None:
            flow_rate["value"] = round(value * factor, 4)
        flow_rate["unit"] = "µL/min"
        return flow_rate

    # Unknown unit — keep as-is but log
    return flow_rate


# =============================================================================
# Assay type standardization
# =============================================================================

VALID_ASSAY_TYPES = {
    "Single Cycle Kinetics",
    "Multi Cycle Kinetics",
    "Single Cycle Steady-State Affinity",
    "Multi Cycle Steady-State Affinity",
    "Screening",
    "Unknown",
}

# Build case-insensitive lookup
_ASSAY_TYPE_LOWER = {v.lower(): v for v in VALID_ASSAY_TYPES}

ASSAY_TYPE_ALIASES = {
    "kinetic titration": "Single Cycle Kinetics",
}


def standardize_assay_type(assay_type):
    """Map assay_type to schema enum value."""
    if not assay_type:
        return assay_type
    s = assay_type.strip()
    if s in VALID_ASSAY_TYPES:
        return s
    low = s.lower()
    if low in _ASSAY_TYPE_LOWER:
        return _ASSAY_TYPE_LOWER[low]
    if low in ASSAY_TYPE_ALIASES:
        return ASSAY_TYPE_ALIASES[low]
    # Composite: "X, Y" or "X or Y"
    if "," in s:
        first = s.split(",")[0].strip()
        return standardize_assay_type(first) if first else "Unknown"
    if " or " in s.lower():
        return "Unknown"
    return "Unknown"


# =============================================================================
# Immobilization strategy standardization
# =============================================================================

def standardize_strategy(strategy):
    """Classify immobilization strategy to schema enum."""
    if not strategy:
        return strategy
    s = strategy.strip()
    low = s.lower()

    # Amine coupling
    if re.search(r'amine[\s-]?coupl', low) or 'edc/nhs' in low or 'nhs/edc' in low:
        return "Amine coupling"

    # SA-Biotin capture (check before NTA since some mention both)
    if any(kw in low for kw in ['streptavidin', 'sa-biotin', 'neutravidin', 'avidin-biotin']):
        return "SA-Biotin capture"
    if 'biotin' in low and 'captur' in low:
        return "SA-Biotin capture"
    if 'biotin' in low and ('sa ' in low or 'sa-' in low or 'strep' in low):
        return "SA-Biotin capture"

    # NTA-His capture
    if any(kw in low for kw in ['nta', 'his-tag', 'his capture', 'his tag', 'ni-nta', 'ni2+']):
        return "NTA-His capture"

    # GST capture
    if 'gst' in low:
        return "GST capture"

    # Aldehyde coupling
    if 'aldehyde' in low and 'coupl' in low:
        return "Aldehyde coupling"

    # Thiol coupling
    if 'thiol' in low or 'gold-thiol' in low:
        return "Thiol coupling"

    # Photo-crosslinking
    if 'photo' in low and 'cross' in low:
        return "Photo-crosslinking"
    if 'uv' in low and 'cross' in low:
        return "Photo-crosslinking"

    # Antibody capture
    if 'antibody' in low and 'captur' in low:
        return "Antibody capture"

    # Direct immobilization / vague references
    if any(kw in low for kw in ['direct immobil', 'direct coat', 'direct print',
                                  'according to', 'as described', 'previous method',
                                  'not specified', 'unknown', 'manufacturer']):
        return "Direct immobilization"

    # Biotin capture without explicit streptavidin (on SA chip context)
    if 'biotin' in low:
        return "SA-Biotin capture"

    # Fallback
    return "Other"


# =============================================================================
# Fitting model standardization
# =============================================================================

def standardize_fitting_model(model):
    """Classify fitting model to schema enum."""
    if not model:
        return model
    s = model.strip()
    low = s.lower()

    if low in ('not specified', 'n/a', 'unknown', 'none'):
        return None

    # Order matters: check specific models first

    # Mass transport
    if 'mass transport' in low or 'mass transfer' in low:
        return "1:1 binding model with mass transport"

    # Two-state
    if 'two-state' in low or 'two state' in low or 'conformational change' in low:
        return "Two-state binding model"

    # Heterogeneous
    if 'heterogeneous' in low:
        return "Heterogeneous binding model"

    # Bivalent
    if 'bivalent' in low:
        return "Bivalent analyte model"

    # Steady-state (but not when combined with 1:1 kinetic)
    is_steady = bool(re.search(r'steady[\s-]?state', low) or
                     'equilibrium' in low or
                     'dose-response' in low or 'dose response' in low)
    is_kinetic_11 = bool(re.search(r'1\s*[:/]\s*1', low) and
                         ('kinetic' in low or 'langmuir' in low))
    if is_steady and not is_kinetic_11:
        return "Steady-state affinity"

    # 1:1 Langmuir (catches 1:1, 1/1, langmuir, bimolecular)
    if (re.search(r'1\s*[:/]\s*1', low) or
        'langmuir' in low or
        'bimolecular interaction' in low or
        'one-to-one' in low or
        'one to one' in low):
        return "1:1 Langmuir binding model"

    # If both steady-state and kinetic mentioned, kinetic wins
    if is_steady and is_kinetic_11:
        return "1:1 Langmuir binding model"

    return "Other"


# =============================================================================
# Subtraction method standardization
# =============================================================================

def standardize_subtraction_method(method):
    """Classify subtraction method to schema enum."""
    if not method:
        return method
    s = method.strip()
    low = s.lower()

    if low in ('not specified', 'n/a', 'unknown', 'none'):
        return None

    # Double referenced
    if 'double' in low:
        return "Double referenced"

    # Multi-method that includes reference + blank/buffer = effectively double
    has_ref = 'reference' in low
    has_blank = 'blank' in low or 'buffer' in low
    has_solvent = 'solvent' in low or 'dmso' in low
    if has_ref and (has_blank or has_solvent):
        return "Double referenced"

    # Reference subtracted
    if has_ref:
        return "Reference subtracted"

    # Solvent correction
    if has_solvent:
        return "Solvent correction"

    # Background subtracted
    if 'background' in low:
        return "Background subtracted"

    # Blank subtracted
    if has_blank and 'subtract' in low:
        return "Blank subtracted"

    return "Other"


# =============================================================================
# Density RU normalization
# =============================================================================

def standardize_density_ru(density):
    """Normalize dashes in density_ru values."""
    if density is None:
        return None
    s = str(density)
    # Replace en-dash and em-dash with hyphen
    s = s.replace('\u2013', '-').replace('\u2014', '-')
    return s


# =============================================================================
# Entry-level standardization
# =============================================================================

def snapshot(sd):
    """Extract target field values for change detection."""
    vals = {}
    imm = sd.get("immobilization") or {}
    ac = sd.get("assay_conditions") or {}
    da = sd.get("data_analysis") or {}
    fr = ac.get("assay_flow_rate") or {}

    vals["strategy"] = imm.get("strategy")
    vals["density_ru"] = str(imm.get("density_ru")) if imm.get("density_ru") is not None else None
    vals["assay_type"] = ac.get("assay_type")
    vals["flow_rate_unit"] = fr.get("unit") if isinstance(fr, dict) else None
    vals["flow_rate_value"] = fr.get("value") if isinstance(fr, dict) else None
    vals["fitting_model"] = da.get("fitting_model")
    vals["subtraction_method"] = da.get("subtraction_method")
    return vals


def standardize_entry(sd):
    """Apply all standardizations to a structured_description dict."""
    if sd is None:
        return sd

    # Immobilization
    imm = sd.get("immobilization")
    if imm:
        if "strategy" in imm:
            imm["strategy"] = standardize_strategy(imm["strategy"])
        if "density_ru" in imm:
            imm["density_ru"] = standardize_density_ru(imm["density_ru"])

    # Assay conditions
    ac = sd.get("assay_conditions")
    if ac:
        if "assay_type" in ac:
            ac["assay_type"] = standardize_assay_type(ac["assay_type"])
        if "assay_flow_rate" in ac:
            ac["assay_flow_rate"] = standardize_flow_rate(ac.get("assay_flow_rate"))

    # Data analysis
    da = sd.get("data_analysis")
    if da:
        if "fitting_model" in da:
            da["fitting_model"] = standardize_fitting_model(da["fitting_model"])
        if "subtraction_method" in da:
            da["subtraction_method"] = standardize_subtraction_method(da["subtraction_method"])

    return sd


# =============================================================================
# Main
# =============================================================================

def main():
    data_dir = Path(__file__).parent / "spr_extraction_results_two_step_bindingdb"

    counters = collections.Counter()
    json_files = sorted(data_dir.glob("*.json"))
    print(f"Processing {len(json_files)} JSON files...")

    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)

        modified = False
        for reactant_set_id, entry in data.items():
            sd = entry.get("structured_description")
            if sd is None:
                continue

            old = snapshot(sd)
            standardize_entry(sd)
            new = snapshot(sd)

            for field in old:
                if old[field] != new[field]:
                    counters[field] += 1
                    modified = True

        if modified:
            with open(jf, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            counters["files_modified"] += 1

    print(f"\nStandardization Summary:")
    print(f"  Files processed: {len(json_files)}")
    print(f"  Files modified:  {counters['files_modified']}")
    for field in ["flow_rate_unit", "flow_rate_value", "assay_type",
                  "strategy", "fitting_model", "subtraction_method", "density_ru"]:
        print(f"  {field:25s} changed: {counters[field]}")


if __name__ == "__main__":
    main()
