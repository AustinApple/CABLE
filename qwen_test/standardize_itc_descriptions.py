#!/usr/bin/env python3
"""
Standardize structured_description fields in ITC extraction results.

Normalizes units and extracts numeric-only values to match itc_schema.json:
  - concentration → numeric in µM
  - solution_volume → numeric in µL
  - stirring_speed → numeric in rpm
  - temperature_c → numeric in °C
  - first_volume, subsequent_volume → numeric in µL
  - duration, spacing → numeric in s
  - total_count → numeric
  - instrument.model → canonical name
  - binding_model → standardized category

Usage:
    python standardize_itc_descriptions.py
"""

import json
import re
import collections
from pathlib import Path


# =============================================================================
# Helpers
# =============================================================================

def normalize_dashes(s):
    """Replace en-dash / em-dash with hyphen."""
    if s is None:
        return None
    return s.replace('\u2013', '-').replace('\u2014', '-')


def _try_float(s):
    """Try to parse a string as float, return None on failure."""
    try:
        return float(s.replace(',', ''))
    except (ValueError, AttributeError):
        return None


def _fmt_num(v):
    """Format a numeric value: drop trailing .0 for integers."""
    if v is None:
        return None
    if v == int(v):
        return str(int(v))
    return str(v)


# Regex for a single number (possibly with comma thousands separator)
_NUM = r'[\d,]+(?:\.\d+)?'
# Regex for a number or range (e.g. "10", "10-30", "10 to 30")
_NUM_OR_RANGE = rf'({_NUM})(?:\s*[-–]\s*({_NUM}))?'


# =============================================================================
# Concentration: extract numeric value in µM
# =============================================================================

# Conversion factors to µM
_CONC_TO_UM = {
    'µm': 1, 'um': 1, '\u03bcm': 1, '\u00b5m': 1,
    'microm': 1, 'micromolar': 1,
    'mm': 1000, 'millimolar': 1000,
    'nm': 0.001, 'nanomolar': 0.001,
    'pm': 0.000001,
    'm': 1e6,  # molar
}

# Pattern: number (optional range) + unit
_CONC_RE = re.compile(
    rf'({_NUM})\s*(?:[-–]\s*({_NUM})\s*)?'
    r'(µM|μM|\u03bcM|\u00b5M|uM|mM|nM|pM|M|micromolar|millimolar|nanomolar|microm)\b',
    re.IGNORECASE
)


def standardize_concentration(conc):
    """Extract numeric concentration in µM. Returns numeric string or original."""
    if not conc:
        return conc

    s = normalize_dashes(conc.strip())
    m = _CONC_RE.search(s)
    if not m:
        # Try bare number (already numeric-only)
        if re.fullmatch(rf'{_NUM}(?:\s*-\s*{_NUM})?', s.strip()):
            return s.strip()
        return conc

    v1_str, v2_str, unit = m.group(1), m.group(2), m.group(3)
    factor = _CONC_TO_UM.get(unit.lower().replace('\u03bc', 'µ').replace('\u00b5', 'µ'), None)
    if factor is None:
        return conc

    v1 = _try_float(v1_str)
    if v1 is None:
        return conc

    if v2_str:
        v2 = _try_float(v2_str)
        if v2 is None:
            return conc
        return f"{_fmt_num(v1 * factor)}-{_fmt_num(v2 * factor)}"

    return _fmt_num(v1 * factor)


# =============================================================================
# Volume: extract numeric value in µL
# =============================================================================

_VOL_TO_UL = {
    'µl': 1, 'ul': 1, '\u03bcl': 1, '\u00b5l': 1,
    'ml': 1000,
    'l': 1e6,
}

_VOL_RE = re.compile(
    rf'({_NUM})\s*(?:[-–]\s*({_NUM})\s*)?'
    r'(µL|μL|\u03bcL|\u03bcl|\u00b5L|\u00b5l|uL|ul|mL|ml|L)\b',
    re.IGNORECASE
)


def standardize_volume(vol):
    """Extract numeric volume in µL. Returns numeric string or original."""
    if not vol:
        return vol

    s = normalize_dashes(vol.strip())
    # Fix "8-μl" → "8 μl" (hyphen between number and unit)
    s = re.sub(r'(\d)-\s*([µμ\u03bc\u00b5uUmM])', r'\1 \2', s)
    m = _VOL_RE.search(s)
    if not m:
        if re.fullmatch(rf'{_NUM}(?:\s*-\s*{_NUM})?', s.strip()):
            return s.strip()
        return vol

    v1_str, v2_str, unit = m.group(1), m.group(2), m.group(3)
    factor = _VOL_TO_UL.get(unit.lower().replace('\u03bc', 'µ').replace('\u00b5', 'µ'), None)
    if factor is None:
        return vol

    v1 = _try_float(v1_str)
    if v1 is None:
        return vol

    if v2_str:
        v2 = _try_float(v2_str)
        if v2 is None:
            return vol
        return f"{_fmt_num(v1 * factor)}-{_fmt_num(v2 * factor)}"

    return _fmt_num(v1 * factor)


# =============================================================================
# Stirring speed: extract numeric value in rpm
# =============================================================================

def standardize_stirring_speed(speed):
    """Extract numeric stirring speed (rpm). Returns numeric string or original."""
    if not speed:
        return speed

    s = speed.strip()

    # Extract first numeric value
    m = re.search(rf'({_NUM})', s)
    if not m:
        return speed

    value = m.group(1).replace(',', '')

    # Verify it's a rotation-rate value (or bare number)
    low = s.lower()
    if any(kw in low for kw in ['rpm', 'r.p.m', 'rotations', 'rev']):
        return value
    if re.fullmatch(rf'{_NUM}', s):
        return value

    return speed


# =============================================================================
# Time: extract numeric value in seconds
# =============================================================================

_TIME_TO_S = {
    's': 1, 'sec': 1, 'second': 1, 'seconds': 1,
    'min': 60, 'mins': 60, 'minute': 60, 'minutes': 60,
    'h': 3600, 'hr': 3600, 'hrs': 3600, 'hour': 3600, 'hours': 3600,
}

_TIME_RE = re.compile(
    rf'({_NUM})\s*(?:[-–]\s*({_NUM})\s*)?'
    r'(seconds?|sec|mins?|minutes?|hrs?|hours?|s)\b',
    re.IGNORECASE
)

_TIME_OR_RE = re.compile(
    rf'({_NUM})\s+or\s+({_NUM})\s+'
    r'(seconds?|sec|mins?|minutes?|hrs?|hours?|s)\b',
    re.IGNORECASE
)


def standardize_time(time_str):
    """Extract numeric time in seconds. Returns numeric string or original."""
    if not time_str:
        return time_str

    s = normalize_dashes(time_str.strip())
    # Fix "2.5-min" → "2.5 min" (hyphen between number and unit)
    s = re.sub(r'(\d)-\s*(s|sec|min)', r'\1 \2', s)

    # Skip rate values (µL/s, µL/min) — not a time
    if '/' in s:
        return time_str

    # Skip compound descriptions with parentheses
    if '(' in s and ')' in s:
        return time_str

    # Already bare number — assume seconds
    if re.fullmatch(rf'{_NUM}(?:\s*-\s*{_NUM})?', s.strip()):
        return s.strip()

    # Try "X or Y unit" format first
    m = _TIME_OR_RE.search(s)
    if m:
        v1_str, v2_str, unit = m.group(1), m.group(2), m.group(3)
        factor = _TIME_TO_S.get(unit.lower().rstrip('s'), None)
        if factor is None:
            factor = _TIME_TO_S.get(unit.lower(), 1)
        v1 = _try_float(v1_str)
        v2 = _try_float(v2_str)
        if v1 is not None and v2 is not None:
            return f"{_fmt_num(v1 * factor)}-{_fmt_num(v2 * factor)}"

    # Try "X-Y unit" or "X unit" format
    m = _TIME_RE.search(s)
    if m:
        v1_str, v2_str, unit = m.group(1), m.group(2), m.group(3)
        # Normalize unit lookup
        unit_key = unit.lower().rstrip('s')
        factor = _TIME_TO_S.get(unit_key, None)
        if factor is None:
            factor = _TIME_TO_S.get(unit.lower(), 1)

        v1 = _try_float(v1_str)
        if v1 is None:
            return time_str

        if v2_str:
            v2 = _try_float(v2_str)
            if v2 is None:
                return time_str
            return f"{_fmt_num(v1 * factor)}-{_fmt_num(v2 * factor)}"

        return _fmt_num(v1 * factor)

    return time_str


# =============================================================================
# Temperature: extract numeric value in °C
# =============================================================================

def standardize_temperature(temp):
    """Extract numeric temperature (°C). Returns numeric string or original."""
    if not temp:
        return temp

    s = temp.strip()

    # Remove °C, C, degrees, Celsius, K labels
    s = re.sub(r'\s*°?\s*[Cc](elsius)?\s*$', '', s)
    s = re.sub(r'\s*degrees?\s*$', '', s, flags=re.IGNORECASE)
    s = s.strip()

    # Handle "25.0" → "25"
    if re.fullmatch(r'\d+\.0', s):
        s = s.split('.')[0]

    return s


# =============================================================================
# Total count: extract numeric value
# =============================================================================

def standardize_total_count(count):
    """Extract numeric total count. Returns numeric string or original."""
    if not count:
        return count

    s = count.strip()

    # Extract first number
    m = re.search(rf'({_NUM})', s)
    if m:
        value = m.group(1).replace(',', '')
        v = _try_float(value)
        if v is not None:
            return _fmt_num(v)

    return count


# =============================================================================
# Binding model standardization
# =============================================================================

def standardize_binding_model(model):
    """Map binding model descriptions to standardized categories."""
    if not model:
        return model

    s = model.strip()
    low = s.lower()

    if low in ('not specified', 'n/a', 'unknown', 'none', 'not reported',
               'not mentioned', 'not described', 'not stated'):
        return None

    # Two-site models
    if any(kw in low for kw in ['two-site', 'two site', 'two sets',
                                 'two independent', 'two binding site',
                                 'dual-site', 'dual site', '2-site',
                                 'two-sequential']):
        return "two-sites"

    # Sequential binding
    if 'sequential' in low and 'two' not in low:
        return "sequential"

    # One-site models (broadest category — check after two-site)
    one_site_kw = [
        'one-site', 'one site', 'single site', 'single-site',
        'single set of', 'one set of', 'one-set-of',
        'single binding site', 'single binding model',
        'independent site', 'independent binding',
        'independent fit', 'independent model',
        'simple binding', '1:1', 'one-to-one',
        'binary interaction', 'binary binding',
        'bimolecular binding',
    ]
    if any(kw in low for kw in one_site_kw):
        return "one-site"

    # Generic "sites model" → one-site
    if re.search(r'\bsit[ei]s?\b', low) and 'model' in low:
        return "one-site"

    # Catch remaining one-site variants
    if low in ('independent', 'onesites', '1 set of sites'):
        return "one-site"

    # Two identical binding sites (no cooperativity) → two-sites
    if 'two identical' in low or 'two ligand-binding' in low:
        return "two-sites"

    # Fitting/analysis methods (not binding models — set to null)
    if 'simplex' in low or 'global analysis' in low or 'global three-parameter' in low:
        return None
    if 'sigmoidal' in low or 'slope of' in low:
        return None

    # Fitting methods (not binding models — set to null)
    fitting_kw = ['least.squares', 'levenberg', 'marquardt', 'chi-square',
                  'curve.fitting', 'wiseman isotherm', 'nonlinear regression',
                  'nonlinear least']
    if any(re.search(kw, low) for kw in fitting_kw):
        return None

    # Software references are not binding models
    if any(kw in low for kw in ['origin', 'microcal', 'nitpic', 'sedphat']):
        return None

    return s


# =============================================================================
# Instrument model standardization
# =============================================================================

INSTRUMENT_MAP = [
    # Order matters: check more specific patterns first
    (r'peaq', 'MicroCal PEAQ-ITC'),
    (r'itc200|itc\s*200', 'MicroCal iTC200'),
    (r'vp[\s-]?itc', 'MicroCal VP-ITC'),
    (r'mcs[\s-]?itc|mcs\s+itc', 'MicroCal MCS-ITC'),
    (r'omega[\s-]?itc|microcal\s+omega', 'MicroCal Omega ITC'),
    (r'auto[\s-]?itc|ta\s+instruments.*itc|affinity\s+itc', 'TA Instruments Affinity ITC'),
    (r'nano[\s-]?itc|calorimetry\s+sciences', 'Nano-ITC'),
    (r'isothermal\s+titration\s+calorimetr', None),  # too generic, keep as-is
]


def standardize_instrument(model):
    """Standardize instrument model name."""
    if not model:
        return model

    s = model.strip()
    low = s.lower()

    for pattern, canonical in INSTRUMENT_MAP:
        if re.search(pattern, low):
            if canonical is None:
                return s  # keep as-is for generic descriptions
            return canonical

    return s


# =============================================================================
# Entry-level standardization
# =============================================================================

def snapshot(sd):
    """Extract target field values for change detection."""
    vals = {}
    inst = sd.get("instrument") or {}
    prot = sd.get("protein") or {}
    lig = sd.get("ligand") or {}
    ac = sd.get("assay_conditions") or {}
    da = sd.get("data_analysis") or {}
    ip = ac.get("injection_parameters") or {}

    vals["instrument_model"] = inst.get("model")
    vals["protein_concentration"] = prot.get("concentration")
    vals["protein_solution_volume"] = prot.get("solution_volume")
    vals["ligand_concentration"] = lig.get("concentration")
    vals["binding_model"] = da.get("binding_model")
    vals["temperature_c"] = ac.get("temperature_c")
    vals["stirring_speed"] = ac.get("stirring_speed")
    vals["first_volume"] = ip.get("first_volume")
    vals["subsequent_volume"] = ip.get("subsequent_volume")
    vals["duration"] = ip.get("duration")
    vals["spacing"] = ip.get("spacing")
    vals["total_count"] = ip.get("total_count")
    return vals


def standardize_entry(sd):
    """Apply all standardizations to a structured_description dict."""
    if sd is None:
        return sd

    # Instrument
    inst = sd.get("instrument")
    if inst:
        if "model" in inst:
            inst["model"] = standardize_instrument(inst["model"])

    # Protein
    prot = sd.get("protein")
    if prot:
        if "concentration" in prot:
            prot["concentration"] = standardize_concentration(prot["concentration"])
        if "solution_volume" in prot:
            prot["solution_volume"] = standardize_volume(prot["solution_volume"])

    # Ligand
    lig = sd.get("ligand")
    if lig:
        if "concentration" in lig:
            lig["concentration"] = standardize_concentration(lig["concentration"])

    # Assay conditions
    ac = sd.get("assay_conditions")
    if ac:
        if "temperature_c" in ac:
            ac["temperature_c"] = standardize_temperature(ac["temperature_c"])
        if "stirring_speed" in ac:
            ac["stirring_speed"] = standardize_stirring_speed(ac["stirring_speed"])

        # Injection parameters
        ip = ac.get("injection_parameters")
        if ip:
            if "first_volume" in ip:
                ip["first_volume"] = standardize_volume(ip["first_volume"])
            if "subsequent_volume" in ip:
                ip["subsequent_volume"] = standardize_volume(ip["subsequent_volume"])
            if "duration" in ip:
                ip["duration"] = standardize_time(ip["duration"])
            if "spacing" in ip:
                ip["spacing"] = standardize_time(ip["spacing"])
            if "total_count" in ip:
                ip["total_count"] = standardize_total_count(ip["total_count"])

    # Data analysis
    da = sd.get("data_analysis")
    if da:
        if "binding_model" in da:
            da["binding_model"] = standardize_binding_model(da["binding_model"])

    return sd


# =============================================================================
# Main
# =============================================================================

def main():
    data_dir = Path(__file__).parent / "itc_extraction_results_two_step_bindingdb"

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
    for field in ["instrument_model", "protein_concentration",
                  "protein_solution_volume", "ligand_concentration",
                  "binding_model", "temperature_c", "stirring_speed",
                  "first_volume", "subsequent_volume", "duration",
                  "spacing", "total_count"]:
        print(f"  {field:30s} changed: {counters[field]}")


if __name__ == "__main__":
    main()
