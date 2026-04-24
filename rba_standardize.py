"""Standardize RBA (Radioligand Binding Assay) extraction JSON files to conform to rba_schema.json.

Normalizations applied to each record's `structured_description`:
- Replace Unicode micro signs (µ U+00B5, μ U+03BC) with ASCII 'u' everywhere in strings.
- biological_preparation.protein_concentration -> string (heterogeneous units; just ASCII-u + strip).
- radioligand.concentration_used              -> number in nM or null.
    * molar units: uM*1000, mM*1e6, pM/1000, fM/1e6
    * ranges ("0.02-150 nM"), multi-values, qualitative -> null
- radioligand.kd_value                         -> {value: number (nM), source: str}
    * Consumes and drops a top-level `unit` field if present.
- radioligand.specific_activity                -> {value: str|number, unit: enum}
    * Light cleaning; preserves shape.
- nsb_definition.compound_concentration        -> number in uM or null.
    * molar units: nM/1000, mM*1000, pM/1e6
- assay_conditions.incubation_time (raw shape {value, unit}) -> number or string in minutes.
- assay_conditions.total_assay_volume          -> number in uL or null (mL*1000).
- assay_conditions.pH                          -> number.  NOTE: schema uses key "pH" (capital H).
- assay_conditions.incubation_temperature      -> string.
- test_compound.concentration_range            -> string with ASCII-u only.
- data_analysis.hill_slope_reported            -> preserved {reported, value}.
- All other string fields: strip + ASCII-u only.

Usage:
    python rba_standardize.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


INPUT_DIR = Path("/data/mwu11/LLM_affinity/qwen_test/rba_extraction_results_two_step_bindingdb")
OUTPUT_DIR = Path("/data/mwu11/LLM_affinity/qwen_test/rba_extraction_results_two_step_bindingdb_standardized")
SCHEMA_PATH = Path("/data/mwu11/LLM_affinity/rba_schema.json")


MICRO_CHARS = ["µ", "μ"]  # µ, μ


def ascii_u(text):
    if not isinstance(text, str):
        return text
    out = text
    for ch in MICRO_CHARS:
        out = out.replace(ch, "u")
    return out


def deep_ascii_u(obj):
    if isinstance(obj, str):
        return ascii_u(obj)
    if isinstance(obj, dict):
        return {k: deep_ascii_u(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_ascii_u(v) for v in obj]
    return obj


# --- unit tables ------------------------------------------------------------

CONC_FACTORS_TO_NM = {
    "m": 1_000_000_000.0,
    "mm": 1_000_000.0,
    "um": 1000.0,
    "nm": 1.0,
    "pm": 0.001,
    "fm": 0.000001,
}

CONC_FACTORS_TO_UM = {
    "m": 1_000_000.0,
    "mm": 1000.0,
    "um": 1.0,
    "nm": 0.001,
    "pm": 0.000001,
}

VOL_FACTORS_TO_UL = {
    "l": 1_000_000.0,
    "ml": 1000.0,
    "ul": 1.0,
}

TIME_FACTORS_TO_MIN = {
    "h": 60.0,
    "hr": 60.0,
    "hour": 60.0,
    "hours": 60.0,
    "min": 1.0,
    "mins": 1.0,
    "minute": 1.0,
    "minutes": 1.0,
    "s": 1.0 / 60.0,
    "sec": 1.0 / 60.0,
    "secs": 1.0 / 60.0,
    "second": 1.0 / 60.0,
    "seconds": 1.0 / 60.0,
    "ms": 1.0 / 60000.0,
}

MASS_OR_AMOUNT_RE = re.compile(
    r"(?:"
    r"\b(?:p?g|ng|ug|mg|kg|g)\s*/\s*(?:u?l|m?l|kg)\b"
    r"|\b(?:f|p|n|u|m)?mol(?!ar)\b"
    r"|\b(?:f|p|n|u|m)?moles?\b"
    r")",
    re.IGNORECASE,
)

NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
RANGE_RE = re.compile(
    r"^\s*~?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*[-–]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b"
)
MULTIVALUE_HINT = re.compile(r"\bor\b|,|;", re.IGNORECASE)


def _fmt(num):
    if num is None:
        return None
    if abs(num - round(num)) < 1e-9:
        return str(int(round(num)))
    s = f"{num:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _find_unit(value_str, factor_table):
    s = ascii_u(value_str).lower()
    for key in sorted(factor_table.keys(), key=len, reverse=True):
        pattern = r"(?<![a-z])" + re.escape(key) + r"(?![a-z])"
        if re.search(pattern, s):
            return key, factor_table[key]
    return None, None


# --- concentration helpers --------------------------------------------------


def _concentration_to_number(value, factor_table):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = ascii_u(value).strip()
    if not s:
        return None
    if MASS_OR_AMOUNT_RE.search(s):
        return None
    if "^" in s:  # e.g. "10^-5 M" — not parseable without deeper logic
        return None
    if RANGE_RE.match(s):
        return None
    if MULTIVALUE_HINT.search(s):
        return None
    _, factor = _find_unit(s, factor_table)
    if factor is None:
        return None
    m = NUMBER_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0)) * factor
    except ValueError:
        return None


def normalize_concentration_nm(value):
    return _concentration_to_number(value, CONC_FACTORS_TO_NM)


def normalize_concentration_um(value):
    return _concentration_to_number(value, CONC_FACTORS_TO_UM)


# --- volume / time / wavelength helpers ------------------------------------


def normalize_volume_ul(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = ascii_u(value).strip()
    if not s:
        return None
    if MASS_OR_AMOUNT_RE.search(s):
        return None
    if RANGE_RE.match(s):
        return None
    _, factor = _find_unit(s, VOL_FACTORS_TO_UL)
    if factor is None:
        if NUMBER_RE.fullmatch(s):
            return float(s)
        return None
    m = NUMBER_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0)) * factor
    except ValueError:
        return None


def normalize_incubation_time(value):
    """Input may be {value, unit} or a bare string/number.
    Output: number in minutes, or string for a range/qualitative, or None.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        v = value.get("value")
        u = value.get("unit")
        factor = 1.0
        if isinstance(u, str) and u.strip():
            _, f = _find_unit(ascii_u(u), TIME_FACTORS_TO_MIN)
            if f is not None:
                factor = f
        if v is None:
            return None
        if isinstance(v, (int, float)):
            scaled = float(v) * factor
            if abs(scaled - round(scaled)) < 1e-9:
                return int(round(scaled))
            return scaled
        if isinstance(v, str):
            s = ascii_u(v).strip()
            if not s:
                return None
            pair = RANGE_RE.match(s)
            if pair:
                a = float(pair.group(1)) * factor
                b = float(pair.group(2)) * factor
                return f"{_fmt(a)}-{_fmt(b)}"
            m = NUMBER_RE.search(s)
            if m:
                scaled = float(m.group(0)) * factor
                if abs(scaled - round(scaled)) < 1e-9:
                    return int(round(scaled))
                return scaled
            return s
        return None

    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        s = ascii_u(value).strip()
        if not s:
            return None
        _, factor = _find_unit(s, TIME_FACTORS_TO_MIN)
        if factor is None:
            factor = 1.0
        pair = RANGE_RE.match(s)
        if pair:
            a = float(pair.group(1)) * factor
            b = float(pair.group(2)) * factor
            return f"{_fmt(a)}-{_fmt(b)}"
        m = NUMBER_RE.search(s)
        if m:
            scaled = float(m.group(0)) * factor
            if abs(scaled - round(scaled)) < 1e-9:
                return int(round(scaled))
            return scaled
        return s
    return None


def normalize_ph(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = ascii_u(value).strip()
        m = NUMBER_RE.search(s)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


def normalize_temperature(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return _fmt(float(value))
    if not isinstance(value, str):
        return str(value)
    s = ascii_u(value).strip()
    if not s:
        return None
    k_match = re.fullmatch(r"\s*([-+]?\d*\.?\d+)\s*K\s*", s)
    if k_match:
        try:
            return _fmt(float(k_match.group(1)) - 273.15)
        except ValueError:
            return s
    c_stripped = re.sub(r"\s*°?\s*[cCfF]\s*$", "", s).strip()
    if NUMBER_RE.fullmatch(c_stripped):
        try:
            return _fmt(float(c_stripped))
        except ValueError:
            pass
    return s


def normalize_text(value):
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    return ascii_u(value).strip() or None


def normalize_concentration_range_str(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return ascii_u(value).strip() or None
    return str(value)


# --- sub-object helpers -----------------------------------------------------


def normalize_kd_value(kd):
    """Input may include an extra `unit` field; convert value to nM and drop `unit`."""
    if not isinstance(kd, dict):
        return {"value": None, "source": "Not specified"}

    unit = kd.get("unit")
    value = kd.get("value")
    source = normalize_text(kd.get("source")) or "Not specified"

    factor = 1.0
    unit_known = False
    if isinstance(unit, str) and unit.strip():
        _, f = _find_unit(ascii_u(unit), CONC_FACTORS_TO_NM)
        if f is not None:
            factor = f
            unit_known = True

    num = None
    if value is None:
        num = None
    elif isinstance(value, (int, float)):
        num = float(value) * factor
    elif isinstance(value, str):
        s = ascii_u(value).strip()
        if not s or MASS_OR_AMOUNT_RE.search(s):
            num = None
        elif RANGE_RE.match(s) or MULTIVALUE_HINT.search(s):
            num = None
        else:
            _, f2 = _find_unit(s, CONC_FACTORS_TO_NM)
            local_factor = f2 if f2 is not None else (factor if unit_known else None)
            if local_factor is None:
                num = None
            else:
                m = NUMBER_RE.search(s)
                if m:
                    try:
                        num = float(m.group(0)) * local_factor
                    except ValueError:
                        num = None
    return {"value": num, "source": source}


def normalize_specific_activity(sa):
    """Schema: {value: str|number, unit: enum}. Preserve shape, light cleaning."""
    if not isinstance(sa, dict):
        return {"value": None, "unit": None}
    value = sa.get("value")
    unit = sa.get("unit")
    if isinstance(value, str):
        value = ascii_u(value).strip() or None
    if isinstance(unit, str):
        unit = ascii_u(unit).strip() or None
    return {"value": value, "unit": unit}


def normalize_reported_value(obj, required_field="reported"):
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    if required_field in out and isinstance(out[required_field], str):
        out[required_field] = ascii_u(out[required_field]).strip() or None
    v = out.get("value")
    if isinstance(v, str):
        s = ascii_u(v).strip()
        m = NUMBER_RE.search(s)
        if m:
            try:
                out["value"] = float(m.group(0))
            except ValueError:
                out["value"] = None
        else:
            out["value"] = None
    return out


# --- full standardization ---------------------------------------------------


def standardize_structured_description(sd):
    if not isinstance(sd, dict):
        return sd

    out = {}

    bp = sd.get("biological_preparation") or {}
    out["biological_preparation"] = {
        "source_type": normalize_text(bp.get("source_type")),
        "tissue_or_cell_line": normalize_text(bp.get("tissue_or_cell_line")),
        "species": normalize_text(bp.get("species")),
        "target_protein": normalize_text(bp.get("target_protein")),
        "protein_concentration": normalize_text(bp.get("protein_concentration")),
    }

    rl = sd.get("radioligand") or {}
    out["radioligand"] = {
        "name": normalize_text(rl.get("name")),
        "isotope": normalize_text(rl.get("isotope")),
        "specific_activity": normalize_specific_activity(rl.get("specific_activity")),
        "concentration_used": normalize_concentration_nm(rl.get("concentration_used")),
        "kd_value": normalize_kd_value(rl.get("kd_value")),
    }

    tc = sd.get("test_compound") or {}
    out["test_compound"] = {
        "description": normalize_text(tc.get("description")),
        "concentration_range": normalize_concentration_range_str(tc.get("concentration_range")),
        "vehicle_solvent": normalize_text(tc.get("vehicle_solvent")),
    }

    nsb = sd.get("nsb_definition") or {}
    out["nsb_definition"] = {
        "compound_name": normalize_text(nsb.get("compound_name")),
        "compound_concentration": normalize_concentration_um(nsb.get("compound_concentration")),
        "selectivity_rationale": normalize_text(nsb.get("selectivity_rationale")),
    }

    ac = sd.get("assay_conditions") or {}
    out["assay_conditions"] = {
        "assay_format": normalize_text(ac.get("assay_format")),
        "buffer_composition": normalize_text(ac.get("buffer_composition")),
        "pH": normalize_ph(ac.get("pH") if "pH" in ac else ac.get("ph")),
        "incubation_temperature": normalize_temperature(ac.get("incubation_temperature")),
        "incubation_time": normalize_incubation_time(ac.get("incubation_time")),
        "equilibrium_confirmed": normalize_text(ac.get("equilibrium_confirmed")),
        "total_assay_volume": normalize_volume_ul(ac.get("total_assay_volume")),
    }

    sep = sd.get("separation_and_detection") or {}
    out["separation_and_detection"] = {
        "separation_method": normalize_text(sep.get("separation_method")),
        "filter_type": normalize_text(sep.get("filter_type")),
        "wash_protocol": normalize_text(sep.get("wash_protocol")),
        "detection_instrument": normalize_text(sep.get("detection_instrument")),
        "scintillation_cocktail": normalize_text(sep.get("scintillation_cocktail")),
    }

    da = sd.get("data_analysis") or {}
    reps = da.get("replicates") or {}
    out["data_analysis"] = {
        "endpoint_reported": normalize_text(da.get("endpoint_reported")),
        "ic50_to_ki_conversion": normalize_text(da.get("ic50_to_ki_conversion")),
        "fitting_model": normalize_text(da.get("fitting_model")),
        "hill_slope_reported": normalize_reported_value(da.get("hill_slope_reported")),
        "software": normalize_text(da.get("software")),
        "replicates": {
            "technical_replicates": normalize_text(reps.get("technical_replicates")),
            "independent_experiments": (
                reps.get("independent_experiments")
                if isinstance(reps.get("independent_experiments"), (int, float))
                else normalize_text(reps.get("independent_experiments"))
            ),
        },
    }

    return out


# --- optional schema validation ---------------------------------------------


def try_validate(record_for_schema):
    try:
        import jsonschema  # type: ignore
    except Exception:
        return None
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    try:
        jsonschema.validate(record_for_schema, schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e)


# --- driver -----------------------------------------------------------------


def process_file(in_path: Path, out_path: Path, validator_issues: list):
    with in_path.open() as f:
        data = json.load(f)

    data = deep_ascii_u(data)

    if not isinstance(data, dict):
        out_path.write_text(json.dumps(data, indent=4, ensure_ascii=False))
        return

    for rid, rec in data.items():
        if not isinstance(rec, dict):
            continue
        sd = rec.get("structured_description")
        if sd is None:
            continue
        new_sd = standardize_structured_description(sd)
        rec["structured_description"] = new_sd

        validation_record = {"pmid": str(rec.get("pmid", "")), **new_sd}
        result = try_validate(validation_record)
        if result is not None:
            ok, err = result
            if not ok:
                validator_issues.append(f"{in_path.name}::{rid}: {err.splitlines()[0]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def main():
    if not INPUT_DIR.is_dir():
        print(f"Input directory not found: {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validator_issues: list = []
    files = sorted(INPUT_DIR.glob("*.json"))
    for in_path in files:
        out_path = OUTPUT_DIR / in_path.name
        process_file(in_path, out_path, validator_issues)

    print(f"Processed {len(files)} file(s). Output -> {OUTPUT_DIR}")
    if validator_issues:
        print(f"Schema validation issues: {len(validator_issues)} (showing up to 20):")
        for line in validator_issues[:20]:
            print(" -", line)


if __name__ == "__main__":
    main()
