"""
DataFrame utility functions for extracting row data.

This module contains helper functions to extract protein, ligand, and affinity
data from DataFrame rows, reducing code duplication across batch processing methods.
"""
import pandas as pd
from typing import Optional, Dict, Any


def get_optional_str(row: pd.Series, col: Optional[str]) -> Optional[str]:
    """Get optional string value from a row."""
    if col and col in row.index and pd.notna(row[col]):
        return str(row[col])
    return None


def extract_affinity_data(
    row: pd.Series,
    affinity_type_col: Optional[str] = None,
    affinity_value_col: Optional[str] = None,
    affinity_relation_col: Optional[str] = None,
    affinity_unit_col: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Extract affinity data from a DataFrame row.

    Args:
        row: DataFrame row
        affinity_type_col: Column name for affinity type (e.g., "Kd")
        affinity_value_col: Column name for affinity value
        affinity_relation_col: Column name for affinity relation (e.g., "=")
        affinity_unit_col: Column name for affinity unit (e.g., "nM")

    Returns:
        Dict with affinity data or None if no value column specified
    """
    if not affinity_value_col or affinity_value_col not in row.index or pd.isna(row[affinity_value_col]):
        return None

    return {
        "type": get_optional_str(row, affinity_type_col) or "Kd",
        "value": float(row[affinity_value_col]) if pd.notna(row[affinity_value_col]) else None,
        "relation": get_optional_str(row, affinity_relation_col) or "=",
        "unit": get_optional_str(row, affinity_unit_col) or "nM"
    }


def extract_row_data(
    row: pd.Series,
    pmid_col: str = "PMID",
    description_col: str = "DESCRIPTION",
    reactant_set_id_col: Optional[str] = None,
    protein_col: Optional[str] = None,
    ligand_name_col: Optional[str] = None,
    ligand_smiles_col: Optional[str] = None,
    affinity_type_col: Optional[str] = None,
    affinity_value_col: Optional[str] = None,
    affinity_relation_col: Optional[str] = None,
    affinity_unit_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract all relevant data from a DataFrame row.

    Args:
        row: DataFrame row
        pmid_col: Column name for PMID
        description_col: Column name for description
        reactant_set_id_col: Column name for reactant set ID
        protein_col: Column name for protein
        ligand_name_col: Column name for ligand name
        ligand_smiles_col: Column name for ligand SMILES
        affinity_type_col: Column name for affinity type
        affinity_value_col: Column name for affinity value
        affinity_relation_col: Column name for affinity relation
        affinity_unit_col: Column name for affinity unit

    Returns:
        Dict containing extracted data
    """
    pmid = str(int(row[pmid_col]))
    description = row[description_col]

    reactant_set_id = None
    if reactant_set_id_col and reactant_set_id_col in row.index and pd.notna(row[reactant_set_id_col]):
        reactant_set_id = row[reactant_set_id_col]

    protein = get_optional_str(row, protein_col)
    ligand_name = get_optional_str(row, ligand_name_col)
    ligand_smiles = get_optional_str(row, ligand_smiles_col)

    affinity_data = extract_affinity_data(
        row,
        affinity_type_col=affinity_type_col,
        affinity_value_col=affinity_value_col,
        affinity_relation_col=affinity_relation_col,
        affinity_unit_col=affinity_unit_col
    )

    return {
        "pmid": pmid,
        "description": description,
        "reactant_set_id": reactant_set_id,
        "protein": protein,
        "ligand_name": ligand_name,
        "ligand_smiles": ligand_smiles,
        "affinity_data": affinity_data
    }


def build_pair_entry(
    pair_id: str,
    row_data: Dict[str, Any],
    pmid: int
) -> Dict[str, Any]:
    """
    Build a pair entry for batched extraction.

    Args:
        pair_id: Identifier for this pair
        row_data: Data extracted from row using extract_row_data
        pmid: PMID as integer

    Returns:
        Dict for use in batched extraction
    """
    return {
        "id": pair_id,
        "assay_description": row_data["description"],
        "protein": row_data["protein"],
        "ligand_smiles": row_data["ligand_smiles"],
        "affinity_data": row_data["affinity_data"]
    }


def build_metadata_entry(
    row_data: Dict[str, Any],
    pmid: int
) -> Dict[str, Any]:
    """
    Build metadata entry for output JSON.

    Args:
        row_data: Data extracted from row using extract_row_data
        pmid: PMID as integer

    Returns:
        Dict with metadata for JSON output
    """
    return {
        "reactant_set_id": int(row_data["reactant_set_id"]) if row_data["reactant_set_id"] else None,
        "pmid": pmid,
        "protein": row_data["protein"],
        "ligand": {
            "reference_name": row_data["ligand_name"],
            "smiles": row_data["ligand_smiles"]
        },
        "affinity_data": row_data["affinity_data"],
        "DESCRIPTION": row_data["description"]
    }
