"""IUPAC name -> SMILES with a fallback cascade: OPSIN -> IUPAC2Struct -> ChemConv."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_names
from .router import BACKENDS, ConversionResult, ConverterError, convert_detailed

__all__ = ["BACKENDS", "ConversionResult", "ConverterError", "iupac_to_smiles"]
__version__ = "0.1.0"


def iupac_to_smiles(
    source: str | Path | list[str],
    *,
    column: str | None = None,
    backends: list[str] | None = None,
    converter_params: dict[str, dict[str, Any]] | None = None,
    validate_smiles: bool = False,
    return_details: bool = False,
) -> list[str] | list[ConversionResult]:
    """Convert IUPAC names to SMILES.

    Args:
        source: list of names, or path to a CSV file.
        column: CSV column with names (default: first column).
        backends: subset/order of ``BACKENDS`` to use (default: all, in cascade order).
        converter_params: per-backend options, e.g. ``{"opsin": {"allow_radicals": True}}``.
        validate_smiles: reject SMILES invalid according to RDKit before fallback (requires [validate]).
        return_details: return ConversionResult objects with backend/status instead of strings.

    Returns:
        One SMILES per input name, in input order. ``""`` where no backend succeeded.

    >>> iupac_to_smiles(["ethanol", "2-acetyloxybenzoic acid"])  # doctest: +SKIP
    ['CCO', 'CC(=O)OC1=CC=CC=C1C(=O)O']
    """
    names = (
        [str(n).strip() if n is not None else "" for n in source]
        if isinstance(source, list)
        else read_names(source, column).tolist()
    )
    records = convert_detailed(
        names, backends=backends, converter_params=converter_params, validate_smiles=validate_smiles
    )
    return records if return_details else [record.smiles for record in records]
