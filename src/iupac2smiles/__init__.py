"""IUPAC name -> SMILES with a fallback cascade: OPSIN -> IUPAC2Struct -> ChemConv."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_names
from .router import BACKENDS, ConverterError, convert

__all__ = ["BACKENDS", "ConverterError", "iupac_to_smiles"]
__version__ = "0.1.0"


def iupac_to_smiles(
    source: str | Path | list[str],
    *,
    column: str | None = None,
    backends: list[str] | None = None,
    converter_params: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Convert IUPAC names to SMILES.

    Args:
        source: list of names, or path to a CSV file.
        column: CSV column with names (default: first column).
        backends: subset/order of ``BACKENDS`` to use (default: all, in cascade order).
        converter_params: per-backend options, e.g. ``{"opsin": {"allow_radicals": True}}``.

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
    return convert(names, backends=backends, converter_params=converter_params)
