"""Fallback cascade with optional SMILES syntax/valence validation and provenance.

Each backend receives only names the preceding backends could not convert.
A syntactically valid SMILES does not prove the structure matches its IUPAC name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from . import chemconv, iupac2struct, opsin

logger = logging.getLogger(__name__)

BACKENDS: dict[str, Any] = {
    "opsin": opsin,
    "iupac2struct": iupac2struct,
    "chemconv": chemconv,
}


@dataclass(frozen=True)
class ConversionResult:
    """One result, including the backend responsible and its outcome."""

    name: str
    smiles: str
    backend: str | None
    status: str  # converted, unconverted, invalid_smiles, empty_input


class ConverterError(Exception):
    """Raised when every backend in the cascade crashed."""

    def __init__(self, message: str, errors: list[tuple[str, Exception]]):
        super().__init__(message)
        self.errors = errors


def _smiles_validator() -> Callable[[str], bool]:
    try:
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise RuntimeError('SMILES validation requires RDKit. Install with pip install "iupac2smiles[validate]".') from exc

    def is_valid(smiles: str) -> bool:
        try:
            with rdBase.BlockLogs():
                mol = Chem.MolFromSmiles(smiles)
            return mol is not None and mol.GetNumAtoms() > 0
        except (TypeError, ValueError, RuntimeError):
            return False

    return is_valid


def convert_detailed(
    names: list[str],
    *,
    backends: list[str] | None = None,
    converter_params: dict[str, dict[str, Any]] | None = None,
    validate_smiles: bool = False,
) -> list[ConversionResult]:
    """Convert names and report backend/status per row.

    When validate_smiles is true, invalid RDKit structures are rejected and
    passed to the next backend. Validation checks SMILES validity, not whether
    its molecular structure corresponds to the input chemical name.
    """
    chain = list(BACKENDS) if backends is None else list(backends)
    if not chain:
        raise ValueError("At least one backend is required")
    unknown = set(chain) - set(BACKENDS)
    if unknown:
        raise ValueError(f"Unknown backends: {sorted(unknown)}. Available: {list(BACKENDS)}")
    if len(chain) != len(set(chain)):
        raise ValueError("Duplicate backends are not allowed")

    params = converter_params or {}
    records = [
        ConversionResult(n, "", None, "unconverted" if n.strip() else "empty_input")
        for n in names
    ]
    pending = [i for i, n in enumerate(names) if n.strip()]
    is_valid = _smiles_validator() if validate_smiles and pending else None
    errors: list[tuple[str, Exception]] = []

    for backend in chain:
        if not pending:
            break
        try:
            result = BACKENDS[backend].iupac_to_smiles([names[i] for i in pending], params=params.get(backend))
            if len(result) != len(pending):
                raise RuntimeError(f"returned {len(result)} results for {len(pending)} names")
            if any(not isinstance(smi, str) for smi in result):
                raise TypeError("backend returned a non-string SMILES value")
        except Exception as exc:
            logger.warning("%s failed: %s", backend, exc)
            errors.append((backend, exc))
            continue

        remaining: list[int] = []
        for i, smiles in zip(pending, result, strict=True):
            smiles = smiles.strip()
            if not smiles:
                remaining.append(i)
            elif is_valid is not None and not is_valid(smiles):
                records[i] = ConversionResult(names[i], "", None, "invalid_smiles")
                remaining.append(i)
            else:
                records[i] = ConversionResult(names[i], smiles, backend, "converted")

        logger.info("%s: converted %d, remaining %d", backend, len(pending) - len(remaining), len(remaining))
        pending = remaining

    if len(errors) == len(chain):
        raise ConverterError("All backends failed", errors)
    return records


def convert(
    names: list[str],
    *,
    backends: list[str] | None = None,
    converter_params: dict[str, dict[str, Any]] | None = None,
    validate_smiles: bool = False,
) -> list[str]:
    """Legacy string-only result, aligned with input (empty string means no conversion)."""
    return [
        item.smiles
        for item in convert_detailed(
            names,
            backends=backends,
            converter_params=converter_params,
            validate_smiles=validate_smiles,
        )
    ]
