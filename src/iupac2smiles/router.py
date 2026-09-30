"""Fallback cascade: OPSIN -> IUPAC2Struct -> ChemConv.

Each backend gets only the names the previous ones failed on.
A backend is any module with ``iupac_to_smiles(names, *, params) -> list[str]``
returning one SMILES per name ("" = not converted).
"""

from __future__ import annotations

import logging
from typing import Any

from . import chemconv, iupac2struct, opsin

logger = logging.getLogger(__name__)

# Order = priority. Rule-based OPSIN first: fast and exact when it parses.
# ChemConv goes last: it returns *some* SMILES for any input (even gibberish),
# so nothing after it would ever be tried.
BACKENDS: dict[str, Any] = {
    "opsin": opsin,
    "iupac2struct": iupac2struct,
    "chemconv": chemconv,
}


class ConverterError(Exception):
    """Raised when every backend in the cascade crashed."""

    def __init__(self, message: str, errors: list[tuple[str, Exception]]):
        super().__init__(message)
        self.errors = errors


def convert(
    names: list[str],
    *,
    backends: list[str] | None = None,
    converter_params: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Convert names to SMILES. Output is aligned with input; "" = not converted."""
    chain = backends or list(BACKENDS)
    unknown = set(chain) - set(BACKENDS)
    if unknown:
        raise ValueError(f"Unknown backends: {sorted(unknown)}. Available: {list(BACKENDS)}")

    params = converter_params or {}
    final = [""] * len(names)
    pending = [i for i, n in enumerate(names) if n and n.strip()]
    errors: list[tuple[str, Exception]] = []

    for name in chain:
        if not pending:
            break
        try:
            result = BACKENDS[name].iupac_to_smiles([names[i] for i in pending], params=params.get(name))
            if len(result) != len(pending):
                raise RuntimeError(f"returned {len(result)} results for {len(pending)} names")
        except Exception as exc:  # backend unavailable (no Java, no torch, ...) -> try the next one
            logger.warning("%s failed: %s", name, exc)
            errors.append((name, exc))
            continue

        still: list[int] = []
        for i, smi in zip(pending, result, strict=True):
            if smi:
                final[i] = smi
            else:
                still.append(i)
        logger.info("%s: converted %d, remaining %d", name, len(pending) - len(still), len(still))
        pending = still

    if len(errors) == len(chain):
        raise ConverterError("All backends failed", errors)
    return final
