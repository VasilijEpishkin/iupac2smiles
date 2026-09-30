"""Accuracy of each backend and of the full cascade against reference SMILES.

Input: IUPAC2Struct-style test file, one ``SMILES, IUPAC name`` pair per line after a header
(e.g. https://github.com/sergsb/IUPAC2Struct/tree/main/data/smiles_lens_1000).

    python scripts/benchmark.py 80.csv [-n 200]

A prediction counts as correct when its RDKit canonical SMILES equals the reference's
(stereochemistry included).
"""

from __future__ import annotations

import argparse
import time

from rdkit import Chem, RDLogger

from iupac2smiles import BACKENDS
from iupac2smiles.router import convert

RDLogger.DisableLog("rdApp.*")


def canon(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToSmiles(mol) if mol else None


def load(path: str) -> tuple[list[str], list[str]]:
    smiles, names = [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            if ", " in line:  # SMILES never contain ", "; names may contain commas
                s, n = line.strip().split(", ", 1)
                smiles.append(s)
                names.append(n)
    return smiles, names


def score(ref: list[str], pred: list[str]) -> tuple[float, float]:
    converted = sum(bool(p) for p in pred) / len(pred)
    correct = sum(canon(r) is not None and canon(r) == canon(p) for r, p in zip(ref, pred, strict=True)) / len(pred)
    return converted, correct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("-n", type=int, help="use only the first N pairs")
    args = ap.parse_args()

    ref, names = load(args.file)
    ref, names = ref[: args.n], names[: args.n]
    print(f"{len(names)} names from {args.file}\n")
    print(f"{'backend':<28}{'converted':>10}{'correct':>10}{'time, s':>10}")
    for label, chain in [*((b, [b]) for b in BACKENDS), ("cascade (" + " > ".join(BACKENDS) + ")", None)]:
        t = time.perf_counter()
        pred = convert(names, backends=chain)
        converted, correct = score(ref, pred)
        print(f"{label:<28}{converted:>10.1%}{correct:>10.1%}{time.perf_counter() - t:>10.0f}")


if __name__ == "__main__":
    main()
