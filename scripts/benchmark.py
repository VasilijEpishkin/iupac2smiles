"""Compare IUPAC-to-SMILES backends on a reference set.

Supports the IUPAC2Struct published evaluation format: a header followed by
"<reference SMILES>, <IUPAC name>" on every line (the name may contain commas).

Measures valid output, molecular identity (with/without stereo) and conditional
accuracy on names that OPSIN could not resolve. Requires RDKit.

    python scripts/benchmark.py 80.csv -n 200 --backends opsin
    python scripts/benchmark.py 80.csv --json bench.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from rdkit import Chem, rdBase

from iupac2smiles import BACKENDS
from iupac2smiles.router import convert


def canon(smiles: str, *, isomeric: bool = True) -> str | None:
    if not smiles:
        return None
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def load(path: str | Path) -> tuple[list[str], list[str]]:
    """Parse IUPAC2Struct's comma-space-delimited evaluation file without dropping rows."""
    smiles, names = [], []
    with Path(path).open(encoding="utf-8-sig") as stream:
        header = next(stream, "").strip().lower().replace(" ", "")
        if header != "input,target":
            raise ValueError("Expected IUPAC2Struct evaluation file with header 'input, target'")
        for lineno, line in enumerate(stream, 2):
            if not line.strip():
                continue
            if ", " not in line:
                raise ValueError(f"Line {lineno} lacks the expected ', ' separator")
            structure, name = line.rstrip("\r\n").split(", ", 1)
            if not structure.strip() or not name.strip():
                raise ValueError(f"Line {lineno} contains an empty reference or name")
            smiles.append(structure.strip())
            names.append(name.strip())
    if not names:
        raise ValueError("Empty benchmark dataset")
    return smiles, names


def score(reference: list[str], predicted: list[str]) -> dict[str, int | float]:
    if len(reference) != len(predicted) or not reference:
        raise ValueError("Reference and prediction arrays must be nonempty and aligned")
    n = len(predicted)
    converted = valid = exact_stereo = exact_connectivity = 0
    for gold, actual in zip(reference, predicted, strict=True):
        gold_stereo = canon(gold)
        if gold_stereo is None:
            raise ValueError(f"Invalid reference SMILES: {gold!r}")
        gold_plain = canon(gold, isomeric=False)
        if not actual:
            continue
        converted += 1
        actual_stereo = canon(actual)
        if actual_stereo is None:
            continue
        valid += 1
        exact_stereo += actual_stereo == gold_stereo
        exact_connectivity += canon(actual, isomeric=False) == gold_plain
    return {
        "total": n,
        "returned": converted,
        "valid": valid,
        "invalid": converted - valid,
        "exact_stereo": exact_stereo,
        "exact_no_stereo": exact_connectivity,
        "coverage": round(converted / n, 6),
        "valid_coverage": round(valid / n, 6),
        "accuracy": round(exact_stereo / n, 6),
        "precision_given_output": round(exact_stereo / converted, 6) if converted else 0.0,
    }


def benchmark(reference: list[str], names: list[str], selected: list[str]) -> list[dict]:
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(BACKENDS):
        raise ValueError(f"Invalid backends: {selected!r}; available: {list(BACKENDS)}")
    variants = [(backend, [backend]) for backend in selected]
    variants.append(("cascade", selected))
    rows: list[dict] = []
    outputs: dict[str, list[str]] = {}
    for label, chain in variants:
        start = time.perf_counter()
        predicted = convert(names, backends=chain)
        seconds = time.perf_counter() - start
        outputs[label] = predicted
        rows.append({"backend": label, "seconds": round(seconds, 3), **score(reference, predicted)})

    if "opsin" in outputs:
        unresolved = [i for i, answer in enumerate(outputs["opsin"]) if not answer]
        for row in rows:
            pred = outputs[row["backend"]]
            row["opsin_unresolved_total"] = len(unresolved)
            row["opsin_unresolved_correct"] = sum(
                canon(reference[i]) == canon(pred[i]) and canon(pred[i]) is not None for i in unresolved
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", type=Path, help="IUPAC2Struct-style reference file")
    ap.add_argument("-n", type=int, help="evaluate only the first N pairs")
    ap.add_argument("--backends", default=",".join(BACKENDS), help="comma-separated backends, in fallback order")
    ap.add_argument("--json", type=Path, help="write metrics and dataset SHA-256 to a JSON file")
    args = ap.parse_args()

    if args.n is not None and args.n < 1:
        ap.error("-n must be positive")
    selected = [value.strip() for value in args.backends.split(",") if value.strip()]
    reference, names = load(args.file)
    if args.n is not None:
        reference, names = reference[: args.n], names[: args.n]
    rows = benchmark(reference, names, selected)
    print(f"{len(names)} entries from {args.file}")
    print(f"{'backend':<16}{'output':>9}{'valid':>9}{'exact':>9}{'precision':>11}{'time (s)':>11}")
    for row in rows:
        print(
            f"{row['backend']:<16}{row['coverage']:>9.1%}{row['valid_coverage']:>9.1%}"
            f"{row['accuracy']:>9.1%}{row['precision_given_output']:>11.1%}{row['seconds']:>11.1f}"
        )

    if args.json:
        payload = {
            "source_file": str(args.file),
            "source_sha256": hashlib.sha256(args.file.read_bytes()).hexdigest(),
            "evaluated_rows": len(names),
            "selected_backends": selected,
            "note": "Accuracy requires canonical molecular identity including stereochemistry. "
            "This dataset may overlap model training data; results are not independent external validation.",
            "results": rows,
        }
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Metrics saved to {args.json}")


if __name__ == "__main__":
    main()
