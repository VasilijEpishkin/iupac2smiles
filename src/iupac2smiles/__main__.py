"""CLI: iupac2smiles names.csv -o out.csv  |  iupac2smiles --name "ethanol" """

from __future__ import annotations

import argparse
import logging
import sys

from . import BACKENDS, iupac_to_smiles
from .io import read_names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="iupac2smiles", description=__doc__)
    ap.add_argument("csv", nargs="?", help="input CSV with IUPAC names")
    ap.add_argument("-n", "--name", action="append", help="convert a name directly (repeatable)")
    ap.add_argument("-c", "--column", help="CSV column with names (default: first)")
    ap.add_argument("-o", "--output", help="output CSV (default: stdout)")
    ap.add_argument(
        "-b",
        "--backends",
        default=",".join(BACKENDS),
        help=f"comma-separated cascade order (default: {','.join(BACKENDS)})",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if not args.csv and not args.name:
        ap.error("give a CSV file or --name")

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    names = args.name or read_names(args.csv, args.column).tolist()
    smiles = iupac_to_smiles(names, backends=backends)

    import pandas as pd

    df = pd.DataFrame({"iupac": names, "smiles": smiles})
    df.to_csv(args.output or sys.stdout, index=False)
    if args.output:
        print(f"{(df.smiles != '').sum()}/{len(df)} converted -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
