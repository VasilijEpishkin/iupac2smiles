# iupac2smiles

[![CI](https://github.com/VasilijEpishkin/iupac2smiles/actions/workflows/ci.yml/badge.svg)](https://github.com/VasilijEpishkin/iupac2smiles/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Convert IUPAC chemical names to SMILES using a **fallback cascade** of one rule-based parser
and two neural translators. Each backend only receives the names the previous ones could not handle,
so you get the precision of a rule-based parser plus the coverage of ML models.

```
names ──► OPSIN ──failed──► IUPAC2Struct ──failed──► ChemConv ──► SMILES ("" if nothing worked)
         (rules, Java)     (Transformer)            (seq2seq, HF)
```

| Backend | Type | Notes |
|---|---|---|
| [OPSIN](https://github.com/dan2097/opsin) | Rule-based grammar | High-precision when it parses; returns nothing for names it doesn't understand |
| [IUPAC2Struct](https://github.com/sergsb/IUPAC2Struct) | Transformer (Krasnov et al., 2021) | Fails only when a name can't be tokenized |
| [ChemConv](https://huggingface.co/knowledgator/IUPAC2SMILES-canonical-base) | Seq2seq (Knowledgator) | Always produces *some* SMILES, so it runs last as a catch-all |

## Installation

```bash
git clone https://github.com/VasilijEpishkin/iupac2smiles.git
cd iupac2smiles
pip install -e ".[ml]"      # without [ml] only OPSIN is available
```

Requirements: Python ≥ 3.10, Java ≥ 8 (for OPSIN).

Model files are **not** stored in the repo. On first use they are downloaded once into
`~/.cache/iupac2smiles` and verified by SHA-256:

- OPSIN CLI jar 2.9.0 — 14 MB
- IUPAC2Struct weights — 200 MB
- ChemConv — pulled from Hugging Face Hub by `chemical-converters`

## Usage

### Python

```python
from iupac2smiles import iupac_to_smiles

iupac_to_smiles(["ethanol", "2-acetyloxybenzoic acid", "not a molecule"])
# Example only; ML models can return an incorrect but well-formed structure

# From CSV, only some backends, with backend options
iupac_to_smiles(
    "names.csv",
    column="iupac",
    backends=["opsin", "iupac2struct"],
    converter_params={"opsin": {"allow_radicals": True}, "iupac2struct": {"beam": 5}},
)
```

The output is always aligned with the input; an empty string means no backend converted that name.

### CLI

```bash
iupac2smiles -n "ethanol" -n "benzoic acid"
iupac2smiles names.csv -c iupac -o smiles.csv -v
iupac2smiles names.csv -b opsin          # rule-based only, no ML dependencies
```

### Backend options

| Backend | Option | Default |
|---|---|---|
| `opsin` | `allow_radicals`, `wildcard_radicals`, `allow_acids_without_acid`, `allow_uninterpretable_stereo`, `detailed_failure_analysis` | all `False` |
| `iupac2struct` | `beam` | `3` |
| `chemconv` | `num_beams`, `batch_size`, `process_in_batch` | `1`, `32`, `True` |

### Environment variables

| Variable | Purpose |
|---|---|
| `IUPAC2SMILES_CACHE` | Cache directory for downloaded models |
| `OPSIN_JAR_PATH` | Use a local OPSIN jar |
| `IUPAC2STRUCT_MODEL_PATH` | Use a local `iupac2smiles_model.pt` |

### Verification and conversion provenance

Pass `--validate-smiles` to reject outputs that RDKit cannot parse/sanitize. An invalid candidate
is sent to the next backend; a row may still be unconverted after all backends.
This is **structural validity checking, not IUPAC-name-to-structure correctness verification**:
a chemically valid SMILES produced by a neural model can still describe the wrong molecule.

```bash
pip install -e ".[validate]"
iupac2smiles names.csv --validate-smiles --details -o smiles.csv
```

`--details` adds `backend` (the producing parser/model) and `status`
(`converted`, `unconverted`, `invalid_smiles`, `empty_input`) columns.
The original two-column CLI output remains the default. Python callers can opt in:

```python
rows = iupac_to_smiles(["ethanol", "unrecognized name"], validate_smiles=True, return_details=True)
for row in rows:
    print(row.name, row.smiles, row.backend, row.status)
```

RDKit is optional for ordinary conversion. Installing `[ml]` is still optional for OPSIN-only operation.

## Benchmark

`scripts/benchmark.py` measures, for each backend and for the full cascade, the share of names
converted and the share whose RDKit canonical SMILES matches the reference (stereo included):

```bash
pip install -e ".[ml,bench]"
curl -LO https://raw.githubusercontent.com/sergsb/IUPAC2Struct/main/data/smiles_lens_1000/80.csv
python scripts/benchmark.py 80.csv -n 200
```

## Development

```bash
pip install -e ".[dev]"
pytest          # cascade logic is tested with fake backends; OPSIN test runs if Java is installed
ruff check . && ruff format --check .
```

## Project layout

```
src/iupac2smiles/
├── __init__.py        # iupac_to_smiles() public API
├── __main__.py        # CLI
├── router.py          # fallback cascade
├── opsin.py           # OPSIN backend (Java subprocess)
├── iupac2struct.py    # IUPAC2Struct backend (PyTorch)
├── chemconv.py        # ChemConv backend (chemical-converters)
├── _download.py       # cached, checksum-verified model downloads
└── _vendor/iupac2struct/  # model code from sergsb/IUPAC2Struct (MIT), needed to unpickle weights
```

## Credits

- [OPSIN](https://github.com/dan2097/opsin) — D. Lowe et al., MIT License
- [IUPAC2Struct](https://github.com/sergsb/IUPAC2Struct) — L. Krasnov, I. Khokhlov, M. Fedorov, S. Sosnin,
  *Transformer-based artificial neural networks for the conversion between chemical notations*,
  Sci. Rep. 11, 14798 (2021). MIT License. Model code vendored in `src/iupac2smiles/_vendor/iupac2struct`.
- [chemical-converters](https://github.com/Knowledgator/chemical-converters) — Knowledgator

## License

MIT — see [LICENSE](LICENSE).
