"""Regression tests for opt-in SMILES validation and per-row provenance."""

import types

import pytest

from iupac2smiles import iupac_to_smiles, router
from iupac2smiles.__main__ import main


def backend(outputs: dict[str, str]):
    def iupac_to_smiles(names, *, params=None):
        return [outputs.get(name, "") for name in names]

    return types.SimpleNamespace(iupac_to_smiles=iupac_to_smiles)


def test_details_keep_row_order_and_provenance(monkeypatch):
    monkeypatch.setattr(
        router,
        "BACKENDS",
        {"rules": backend({"ethanol": "CCO"}), "model": backend({"benzene": "c1ccccc1"})},
    )
    records = iupac_to_smiles(["benzene", "", "ethanol", "missing"], return_details=True)
    assert [(r.smiles, r.backend, r.status) for r in records] == [
        ("c1ccccc1", "model", "converted"),
        ("", None, "empty_input"),
        ("CCO", "rules", "converted"),
        ("", None, "unconverted"),
    ]
    assert iupac_to_smiles(["benzene", "ethanol"]) == ["c1ccccc1", "CCO"]


def test_invalid_smiles_is_rejected_and_falls_back(monkeypatch):
    monkeypatch.setattr(
        router,
        "BACKENDS",
        {
            "rules": backend({"ethanol": "C1(", "benzene": "c1ccccc1", "bad": "C1("}),
            "model": backend({"ethanol": "CCO", "bad": "C[N](C)(C)C"}),
        },
    )
    records = iupac_to_smiles(["ethanol", "benzene", "bad"], validate_smiles=True, return_details=True)
    assert [(r.smiles, r.backend, r.status) for r in records] == [
        ("CCO", "model", "converted"),
        ("c1ccccc1", "rules", "converted"),
        ("", None, "invalid_smiles"),
    ]
    assert iupac_to_smiles(["ethanol"], validate_smiles=False) == ["C1("]


def test_cli_detailed_validation_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(router, "BACKENDS", {"rules": backend({"ethanol": "CCO", "invalid": "C1("})})
    out = tmp_path / "result.csv"
    args = ["-n", "ethanol", "-n", "invalid", "-b", "rules", "--validate-smiles", "--details", "-o", str(out)]
    assert main(args) == 0
    assert out.read_text().splitlines() == [
        "iupac,smiles,backend,status",
        "ethanol,CCO,rules,converted",
        "invalid,,,invalid_smiles",
    ]


def test_empty_backend_list_is_not_silently_replaced(monkeypatch):
    monkeypatch.setattr(router, "BACKENDS", {"rules": backend({"ethanol": "CCO"})})
    with pytest.raises(ValueError, match="At least one backend"):
        iupac_to_smiles(["ethanol"], backends=[])
