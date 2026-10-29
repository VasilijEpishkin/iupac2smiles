"""Offline unit tests for benchmark parser and RDKit metrics."""

import pytest

pytest.importorskip("rdkit")

from scripts.benchmark import load, score


def test_loader_keeps_commas_in_chemical_names(tmp_path):
    p = tmp_path / "reference.csv"
    p.write_text("input, target\nCCO, ethanol\nCC(=O)O, 2,3-test chemical, complex name\n")
    refs, names = load(p)
    assert refs == ["CCO", "CC(=O)O"]
    assert names == ["ethanol", "2,3-test chemical, complex name"]


def test_loader_rejects_malformed_lines(tmp_path):
    p = tmp_path / "reference.csv"
    p.write_text("input, target\nCCO, ethanol\nbad line\n")
    with pytest.raises(ValueError, match="Line 3"):
        load(p)


def test_stereo_vs_connectivity_and_invalid_outputs():
    refs = ["CCO", "F[C@H](Cl)Br", "CCO"]
    preds = ["CCO", "F[C@@H](Cl)Br", "C1("]
    result = score(refs, preds)
    assert result["returned"] == 3
    assert result["valid"] == 2
    assert result["invalid"] == 1
    assert result["exact_stereo"] == 1
    assert result["exact_no_stereo"] == 2
    assert result["accuracy"] == pytest.approx(1 / 3, abs=1e-6)


def test_missing_predictions_not_counted_as_valid():
    result = score(["CCO"], [""])
    assert result["returned"] == 0
    assert result["precision_given_output"] == 0.0


def test_invalid_reference_is_reported():
    with pytest.raises(ValueError, match="Invalid reference"):
        score(["C1("], ["CCO"])
