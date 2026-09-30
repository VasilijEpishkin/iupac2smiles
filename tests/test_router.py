import types

import pytest

from iupac2smiles import iupac_to_smiles, router
from iupac2smiles.__main__ import main


def fake(known: dict[str, str], calls: list):
    def iupac_to_smiles(names, *, params=None):
        calls.append(list(names))
        return [known.get(n, "") for n in names]

    return types.SimpleNamespace(iupac_to_smiles=iupac_to_smiles)


def broken():
    def iupac_to_smiles(names, *, params=None):
        raise RuntimeError("java not found")

    return types.SimpleNamespace(iupac_to_smiles=iupac_to_smiles)


@pytest.fixture
def backends(monkeypatch):
    calls = {"a": [], "b": []}
    fakes = {
        "a": fake({"ethanol": "CCO"}, calls["a"]),
        "b": fake({"benzene": "c1ccccc1", "ethanol": "WRONG"}, calls["b"]),
    }
    monkeypatch.setattr(router, "BACKENDS", fakes)
    return calls


def test_cascade_only_passes_failures_forward_and_keeps_order(backends):
    out = iupac_to_smiles(["benzene", "", "ethanol", "unknownium"])
    assert out == ["c1ccccc1", "", "CCO", ""]
    assert backends["a"] == [["benzene", "ethanol", "unknownium"]]  # empty name never sent
    assert backends["b"] == [["benzene", "unknownium"]]  # ethanol already solved by "a"


def test_crashed_backend_is_skipped(monkeypatch, backends):
    monkeypatch.setitem(router.BACKENDS, "a", broken())
    assert iupac_to_smiles(["ethanol", "benzene"]) == ["WRONG", "c1ccccc1"]


def test_all_backends_crashed(monkeypatch):
    monkeypatch.setattr(router, "BACKENDS", {"a": broken(), "b": broken()})
    with pytest.raises(router.ConverterError) as e:
        iupac_to_smiles(["ethanol"])
    assert [name for name, _ in e.value.errors] == ["a", "b"]


def test_unknown_backend(backends):
    with pytest.raises(ValueError):
        iupac_to_smiles(["ethanol"], backends=["nope"])


def test_csv_and_cli(tmp_path, backends, capsys):
    src = tmp_path / "in.csv"
    src.write_text('id,name\n1,ethanol\n2,\n3,"1,2-dichloro-benzene"\n4,benzene\n')
    assert iupac_to_smiles(src, column="name") == ["CCO", "", "", "c1ccccc1"]

    out = tmp_path / "out.csv"
    assert main([str(src), "-c", "name", "-o", str(out), "-b", "a,b"]) == 0
    assert out.read_text().splitlines()[1:] == ["ethanol,CCO", ",", '"1,2-dichloro-benzene",', "benzene,c1ccccc1"]
