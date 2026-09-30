"""Real OPSIN run (downloads the 14 MB jar once). Skipped without Java."""

import shutil

import pytest

from iupac2smiles import opsin

pytestmark = pytest.mark.skipif(shutil.which("java") is None, reason="Java not installed")


def test_opsin_keeps_alignment_on_failures():
    out = opsin.iupac_to_smiles(["ethanol", "definitely not a name", "benzoic\nacid"])
    assert out[0] == "C(C)O"
    assert out[1] == ""
    assert out[2] == "C(C1=CC=CC=C1)(=O)O"
