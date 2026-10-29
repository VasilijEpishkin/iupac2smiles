"""Offline regression checks for artifact-cache integrity."""

import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from iupac2smiles import _download

URL = "https://example.org/weights.bin"
GOOD = b"verified model bytes"
SHA = hashlib.sha256(GOOD).hexdigest()


def test_reuse_only_verified_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(_download, "CACHE_DIR", tmp_path)
    cached = tmp_path / "weights.bin"
    cached.write_bytes(GOOD)

    def forbidden(*args):
        raise AssertionError("No download should occur for a verified file")

    monkeypatch.setattr(_download.urllib.request, "urlretrieve", forbidden)
    assert _download.fetch(URL, SHA) == cached
    assert cached.read_bytes() == GOOD


def test_corrupt_cache_is_replaced_with_verified_download(monkeypatch, tmp_path):
    monkeypatch.setattr(_download, "CACHE_DIR", tmp_path)
    cached = tmp_path / "weights.bin"
    cached.write_bytes(b"damaged")

    def retrieve(url, target):
        assert url == URL
        target.write_bytes(GOOD)

    monkeypatch.setattr(_download.urllib.request, "urlretrieve", retrieve)
    assert _download.fetch(URL, SHA) == cached
    assert cached.read_bytes() == GOOD
    assert not list(tmp_path.glob("*.part"))


def test_bad_download_is_rejected_and_temp_removed(monkeypatch, tmp_path):
    monkeypatch.setattr(_download, "CACHE_DIR", tmp_path)

    def retrieve(url, target):
        target.write_bytes(b"wrong")

    monkeypatch.setattr(_download.urllib.request, "urlretrieve", retrieve)
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        _download.fetch(URL, SHA)
    assert not (tmp_path / "weights.bin").exists()
    assert not list(tmp_path.glob("*.part"))


def test_failed_download_does_not_publish_partial_file(monkeypatch, tmp_path):
    monkeypatch.setattr(_download, "CACHE_DIR", tmp_path)

    def retrieve(url, target):
        target.write_bytes(b"partial")
        raise OSError("connection interrupted")

    monkeypatch.setattr(_download.urllib.request, "urlretrieve", retrieve)
    with pytest.raises(OSError, match="interrupted"):
        _download.fetch(URL, SHA)
    assert not (tmp_path / "weights.bin").exists()
    assert not list(tmp_path.glob("*.part"))


def test_concurrent_cache_population_is_safe(monkeypatch, tmp_path):
    monkeypatch.setattr(_download, "CACHE_DIR", tmp_path)

    def retrieve(url, target):
        target.write_bytes(GOOD)

    monkeypatch.setattr(_download.urllib.request, "urlretrieve", retrieve)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: _download.fetch(URL, SHA), range(4)))
    assert len(set(paths)) == 1
    assert paths[0].read_bytes() == GOOD
    assert not list(tmp_path.glob("*.part"))
