"""Download and validate pinned third-party artifacts in a local cache."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("IUPAC2SMILES_CACHE", Path.home() / ".cache" / "iupac2smiles"))


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, sha256: str) -> Path:
    """Return a checksum-verified artifact; never trust cached bytes by filename alone.

    A unique temporary file and atomic replace keep partial downloads out of the
    cache even if several processes request the same artifact concurrently.
    Concurrent initial downloads may perform redundant network work.
    """
    path = CACHE_DIR / url.rsplit("/", 1)[-1]
    if path.is_file():
        if _digest(path) == sha256:
            return path
        logger.warning("Cache checksum mismatch for %s; downloading a verified replacement", path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".part", dir=CACHE_DIR, delete=False) as f:
        tmp = Path(f.name)
    try:
        logger.info("Downloading %s -> %s", url, path)
        urllib.request.urlretrieve(url, tmp)
        got = _digest(tmp)
        if got != sha256:
            raise RuntimeError(f"Checksum mismatch for {url}: got {got}, expected {sha256}")
        tmp.replace(path)
        return path
    finally:
        tmp.unlink(missing_ok=True)
