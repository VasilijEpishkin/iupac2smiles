"""Download third-party artifacts (OPSIN jar, IUPAC2Struct weights) into a local cache."""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("IUPAC2SMILES_CACHE", Path.home() / ".cache" / "iupac2smiles"))


def fetch(url: str, sha256: str) -> Path:
    """Return a cached copy of ``url``, downloading and verifying it on first use."""
    path = CACHE_DIR / url.rsplit("/", 1)[-1]
    if path.is_file():
        return path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    logger.warning("Downloading %s -> %s (one-time)", url, path)
    urllib.request.urlretrieve(url, tmp)

    h = hashlib.sha256()
    with tmp.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != sha256:
        tmp.unlink()
        raise RuntimeError(f"Checksum mismatch for {url}: got {h.hexdigest()}, expected {sha256}")
    tmp.replace(path)
    return path
