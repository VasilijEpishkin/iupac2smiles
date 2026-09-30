"""IUPAC2Struct backend: Transformer from Krasnov et al. (https://github.com/sergsb/IUPAC2Struct).

Params (``converter_params["iupac2struct"]``):
    beam  beam width (default 3)

Env: ``IUPAC2STRUCT_MODEL_PATH`` - use a local ``iupac2smiles_model.pt`` instead of downloading.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import threading

from ._download import fetch

logger = logging.getLogger(__name__)

MODEL_URL = "https://media.githubusercontent.com/media/sergsb/IUPAC2Struct/main/models/iupac2smiles_model.pt"
MODEL_SHA256 = "28c2c0a2dbe5d8870fc3e392bd01eec2db7a65f99b90fb08a200bf8ece158dae"

_lock = threading.Lock()


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@functools.lru_cache(maxsize=1)
def _model():
    import torch

    from ._vendor.iupac2struct import iupac_model, model, smiles_model, transformer

    # The checkpoint is a full pickle that references these top-level module names.
    for mod in (iupac_model, model, smiles_model, transformer):
        sys.modules.setdefault(mod.__name__.rsplit(".", 1)[-1], mod)

    path = os.environ.get("IUPAC2STRUCT_MODEL_PATH") or fetch(MODEL_URL, MODEL_SHA256)
    device = _device()
    logger.info("Loading IUPAC2Struct from %s on %s", path, device)
    # weights_only=False is required for this legacy pickle; the file is pinned by sha256 above.
    m = torch.load(path, map_location=device, weights_only=False)
    if isinstance(m.T, model.DataParallel):  # saved wrapped in DataParallel(...).cuda(): breaks on CPU/MPS
        m.T = m.T.model.module
    m.T.to(device).eval()
    m.device = device
    return m


def iupac_to_smiles(names: list[str], *, params: dict | None = None) -> list[str]:
    if not names:
        return []
    beam = (params or {}).get("beam", 3)
    m = _model()
    out: list[str] = []
    with _lock:
        for name in names:
            try:
                smiles, _probs = m.predict_single(name, beam=beam)
                out.append(smiles[0] or "" if smiles else "")
            except Exception as exc:  # untokenizable name, unknown token, ...
                logger.debug("iupac2struct failed on %r: %s", name, exc)
                out.append("")
    return out
