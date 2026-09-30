"""ChemConv backend: Knowledgator's seq2seq model via ``chemical-converters``.

Params (``converter_params["chemconv"]``):
    num_beams        beam width (default 1)
    process_in_batch batch inference (default True)
    batch_size       batch size (default 32)
"""

from __future__ import annotations

import functools
import logging
import re
import threading

logger = logging.getLogger(__name__)

MODEL_NAME = "knowledgator/IUPAC2SMILES-canonical-base"
_TAG_RE = re.compile(r"<[A-Z]+>")  # model prefixes output with <SYST>, <BASE>, <TRAD>
_lock = threading.Lock()


@functools.lru_cache(maxsize=1)
def _converter():
    from chemicalconverters import NamesConverter

    logger.info("Loading ChemConv model %s", MODEL_NAME)
    return NamesConverter(model_name=MODEL_NAME)


def _clean(raw) -> str:
    if not raw:
        return ""
    text = raw[0] if isinstance(raw, (list, tuple)) else str(raw)
    return _TAG_RE.sub("", text).strip()


def iupac_to_smiles(names: list[str], *, params: dict | None = None) -> list[str]:
    if not names:
        return []
    p = params or {}
    with _lock:
        raw = _converter().iupac_to_smiles(
            names,
            num_beams=p.get("num_beams", 1),
            process_in_batch=p.get("process_in_batch", True),
            batch_size=p.get("batch_size", 32),
        )
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    if len(raw) != len(names):
        raise RuntimeError(f"ChemConv returned {len(raw)} results for {len(names)} names")
    return [_clean(r) for r in raw]
