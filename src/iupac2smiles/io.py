"""CSV input helper."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def read_names(path: str | Path, column: str | None = None) -> pd.Series:
    """Read the name column from a CSV (first column if ``column`` is None). Row order is kept."""
    df = pd.read_csv(path, sep=None, engine="python")
    if column is None:
        column = df.columns[0]
        logger.info("No column given, using %r", column)
    elif column not in df.columns:
        raise ValueError(f"Column {column!r} not found. Available: {list(df.columns)}")
    return df[column].fillna("").astype(str).str.strip()
