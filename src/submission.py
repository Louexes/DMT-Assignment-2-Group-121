"""Submission file generation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as C


def write_submission(
    df: pd.DataFrame,
    rank_col: str,
    path: Path,
) -> Path:
    """Write a Kaggle submission CSV.

    `df` must have columns srch_id, prop_id, and `rank_col` (lower == better).
    Output is sorted by srch_id then rank_col asc, with header `srch_id,prop_id`
    to match `Data/submission_sample.csv` exactly.
    """
    df = df.sort_values(["srch_id", rank_col], kind="stable").reset_index(drop=True)
    df[["srch_id", "prop_id"]].to_csv(path, index=False)
    return path


def verify_against_sample(submission_path: Path) -> dict:
    sub = pd.read_csv(submission_path)
    sample = pd.read_csv(C.SUBMISSION_SAMPLE)
    out = {
        "rows_submission": len(sub),
        "rows_sample": len(sample),
        "queries_match": (
            sorted(sub["srch_id"].unique().tolist())
            == sorted(sample["srch_id"].unique().tolist())
        ),
        "rows_match": len(sub) == len(sample),
    }
    # Per-query row count check.
    sub_counts = sub.groupby("srch_id").size()
    samp_counts = sample.groupby("srch_id").size()
    out["per_query_counts_match"] = bool(
        (sub_counts.sort_index().values == samp_counts.sort_index().values).all()
    )
    return out
