"""CSV → Parquet conversion and dtype-optimised loaders (pandas-only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# Columns that should be (small) signed integers.
_INT8 = {
    "prop_starrating",
    "prop_brand_bool",
    "promotion_flag",
    "srch_saturday_night_bool",
    "random_bool",
    "click_bool",
    "booking_bool",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "srch_length_of_stay",
}
_INT16 = {"site_id", "position"}
_INT32 = {
    "srch_id",
    "visitor_location_country_id",
    "prop_country_id",
    "prop_id",
    "srch_destination_id",
    "srch_booking_window",
}
_FLOAT32 = {
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "price_usd",
    "srch_query_affinity_score",
    "orig_destination_distance",
    "gross_bookings_usd",
}
_COMP_INT = {f"comp{i}_rate" for i in C.COMP_INDICES} | {
    f"comp{i}_inv" for i in C.COMP_INDICES
}
_COMP_FLOAT = {f"comp{i}_rate_percent_diff" for i in C.COMP_INDICES}


def _pandas_dtypes(columns: list[str]) -> dict[str, str]:
    """Map column name → pandas read_csv dtype. Use nullable dtypes for ints
    that may contain NULL (comp{i}_rate, comp{i}_inv); use plain numpy dtypes
    everywhere else."""
    out: dict[str, str] = {}
    for col in columns:
        if col == "date_time":
            continue
        if col in _INT8:
            # Some Int8 columns (e.g. prop_starrating) can be NULL; use nullable.
            out[col] = "Int8"
        elif col in _INT16:
            out[col] = "Int16"
        elif col in _INT32:
            out[col] = "Int32"
        elif col in _FLOAT32 or col in _COMP_FLOAT:
            out[col] = "float32"
        elif col in _COMP_INT:
            out[col] = "Int8"
    return out


def csv_to_parquet(csv_path: Path, parquet_path: Path) -> int:
    """Read CSV with pandas, parse `date_time`, sort by `srch_id`, write Parquet.

    Returns the number of rows written.
    """
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Peek the header for dtype mapping.
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    dtypes = _pandas_dtypes(header)

    df = pd.read_csv(
        csv_path,
        dtype=dtypes,
        na_values=["NULL"],
        keep_default_na=True,
        parse_dates=["date_time"],
    )
    df = df.sort_values("srch_id", kind="stable").reset_index(drop=True)
    df.to_parquet(parquet_path, compression="zstd", index=False)
    return len(df)


def load_labeled() -> pd.DataFrame:
    """Load the labeled training data (was Data/test.csv)."""
    return pd.read_parquet(C.LABELED_PARQUET)


def load_submit() -> pd.DataFrame:
    """Load the unlabeled submission set (was Data/train.csv)."""
    return pd.read_parquet(C.SUBMIT_PARQUET)
