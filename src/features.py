"""Feature engineering on pandas DataFrames.

Public entry points:
    make_split(df_labeled)          -> (train_qids, val_qids)
    compute_prop_history(df_train)  -> per-prop_id stats
    compute_target_encoding(df_train, group_col) -> (oof, full) k-fold TE tables
    compute_dest_history(df_train)  -> per-destination_id stats
    compute_dest_month_pos(df_train)-> expected position by (destination, month)
    compute_country_loc2(df_train)  -> per-country median prop_location_score2
    engineer(df, hist..., is_labeled) -> df with all engineered columns
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config as C


# ---------------------------------------------------------------------------
# Train / val split (group-aware, by srch_id)
# ---------------------------------------------------------------------------

def make_split(df_labeled: pd.DataFrame, val_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_qids, val_qids) — disjoint arrays of srch_id values."""
    qids = df_labeled["srch_id"].unique()
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=C.SEED)
    y = np.zeros(len(qids))
    train_pos, val_pos = next(splitter.split(qids, y, groups=qids))
    return qids[train_pos], qids[val_pos]


# ---------------------------------------------------------------------------
# Zero-as-missing repair
# ---------------------------------------------------------------------------

def fix_zero_as_missing(df: pd.DataFrame) -> pd.DataFrame:
    """`prop_starrating==0` means "0 stars / unknown / cannot publicize" and
    `prop_log_historical_price==0` means "not sold last period" — treat as NaN.

    `prop_review_score==0` is left as a real value because empirically the
    "no reviews" sentinel correlates with click rate.
    """
    df = df.copy()
    df["prop_starrating"] = df["prop_starrating"].astype("float32")
    df.loc[df["prop_starrating"] == 0, "prop_starrating"] = np.nan
    df["prop_log_historical_price"] = df["prop_log_historical_price"].astype("float32")
    df.loc[df["prop_log_historical_price"] == 0, "prop_log_historical_price"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Property history (computed on training fold only, joined onto everyone)
# ---------------------------------------------------------------------------

def compute_prop_history(df_train: pd.DataFrame, smooth: float = 50.0) -> pd.DataFrame:
    """Per-prop_id stats: smoothed CTR/BTR, position, price/star/review stats,
    plus `position_importance` (Besseling et al., harmonic-mean-of-1/rank
    with confidence scaling)."""
    global_ctr = float(df_train["click_bool"].mean())
    global_btr = float(df_train["booking_bool"].mean())
    global_pos = float(df_train["position"].mean())
    global_price = float(df_train["price_usd"].mean())

    g = df_train.groupby("prop_id", sort=False)
    out = pd.DataFrame({
        "prop_count": g.size().astype("int64"),
        "prop_click_sum": g["click_bool"].sum().astype("int64"),
        "prop_book_sum": g["booking_bool"].sum().astype("int64"),
        "prop_pos_sum": g["position"].sum().astype("float64"),
        "prop_pos_std": g["position"].std().fillna(0.0).astype("float32"),
        "prop_price_mean": g["price_usd"].mean().astype("float32"),
        "prop_price_std": g["price_usd"].std().fillna(0.0).astype("float32"),
        "prop_price_median": g["price_usd"].median().astype("float32"),
        "prop_price_q25": g["price_usd"].quantile(0.25).astype("float32"),
        "prop_price_q75": g["price_usd"].quantile(0.75).astype("float32"),
        "prop_price_skew": g["price_usd"].skew().fillna(0.0).astype("float32"),
        "prop_loghist_mean": g["prop_log_historical_price"].mean().astype("float32"),
        "prop_loghist_std": g["prop_log_historical_price"].std().fillna(0.0).astype("float32"),
        "prop_loghist_skew": g["prop_log_historical_price"].skew().fillna(0.0).astype("float32"),
        "prop_review_mean": g["prop_review_score"].mean().astype("float32"),
        "prop_review_std": g["prop_review_score"].std().fillna(0.0).astype("float32"),
    }).reset_index()

    out["prop_ctr"] = (
        (out["prop_click_sum"] + smooth * global_ctr) / (out["prop_count"] + smooth)
    ).astype("float32")
    out["prop_btr"] = (
        (out["prop_book_sum"] + smooth * global_btr) / (out["prop_count"] + smooth)
    ).astype("float32")
    out["prop_avg_position"] = (
        (out["prop_pos_sum"] + smooth * global_pos) / (out["prop_count"] + smooth)
    ).astype("float32")
    out["prop_price_iqr"] = (out["prop_price_q75"] - out["prop_price_q25"]).astype("float32")
    out["prop_price_mean"] = out["prop_price_mean"].fillna(global_price).astype("float32")
    out["prop_price_median"] = out["prop_price_median"].fillna(global_price).astype("float32")
    out["prop_count_log"] = np.log(out["prop_count"].astype("float32") + 1.0).astype("float32")

    # Position importance (Besseling et al., 2023) — HM(1/r_i) = 1/mean(r_i),
    # scaled by a confidence factor based on observation count. Sample paper's
    # #10 most-important feature.
    cnt = out["prop_count"].astype("float32")
    out["position_importance"] = (
        (1.0 / (out["prop_avg_position"] + 0.5)) * (cnt / (cnt + smooth))
    ).astype("float32")

    return out[[
        "prop_id",
        "prop_ctr",
        "prop_btr",
        "prop_avg_position",
        "prop_pos_std",
        "prop_price_mean",
        "prop_price_median",
        "prop_price_std",
        "prop_price_q25",
        "prop_price_q75",
        "prop_price_iqr",
        "prop_price_skew",
        "prop_loghist_mean",
        "prop_loghist_std",
        "prop_loghist_skew",
        "prop_review_mean",
        "prop_review_std",
        "prop_count_log",
        "position_importance",
    ]]


def _relevance_target(df: pd.DataFrame) -> np.ndarray:
    """5*book + 1*(click & not book) — the assignment relevance grade."""
    book = df["booking_bool"].to_numpy().astype("int8")
    click = df["click_bool"].to_numpy().astype("int8")
    return (book * 5 + ((click == 1) & (book == 0)).astype("int8")).astype("float32")


def compute_target_encoding(
    df_train: pd.DataFrame,
    group_col: str,
    n_folds: int = 5,
    smooth: float = 50.0,
    out_oof: str | None = None,
    out_full: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-fold out-of-fold target encoding for a high-cardinality categorical.

    Target = the assignment's relevance grade (5*book + click_only). Folds are
    assigned by `srch_id` so a single search never appears in both halves of a
    fold. Returns (oof_table, full_table):

    - `oof_table`  — one row per (srch_id, group_col) in df_train with the OOF
      mean. Used for *train* rows on the labeled set.
    - `full_table` — group_col → full-train target mean. Used for val rows and
      the unlabeled submit set.
    """
    if out_oof is None:
        out_oof = f"{group_col}_te_oof"
    if out_full is None:
        out_full = f"{group_col}_te_full"

    target = _relevance_target(df_train)
    global_mean = float(target.mean())

    qids = df_train["srch_id"].unique()
    rng = np.random.default_rng(C.SEED)
    perm = rng.permutation(len(qids))
    fold_assign = dict(zip(qids.tolist(), (perm % n_folds).tolist()))
    folds = df_train["srch_id"].map(fold_assign).astype("int8").to_numpy()

    df_local = pd.DataFrame({
        "srch_id": df_train["srch_id"].to_numpy(),
        group_col: df_train[group_col].to_numpy(),
        "_target": target,
        "_fold": folds,
    })

    out_frames: list[pd.DataFrame] = []
    for k in range(n_folds):
        train_part = df_local[df_local["_fold"] != k]
        agg = train_part.groupby(group_col, sort=False).agg(
            _t_sum=("_target", "sum"),
            _t_n=("_target", "size"),
        )
        agg[out_oof] = (
            (agg["_t_sum"] + smooth * global_mean) / (agg["_t_n"] + smooth)
        ).astype("float32")
        agg = agg[[out_oof]].reset_index()
        fold_rows = df_local.loc[df_local["_fold"] == k, ["srch_id", group_col]].drop_duplicates()
        fold_rows = fold_rows.merge(agg, on=group_col, how="left")
        out_frames.append(fold_rows)

    oof_table = pd.concat(out_frames, axis=0, ignore_index=True)
    oof_table[out_oof] = oof_table[out_oof].fillna(global_mean).astype("float32")

    full_agg = df_local.groupby(group_col, sort=False).agg(
        _t_sum=("_target", "sum"),
        _t_n=("_target", "size"),
    )
    full_agg[out_full] = (
        (full_agg["_t_sum"] + smooth * global_mean) / (full_agg["_t_n"] + smooth)
    ).astype("float32")
    full_table = full_agg[[out_full]].reset_index()

    return oof_table, full_table


# ---------------------------------------------------------------------------
# Destination history — group statistics by srch_destination_id
# ---------------------------------------------------------------------------

def compute_dest_history(df_train: pd.DataFrame, smooth: float = 200.0) -> pd.DataFrame:
    """Numeric encoding of high-cardinality `srch_destination_id`."""
    global_ctr = float(df_train["click_bool"].mean())
    global_btr = float(df_train["booking_bool"].mean())
    global_price = float(df_train["price_usd"].mean())
    global_star = float(df_train["prop_starrating"].mean())

    g = df_train.groupby("srch_destination_id", sort=False)
    out = pd.DataFrame({
        "dest_count": g.size().astype("int64"),
        "dest_click_sum": g["click_bool"].sum().astype("int64"),
        "dest_book_sum": g["booking_bool"].sum().astype("int64"),
        "dest_price_mean": g["price_usd"].mean().astype("float32"),
        "dest_price_std": g["price_usd"].std().fillna(0.0).astype("float32"),
        "dest_price_median": g["price_usd"].median().astype("float32"),
        "dest_price_q25": g["price_usd"].quantile(0.25).astype("float32"),
        "dest_price_q75": g["price_usd"].quantile(0.75).astype("float32"),
        "dest_price_skew": g["price_usd"].skew().fillna(0.0).astype("float32"),
        "dest_star_mean": g["prop_starrating"].mean().astype("float32"),
        "dest_star_std": g["prop_starrating"].std().fillna(0.0).astype("float32"),
        "dest_loc2_median": g["prop_location_score2"].median().astype("float32"),
        "dest_loc2_max": g["prop_location_score2"].max().astype("float32"),
        "dest_loghist_mean": g["prop_log_historical_price"].mean().astype("float32"),
    }).reset_index()

    out["dest_ctr"] = (
        (out["dest_click_sum"] + smooth * global_ctr) / (out["dest_count"] + smooth)
    ).astype("float32")
    out["dest_btr"] = (
        (out["dest_book_sum"] + smooth * global_btr) / (out["dest_count"] + smooth)
    ).astype("float32")
    out["dest_count_log"] = np.log(out["dest_count"].astype("float32") + 1.0).astype("float32")
    out["dest_price_iqr"] = (out["dest_price_q75"] - out["dest_price_q25"]).astype("float32")
    out["dest_price_mean"] = out["dest_price_mean"].fillna(global_price).astype("float32")
    out["dest_price_median"] = out["dest_price_median"].fillna(global_price).astype("float32")
    out["dest_star_mean"] = out["dest_star_mean"].fillna(global_star).astype("float32")

    return out[[
        "srch_destination_id",
        "dest_ctr",
        "dest_btr",
        "dest_count_log",
        "dest_price_mean",
        "dest_price_median",
        "dest_price_std",
        "dest_price_iqr",
        "dest_price_skew",
        "dest_star_mean",
        "dest_star_std",
        "dest_loc2_median",
        "dest_loc2_max",
        "dest_loghist_mean",
    ]]


# ---------------------------------------------------------------------------
# Expected position proxy — mean position by (destination, month) with
# hierarchical empirical-Bayes smoothing toward the destination's overall mean.
# ---------------------------------------------------------------------------

def compute_dest_month_pos(df_train: pd.DataFrame, smooth: float = 30.0) -> pd.DataFrame:
    """Expected position when this destination is searched in this month.

    Hierarchical empirical-Bayes: each (dest, month) cell shrinks toward the
    destination-wide mean, which shrinks toward the global mean.
    """
    global_pos = float(df_train["position"].mean())

    pos = df_train["position"].astype("float32")
    months = df_train["date_time"].dt.month.astype("int8")

    work = pd.DataFrame({
        "srch_destination_id": df_train["srch_destination_id"].to_numpy(),
        "month": months.to_numpy(),
        "_pos": pos.to_numpy(),
    })

    dest_stats = work.groupby("srch_destination_id", sort=False).agg(
        d_count=("_pos", "size"),
        d_sum=("_pos", "sum"),
    )
    dest_stats["d_prior"] = (
        (dest_stats["d_sum"] + smooth * global_pos) / (dest_stats["d_count"] + smooth)
    ).astype("float32")
    dest_prior = dest_stats[["d_prior"]].reset_index()

    cell = work.groupby(["srch_destination_id", "month"], sort=False).agg(
        dm_count=("_pos", "size"),
        dm_pos_mean=("_pos", "mean"),
    ).reset_index()
    cell = cell.merge(dest_prior, on="srch_destination_id", how="left")
    cell["expected_position"] = (
        (cell["dm_pos_mean"] * cell["dm_count"] + smooth * cell["d_prior"])
        / (cell["dm_count"] + smooth)
    ).astype("float32")

    return cell[["srch_destination_id", "month", "expected_position"]]


# ---------------------------------------------------------------------------
# Country-level prop_location_score2 median (for missing-value imputation)
# ---------------------------------------------------------------------------

def compute_country_loc2(df_train: pd.DataFrame) -> pd.DataFrame:
    return (
        df_train.groupby("prop_country_id", sort=False)["prop_location_score2"]
        .median()
        .astype("float32")
        .rename("country_loc2_median")
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Per-row features (post-fix)
# ---------------------------------------------------------------------------

def _per_row_features(df: pd.DataFrame, log_price_clip: tuple[float, float]) -> pd.DataFrame:
    lo, hi = log_price_clip
    df["log_price"] = np.log1p(df["price_usd"]).astype("float32")
    df["log_price_clipped"] = df["log_price"].clip(lo, hi).astype("float32")
    df["month"] = df["date_time"].dt.month.astype("int8")
    # pandas weekday is 0..6; polars weekday is 1..7. We use values directly
    # for the model — absolute encoding doesn't matter as long as it's stable.
    df["day_of_week"] = df["date_time"].dt.weekday.astype("int8")
    df["hour"] = df["date_time"].dt.hour.astype("int8")

    df["price_diff_visitor"] = (
        df["price_usd"].astype("float32") - df["visitor_hist_adr_usd"].astype("float32")
    ).astype("float32")
    df["star_diff_visitor"] = (
        df["prop_starrating"].astype("float32") - df["visitor_hist_starrating"].astype("float32")
    ).astype("float32")
    df["has_visitor_history"] = df["visitor_hist_adr_usd"].notna().astype("int8")
    df["family_search"] = (df["srch_children_count"] > 0).astype("int8")
    df["guests_total"] = (
        df["srch_adults_count"].astype("int16") + df["srch_children_count"].astype("int16")
    ).astype("int16")

    adults_f = df["srch_adults_count"].astype("float32")
    children_f = df["srch_children_count"].astype("float32")
    nights = df["srch_length_of_stay"].astype("float32").clip(1.0, 60.0)

    df["price_per_person"] = (df["price_usd"] / (adults_f + 0.5 * children_f + 0.001)).astype("float32")
    df["price_per_night"] = (df["price_usd"] / nights).astype("float32")
    df["combined_quality"] = (
        df["prop_starrating"].astype("float32").fillna(0.0)
        + 0.5 * df["prop_review_score"].astype("float32").fillna(0.0)
    ).astype("float32")
    df["combined_location"] = (
        df["prop_location_score1"].astype("float32") * df["prop_location_score2"].astype("float32")
    ).astype("float32")
    df["loc2_over_loc1"] = (
        df["prop_location_score2"].astype("float32") / (df["prop_location_score1"].astype("float32") + 1e-4)
    ).astype("float32")
    df["log_price_x_loc2"] = (
        df["log_price"] * df["prop_location_score2"].astype("float32").fillna(0.0)
    ).astype("float32")

    # Check-in date features.
    check_in = df["date_time"] + pd.to_timedelta(
        df["srch_booking_window"].astype("int64"), unit="D"
    )
    df["check_in_month"] = check_in.dt.month.astype("int8")
    df["check_in_day_of_week"] = check_in.dt.weekday.astype("int8")

    # Missing-value flags for high-missing columns (post-zero-fix).
    for c in C.HIGH_MISSING_COLS:
        if c in df.columns:
            df[f"{c}_isnull"] = df[c].isna().astype("int8")
    return df


# ---------------------------------------------------------------------------
# Within-search aggregations (rank, z-score, diff-from-mean/median)
# ---------------------------------------------------------------------------

def _within_search_features(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    grp = df.groupby("srch_id", sort=False)
    for c in cols:
        col = df[c].astype("float32")
        mean = grp[c].transform("mean").astype("float32")
        median = grp[c].transform("median").astype("float32")
        std = grp[c].transform("std").fillna(0.0).astype("float32")
        df[f"{c}_mean_diff"] = (col - mean).astype("float32")
        df[f"{c}_median_diff"] = (col - median).astype("float32")
        df[f"{c}_zscore"] = ((col - mean) / (std + 1e-6)).astype("float32")
        df[f"{c}_rank"] = grp[c].rank(method="average").astype("float32")
    return df


# ---------------------------------------------------------------------------
# Competitor feature collapse
# ---------------------------------------------------------------------------

def _comp_features(df: pd.DataFrame) -> pd.DataFrame:
    rate = df[C.COMP_RATE_COLS]
    inv = df[C.COMP_INV_COLS]
    diff = df[C.COMP_DIFF_COLS]

    df["comp_lower_count"] = (rate == 1).sum(axis=1).astype("int8")
    df["comp_higher_count"] = (rate == -1).sum(axis=1).astype("int8")
    df["comp_equal_count"] = (rate == 0).sum(axis=1).astype("int8")
    df["comp_unavail_count"] = (inv == 1).sum(axis=1).astype("int8")
    df["comp_rate_diff_mean"] = diff.mean(axis=1).fillna(0.0).astype("float32")
    df["any_cheaper_competitor"] = (df["comp_lower_count"] > 0).astype("int8")
    return df


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def engineer(
    df: pd.DataFrame,
    prop_history: pd.DataFrame,
    dest_history: pd.DataFrame,
    dest_month_pos: pd.DataFrame,
    country_loc2: pd.DataFrame,
    is_labeled: bool,
    log_price_clip: tuple[float, float] = (0.0, 12.0),
    prop_te_oof: pd.DataFrame | None = None,
    prop_te_full: pd.DataFrame | None = None,
    dest_te_oof: pd.DataFrame | None = None,
    dest_te_full: pd.DataFrame | None = None,
    train_qids: np.ndarray | None = None,
) -> pd.DataFrame:
    df = fix_zero_as_missing(df)
    df = _per_row_features(df, log_price_clip)
    df = _within_search_features(df, C.WITHIN_SRCH_NUMERIC)
    df = _comp_features(df)

    df = df.merge(prop_history, on="prop_id", how="left")
    df = df.merge(dest_history, on="srch_destination_id", how="left")
    df = df.merge(dest_month_pos, on=["srch_destination_id", "month"], how="left")
    df = df.merge(country_loc2, on="prop_country_id", how="left")

    # K-fold target encoding — one of the most important features by gain.
    if prop_te_full is not None:
        df = df.merge(prop_te_full, on="prop_id", how="left")
        if is_labeled and prop_te_oof is not None and train_qids is not None:
            df = df.merge(prop_te_oof, on=["srch_id", "prop_id"], how="left")
            in_train = df["srch_id"].isin(train_qids).to_numpy()
            df["prop_te"] = np.where(
                in_train, df["prop_id_te_oof"].to_numpy(), df["prop_id_te_full"].to_numpy()
            ).astype("float32")
            df = df.drop(columns=["prop_id_te_oof", "prop_id_te_full"])
        else:
            df["prop_te"] = df["prop_id_te_full"].astype("float32")
            df = df.drop(columns=["prop_id_te_full"])

    if dest_te_full is not None:
        df = df.merge(dest_te_full, on="srch_destination_id", how="left")
        if is_labeled and dest_te_oof is not None and train_qids is not None:
            df = df.merge(dest_te_oof, on=["srch_id", "srch_destination_id"], how="left")
            in_train = df["srch_id"].isin(train_qids).to_numpy()
            df["dest_te"] = np.where(
                in_train,
                df["srch_destination_id_te_oof"].to_numpy(),
                df["srch_destination_id_te_full"].to_numpy(),
            ).astype("float32")
            df = df.drop(columns=["srch_destination_id_te_oof", "srch_destination_id_te_full"])
        else:
            df["dest_te"] = df["srch_destination_id_te_full"].astype("float32")
            df = df.drop(columns=["srch_destination_id_te_full"])

    # Country-median imputation for prop_location_score2, then global fallback.
    global_loc2 = float(country_loc2["country_loc2_median"].median())
    loc2 = df["prop_location_score2"].astype("float32")
    df["prop_location_score2_imputed"] = (
        loc2.fillna(df["country_loc2_median"]).fillna(global_loc2).astype("float32")
    )

    # Fill defaults for unseen joins.
    fill_map = {
        "prop_ctr": float(prop_history["prop_ctr"].mean()),
        "prop_btr": float(prop_history["prop_btr"].mean()),
        "prop_avg_position": float(prop_history["prop_avg_position"].mean()),
        "prop_pos_std": 0.0,
        "prop_price_mean": float(prop_history["prop_price_mean"].mean()),
        "prop_price_median": float(prop_history["prop_price_median"].mean()),
        "prop_price_std": 0.0,
        "prop_price_q25": float(prop_history["prop_price_q25"].median()),
        "prop_price_q75": float(prop_history["prop_price_q75"].median()),
        "prop_price_iqr": float(prop_history["prop_price_iqr"].median()),
        "prop_price_skew": 0.0,
        "prop_loghist_mean": float(prop_history["prop_loghist_mean"].mean()),
        "prop_loghist_std": 0.0,
        "prop_loghist_skew": 0.0,
        "prop_review_mean": float(prop_history["prop_review_mean"].mean()),
        "prop_review_std": 0.0,
        "prop_count_log": 0.0,
        "position_importance": 0.0,
        "dest_ctr": float(dest_history["dest_ctr"].mean()),
        "dest_btr": float(dest_history["dest_btr"].mean()),
        "dest_count_log": 0.0,
        "dest_price_mean": float(dest_history["dest_price_mean"].mean()),
        "dest_price_median": float(dest_history["dest_price_median"].mean()),
        "dest_price_std": 0.0,
        "dest_price_iqr": float(dest_history["dest_price_iqr"].median()),
        "dest_price_skew": 0.0,
        "dest_star_mean": float(dest_history["dest_star_mean"].mean()),
        "dest_star_std": 0.0,
        "dest_loc2_median": float(dest_history["dest_loc2_median"].median()),
        "dest_loc2_max": float(dest_history["dest_loc2_max"].median()),
        "dest_loghist_mean": float(dest_history["dest_loghist_mean"].mean()),
        "expected_position": float(dest_month_pos["expected_position"].median()),
        "country_loc2_median": global_loc2,
    }
    for col, val in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(val).astype("float32")

    if prop_te_full is not None:
        df["prop_te"] = df["prop_te"].fillna(float(prop_te_full["prop_id_te_full"].mean())).astype("float32")
    if dest_te_full is not None:
        df["dest_te"] = df["dest_te"].fillna(float(dest_te_full["srch_destination_id_te_full"].mean())).astype("float32")

    # Derived features that require the joined statistics.
    price = df["price_usd"].astype("float32")
    df["recent_price_delta"] = (price - df["prop_price_median"]).astype("float32")
    df["price_vs_prop_mean_ratio"] = (price / (df["prop_price_mean"] + 1e-3)).astype("float32")
    df["price_vs_dest_mean_ratio"] = (price / (df["dest_price_mean"] + 1e-3)).astype("float32")
    df["price_z_in_dest"] = (
        (price - df["dest_price_mean"]) / (df["dest_price_std"] + 1e-3)
    ).astype("float32")
    df["star_vs_dest_diff"] = (
        df["prop_starrating"].astype("float32") - df["dest_star_mean"]
    ).astype("float32")
    df["loc2_vs_dest_max_ratio"] = (
        df["prop_location_score2_imputed"] / (df["dest_loc2_max"] + 1e-4)
    ).astype("float32")

    if is_labeled:
        book = df["booking_bool"].to_numpy().astype("int8")
        click = df["click_bool"].to_numpy().astype("int8")
        df["relevance"] = (book * 5 + ((click == 1) & (book == 0)).astype("int8")).astype("int8")
    return df


# ---------------------------------------------------------------------------
# Feature column listing (single source of truth for model inputs)
# ---------------------------------------------------------------------------

def model_input_columns(df: pd.DataFrame) -> list[str]:
    """Columns to feed the model. Excludes targets, ids, raw text, comp{i}_* originals."""
    excluded = {
        "srch_id",
        "date_time",
        "position",
        "click_bool",
        "booking_bool",
        "gross_bookings_usd",
        "relevance",
        # High-card identifiers; replaced by their frequency (count_log) and
        # target encoding (te) features. Sample paper reports both as
        # overfitting when fed in raw.
        "prop_id",
        "srch_destination_id",
        # Auxiliaries used only for joining / imputation; not direct model inputs.
        "country_loc2_median",
        # `prop_ctr`/`prop_btr` are computed on the full train fold and so include
        # each row's own labels, leaking signal at train time. We saw a +0.26
        # train-vs-val NDCG gap which traced to heavy reliance on these. The
        # k-fold OOF `prop_te` carries the same signal without the leak.
        "prop_ctr",
        "prop_btr",
        "dest_ctr",
        "dest_btr",
    }
    excluded.update(C.COMP_RATE_COLS)
    excluded.update(C.COMP_INV_COLS)
    excluded.update(C.COMP_DIFF_COLS)
    return [c for c in df.columns if c not in excluded]
