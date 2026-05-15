"""Exploratory data analysis on the labeled training set.

Each function writes one artefact (plot or JSON section) and returns a dict
that the entry-point script aggregates into outputs/eda_stats.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C


def _save(fig, name: str) -> Path:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIG_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def basic_counts(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_queries": int(df["srch_id"].nunique()),
        "n_props": int(df["prop_id"].nunique()),
        "n_destinations": int(df["srch_destination_id"].nunique()),
        "click_rate": float(df["click_bool"].mean()),
        "book_rate": float(df["booking_bool"].mean()),
        "click_given_book": float(df.loc[df["booking_bool"] == 1, "click_bool"].mean()),
        "random_share": float(df["random_bool"].mean()),
    }


def missing_value_plot(df: pd.DataFrame) -> dict:
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(miss))))
    sns.barplot(x=miss.values, y=miss.index, ax=ax, color="steelblue")
    ax.set_xlabel("% missing")
    ax.set_title("Missing values by column (labeled set)")
    _save(fig, "missing_values.png")
    return {"missing_pct": {k: float(v) for k, v in miss.items()}}


def props_per_query_plot(df: pd.DataFrame) -> dict:
    counts = df.groupby("srch_id", sort=False).size()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(counts.values, bins=range(1, counts.max() + 2), color="steelblue", edgecolor="white")
    ax.set_xlabel("properties per search")
    ax.set_ylabel("number of searches")
    ax.set_title("List length distribution")
    _save(fig, "props_per_query.png")
    return {
        "props_per_query_mean": float(counts.mean()),
        "props_per_query_median": float(counts.median()),
        "props_per_query_max": int(counts.max()),
        "props_per_query_min": int(counts.min()),
    }


def position_bias_plot(df: pd.DataFrame) -> dict:
    """Click rate by position, split by random_bool. The gap quantifies position bias."""
    sub = df[["position", "click_bool", "booking_bool", "random_bool"]].copy()
    sorted_ = sub[sub["random_bool"] == 0]
    random_ = sub[sub["random_bool"] == 1]

    sorted_ctr = sorted_.groupby("position")["click_bool"].mean()
    random_ctr = random_.groupby("position")["click_bool"].mean()
    sorted_btr = sorted_.groupby("position")["booking_bool"].mean()
    random_btr = random_.groupby("position")["booking_bool"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    axes[0].plot(sorted_ctr.index, sorted_ctr.values, label="sorted (random_bool=0)", color="steelblue")
    axes[0].plot(random_ctr.index, random_ctr.values, label="random (random_bool=1)", color="darkorange")
    axes[0].set_xlabel("displayed position")
    axes[0].set_ylabel("click rate")
    axes[0].set_title("Click rate by position")
    axes[0].legend()
    axes[1].plot(sorted_btr.index, sorted_btr.values, label="sorted", color="steelblue")
    axes[1].plot(random_btr.index, random_btr.values, label="random", color="darkorange")
    axes[1].set_xlabel("displayed position")
    axes[1].set_ylabel("booking rate")
    axes[1].set_title("Booking rate by position")
    axes[1].legend()
    _save(fig, "position_bias.png")

    # Quantify: ratio of CTR at pos 1 to pos 5 in each subset.
    def _ratio(s):
        return float(s.loc[1] / max(s.loc[5], 1e-9)) if 1 in s.index and 5 in s.index else None

    return {
        "ctr_pos1_over_pos5_sorted": _ratio(sorted_ctr),
        "ctr_pos1_over_pos5_random": _ratio(random_ctr),
        "ctr_at_pos1_sorted": float(sorted_ctr.loc[1]) if 1 in sorted_ctr.index else None,
        "ctr_at_pos1_random": float(random_ctr.loc[1]) if 1 in random_ctr.index else None,
    }


def price_star_plots(df: pd.DataFrame) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    price = df["price_usd"].clip(0, df["price_usd"].quantile(0.99))
    axes[0].hist(price.dropna(), bins=80, color="steelblue", edgecolor="white")
    axes[0].set_title("price_usd (clipped at p99)")
    axes[0].set_xlabel("USD")
    axes[1].hist(np.log1p(df["price_usd"].dropna()), bins=80, color="steelblue", edgecolor="white")
    axes[1].set_title("log1p(price_usd)")
    _save(fig, "price_distribution.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x="prop_starrating", data=df, color="steelblue", ax=ax)
    ax.set_title("prop_starrating distribution")
    _save(fig, "starrating.png")

    return {
        "price_p50": float(df["price_usd"].median()),
        "price_p99": float(df["price_usd"].quantile(0.99)),
        "starrating_value_counts": df["prop_starrating"].value_counts().to_dict(),
    }


def brand_exposure_plot(df: pd.DataFrame) -> dict:
    """Chain (prop_brand_bool=1) vs independent (=0) exposure metrics."""
    g = df.groupby("prop_brand_bool", sort=False).agg(
        impressions=("srch_id", "size"),
        clicks=("click_bool", "sum"),
        bookings=("booking_bool", "sum"),
    )
    g["ctr"] = g["clicks"] / g["impressions"]
    g["btr"] = g["bookings"] / g["impressions"]

    fig, ax = plt.subplots(figsize=(6, 4))
    g[["ctr", "btr"]].plot(kind="bar", ax=ax, color=["steelblue", "darkorange"])
    ax.set_xticklabels(["independent", "chain"], rotation=0)
    ax.set_title("CTR / BTR by brand")
    ax.set_ylabel("rate")
    _save(fig, "brand_exposure.png")

    return {
        "brand_split": {
            int(k): {
                "impressions": int(v["impressions"]),
                "clicks": int(v["clicks"]),
                "bookings": int(v["bookings"]),
                "ctr": float(v["ctr"]),
                "btr": float(v["btr"]),
            }
            for k, v in g.iterrows()
        }
    }


def correlation_heatmap(df: pd.DataFrame) -> dict:
    cols = [
        "price_usd",
        "prop_starrating",
        "prop_review_score",
        "prop_brand_bool",
        "prop_location_score1",
        "prop_location_score2",
        "prop_log_historical_price",
        "promotion_flag",
        "srch_length_of_stay",
        "srch_booking_window",
        "srch_adults_count",
        "srch_children_count",
        "srch_room_count",
        "srch_query_affinity_score",
        "orig_destination_distance",
        "click_bool",
        "booking_bool",
    ]
    cols = [c for c in cols if c in df.columns]
    sample = df[cols].sample(n=min(500_000, len(df)), random_state=C.SEED)
    corr = sample.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Pearson correlation (500k sample)")
    _save(fig, "correlation_heatmap.png")
    return {"correlation_with_book": corr["booking_bool"].drop("booking_bool").to_dict()}


def run_all(df: pd.DataFrame) -> dict:
    out: dict = {}
    out["counts"] = basic_counts(df)
    out["missing"] = missing_value_plot(df)
    out["list_length"] = props_per_query_plot(df)
    out["position_bias"] = position_bias_plot(df)
    out["price_star"] = price_star_plots(df)
    out["brand_exposure"] = brand_exposure_plot(df)
    out["correlations"] = correlation_heatmap(df)
    return out


def write_summary(stats: dict) -> Path:
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = C.OUT_DIR / "eda_stats.json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    return path
