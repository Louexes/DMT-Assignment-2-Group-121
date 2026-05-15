"""Train LightGBM on a named subset from artifacts/feature_ranking.json.

Usage: python scripts/04b_train_lgbm_subset.py <subset_name>
Example: python scripts/04b_train_lgbm_subset.py top_30
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src import lightgbm_train as L


def main() -> None:
    subset = sys.argv[1] if len(sys.argv) > 1 else "top_30"

    with open(C.ART_DIR / "feature_ranking.json") as f:
        ranking = json.load(f)
    if subset not in ranking:
        raise SystemExit(f"Unknown subset {subset!r}. Available: {sorted(ranking)}")
    feat_cols = ranking[subset]
    cat_cols = [c for c in C.CATEGORICAL_COLS if c in feat_cols]
    print(f"[subset] {subset}: {len(feat_cols)} features ({len(cat_cols)} categorical)")

    t0 = time.time()
    df, _, _ = L.load_features()
    print(f"[load] {len(df):,} rows  ({time.time()-t0:.1f}s)")

    parts = L.split_train_val(df, feat_cols)
    print(f"[split] train rows={len(parts['X_tr']):,}  val rows={len(parts['X_va']):,}")

    save_path = C.ART_DIR / f"lightgbm_{subset}.txt"
    booster, _ = L.train(parts, cat_cols, save_path=save_path)
    metrics = L.evaluate(booster, parts)
    print(f"[eval:{subset}] NDCG@5={metrics['ndcg@5']:.4f}  NDCG@10={metrics['ndcg@10']:.4f}  best_iter={metrics['best_iteration']}")

    with open(C.OUT_DIR / f"lightgbm_metrics_{subset}.json", "w") as f:
        json.dump({
            "subset": subset,
            "n_features": len(feat_cols),
            "ndcg@5": metrics["ndcg@5"],
            "ndcg@10": metrics["ndcg@10"],
            "best_iteration": int(metrics["best_iteration"]),
            "features": feat_cols,
            "categorical": cat_cols,
        }, f, indent=2)
    print(f"[ok] {subset} done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
