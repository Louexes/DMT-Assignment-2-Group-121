"""Phase 4 — train the unweighted LightGBM LambdaRank model."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src import lightgbm_train as L


def feature_importance_plot(booster, feat_cols, path: Path) -> None:
    imp = pd.DataFrame({
        "feature": feat_cols,
        "gain": booster.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=True).tail(40)
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.barh(imp["feature"], imp["gain"], color="steelblue")
    ax.set_title("LightGBM feature importance (gain) — top 40")
    fig.savefig(path, dpi=140, bbox_inches="tight")


def main() -> None:
    t0 = time.time()
    df, feat_cols, cat_cols = L.load_features()
    print(f"[load] {len(df):,} rows  ({time.time()-t0:.1f}s)")

    parts = L.split_train_val(df, feat_cols)
    print(f"[split] train rows={len(parts['X_tr']):,}  val rows={len(parts['X_va']):,}")
    print(f"[split] train queries={len(parts['g_tr']):,}  val queries={len(parts['g_va']):,}")

    booster, eval_log = L.train(parts, cat_cols, save_path=C.LGBM_MODEL)
    metrics = L.evaluate(booster, parts)
    print(f"[eval] NDCG@5={metrics['ndcg@5']:.4f}  NDCG@10={metrics['ndcg@10']:.4f}  best_iter={metrics['best_iteration']}")

    # Persist val predictions for ensemble + bias analysis.
    val_df = pd.DataFrame({
        "srch_id": parts["q_va"],
        "relevance": parts["y_va"],
        "click_bool": parts["click_va"],
        "booking_bool": parts["book_va"],
        "score": metrics["preds"],
    })
    val_df.to_parquet(C.LGBM_VAL_PREDS, compression="zstd", index=False)

    out_metrics = {
        "best_iteration": metrics["best_iteration"],
        "ndcg@5": metrics["ndcg@5"],
        "ndcg@10": metrics["ndcg@10"],
        "n_features": len(feat_cols),
        "training_time_sec": round(time.time() - t0, 1),
        "params": C.LGBM_PARAMS,
    }
    with open(C.OUT_DIR / "lightgbm_metrics.json", "w") as f:
        json.dump(out_metrics, f, indent=2)

    feature_importance_plot(booster, feat_cols, C.FIG_DIR / "lightgbm_feature_importance.png")
    print(f"[ok] phase 4 done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
