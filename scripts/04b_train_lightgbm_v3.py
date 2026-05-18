"""Phase 4b — train a second LightGBM booster ("v3") as a diverse leg of
the final ensemble.

Different seed and operating point from the primary booster trained in
phase 4: more leaves and depth, weaker min_data_in_leaf, compensated by
heavier lambda_l2 + path_smooth. Its scores are correlated with v1 but
not identical, which is what makes the blend in phase 7 add lift over
either booster alone.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src import lightgbm_train as L


def main() -> None:
    t0 = time.time()
    df, feat_cols, cat_cols = L.load_features()
    print(f"[load] {len(df):,} rows  ({time.time()-t0:.1f}s)")

    parts = L.split_train_val(df, feat_cols)
    print(f"[split] train rows={len(parts['X_tr']):,}  val rows={len(parts['X_va']):,}")

    print(f"[train] params={C.LGBM_V3_PARAMS}", flush=True)
    booster, _ = L.train(parts, cat_cols, params=C.LGBM_V3_PARAMS, save_path=C.LGBM_V3_MODEL)
    metrics = L.evaluate(booster, parts)
    print(f"[eval] NDCG@5={metrics['ndcg@5']:.4f}  NDCG@10={metrics['ndcg@10']:.4f}  "
          f"best_iter={metrics['best_iteration']}")

    val_df = pd.DataFrame({
        "srch_id": parts["q_va"],
        "relevance": parts["y_va"],
        "score": metrics["preds"],
    })
    val_df.to_parquet(C.LGBM_V3_VAL_PREDS, compression="zstd", index=False)

    out = {
        "best_iteration": metrics["best_iteration"],
        "ndcg@5": metrics["ndcg@5"],
        "ndcg@10": metrics["ndcg@10"],
        "n_features": len(feat_cols),
        "training_time_sec": round(time.time() - t0, 1),
        "params": C.LGBM_V3_PARAMS,
    }
    with open(C.OUT_DIR / "lightgbm_v3_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ok] phase 4b done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
