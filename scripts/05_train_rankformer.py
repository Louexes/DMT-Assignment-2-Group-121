"""Phase 5 — train RankFormer."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src.rankformer import RankFormer, RankFormerConfig
from src import rankformer_train as RT


def main() -> None:
    t0 = time.time()
    with open(C.FEATURE_LIST) as f:
        meta = json.load(f)
    feat_cols: list[str] = meta["features"]
    cat_cols: list[str] = meta["categorical"]
    num_cols = [c for c in feat_cols if c not in cat_cols]

    cols = list({"srch_id", "is_val", "click_bool", "booking_bool", "relevance", *feat_cols})
    df = pd.read_parquet(C.LABELED_FEAT, columns=cols)
    df = df.sort_values("srch_id", kind="stable").reset_index(drop=True)

    tr_df = df[df["is_val"] == 0].reset_index(drop=True)
    va_df = df[df["is_val"] == 1].reset_index(drop=True)
    print(f"[load] train rows={len(tr_df):,} val rows={len(va_df):,}  ({time.time()-t0:.1f}s)", flush=True)
    del df

    stats = RT.fit_feature_stats(tr_df, num_cols, cat_cols)
    n_num = len(num_cols)
    print(f"[features] N_num={n_num} cat_cards={stats.cat_cardinalities}", flush=True)

    tr_num, tr_cats = RT.transform(tr_df, stats)
    va_num, va_cats = RT.transform(va_df, stats)

    tr_offsets, tr_qids = RT.build_session_arrays(tr_df, tr_num, tr_cats)
    va_offsets, va_qids = RT.build_session_arrays(va_df, va_num, va_cats)

    tr_rel = tr_df["relevance"].to_numpy().astype(np.int8)
    va_rel = va_df["relevance"].to_numpy().astype(np.int8)
    # Drop pandas frames now that we have flat numpy arrays.
    del tr_df, va_df
    print(f"[memory] tr_num={tr_num.nbytes/1024**3:.2f} GB  va_num={va_num.nbytes/1024**3:.2f} GB", flush=True)

    tr_set = RT.SessionDataset(tr_offsets, tr_qids, tr_num, tr_cats, tr_rel)
    va_set = RT.SessionDataset(va_offsets, va_qids, va_num, va_cats, va_rel)

    cat_dims = {c: 32 if c == "srch_destination_id"
                  else 8 if c.endswith("country_id")
                  else 4 for c in cat_cols}
    cfg_model = RankFormerConfig(
        n_num=n_num,
        cat_cardinalities=stats.cat_cardinalities,
        cat_dims=cat_dims,
        d_model=C.RF_PARAMS["d_model"],
        n_heads=C.RF_PARAMS["n_heads"],
        n_layers=C.RF_PARAMS["n_layers"],
        ffn_dim=C.RF_PARAMS["ffn_dim"],
        dropout=C.RF_PARAMS["dropout"],
        max_list_len=C.RF_TRAIN["max_list_len"],
        numeric_mlp=C.RF_PARAMS.get("numeric_mlp", False),
        fuse_mlp=C.RF_PARAMS.get("fuse_mlp", False),
        use_pos_emb=C.RF_PARAMS.get("use_pos_emb", True),
    )
    model = RankFormer(cfg_model).to(RT.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] device={RT.DEVICE} params={n_params/1e6:.2f}M")

    collate = RT.make_collate(C.RF_TRAIN["max_list_len"], n_num, list(cat_cols))
    g = torch.Generator()
    g.manual_seed(C.SEED)
    tr_loader = DataLoader(
        tr_set, batch_size=C.RF_TRAIN["batch_sessions"], shuffle=True,
        collate_fn=collate, generator=g, num_workers=0, drop_last=True,
    )
    va_loader = DataLoader(
        va_set, batch_size=C.RF_TRAIN["batch_sessions"], shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    log = RT.train_loop(model, tr_loader, va_loader, C.RF_TRAIN, C.RF_MODEL)
    print(f"[best] val_ndcg@5={log['best_ndcg@5']:.4f}")

    # Reload best model and produce val predictions for ensembling.
    model.load_state_dict(torch.load(C.RF_MODEL, map_location=RT.DEVICE)["state_dict"])
    qids, scores, _ = RT.predict_loader(model, va_loader)
    val_df = pd.DataFrame({"srch_id": qids, "score": scores})
    val_df.to_parquet(C.RF_VAL_PREDS, compression="zstd", index=False)

    out = {
        "n_params": n_params,
        "training_time_sec": round(time.time() - t0, 1),
        "best_ndcg@5": log["best_ndcg@5"],
        "train_loss": log["train_loss"],
        "val_ndcg@5_per_epoch": log["val_ndcg@5"],
        "params": {**C.RF_PARAMS, **C.RF_TRAIN},
    }
    with open(C.OUT_DIR / "rankformer_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    # Stash feature stats so phase 7 can reuse them.
    np.savez(
        C.ART_DIR / "rankformer_stats.npz",
        num_cols=np.array(num_cols),
        num_means=stats.num_means,
        num_stds=stats.num_stds,
        cat_cols=np.array(cat_cols),
        **{f"map__{c}__keys": np.array(list(stats.cat_maps[c].keys()), dtype=np.int64) for c in cat_cols},
        **{f"map__{c}__vals": np.array(list(stats.cat_maps[c].values()), dtype=np.int32) for c in cat_cols},
    )
    print(f"[ok] phase 5 done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
