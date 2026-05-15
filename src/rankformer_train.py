"""Training loop and dataset for RankFormer.

Memory layout:
    All rows stay flat in numpy arrays sorted by srch_id. We compute group
    offsets so a single batch slices `(start, end)` from the flat tensors and
    then pads to the maximum list length in the batch.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .ndcg import ndcg_at_k
from .rankformer import (
    RankFormer,
    RankFormerConfig,
    lambdarank_loss,
    listnet_loss,
    pairwise_hinge,
)

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


# ---------------------------------------------------------------------------
# Pre-processing: standardise numeric + remap categoricals to dense indices.
# ---------------------------------------------------------------------------


@dataclass
class FeatureStats:
    num_cols: list[str]
    cat_cols: list[str]
    num_means: np.ndarray   # (N_num,)
    num_stds: np.ndarray    # (N_num,)
    cat_maps: dict[str, dict]   # raw_id -> dense_index (0 reserved for unseen/padding)
    cat_cardinalities: dict[str, int]


def fit_feature_stats(
    df_tr: pd.DataFrame, num_cols: list[str], cat_cols: list[str]
) -> FeatureStats:
    means = df_tr[num_cols].mean(numeric_only=False).fillna(0.0).to_numpy(dtype=np.float32)
    stds = df_tr[num_cols].std(numeric_only=False).fillna(1.0).to_numpy(dtype=np.float32)
    stds[stds < 1e-6] = 1.0
    cat_maps = {}
    cat_cards = {}
    for c in cat_cols:
        uniq = df_tr[c].dropna().unique()
        # Index 0 reserved for unseen/padding; valid ids start at 1.
        m = {int(v): i + 1 for i, v in enumerate(uniq)}
        cat_maps[c] = m
        cat_cards[c] = len(m) + 1
    return FeatureStats(
        num_cols=num_cols, cat_cols=cat_cols,
        num_means=means, num_stds=stds,
        cat_maps=cat_maps, cat_cardinalities=cat_cards,
    )


def transform(df: pd.DataFrame, stats: FeatureStats) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Returns standardised numeric matrix (float32) and per-cat int32 arrays."""
    num = df[stats.num_cols].to_numpy(dtype=np.float32, copy=False)
    # Replace NaN with the per-column mean *before* standardisation by setting
    # to mean so the standardised value becomes 0.
    nan_mask = np.isnan(num)
    if nan_mask.any():
        # broadcast mean
        col_means = np.broadcast_to(stats.num_means, num.shape)
        num = np.where(nan_mask, col_means, num)
    num = (num - stats.num_means) / stats.num_stds
    cats: dict[str, np.ndarray] = {}
    for c in stats.cat_cols:
        m = stats.cat_maps[c]
        raw = df[c].fillna(-1).to_numpy()
        idx = np.zeros(len(raw), dtype=np.int32)
        # Use a vectorised lookup via dict.get; small categorical sets so loop
        # is fine for ~5M; precompute via pandas map for speed.
        ser = pd.Series(raw).map(m).fillna(0).astype("int32")
        idx = ser.to_numpy()
        cats[c] = idx
    return num.astype(np.float32, copy=False), cats


# ---------------------------------------------------------------------------
# Dataset & collate
# ---------------------------------------------------------------------------


class SessionDataset(Dataset):
    """Yields per-query slices from flat arrays."""
    def __init__(
        self,
        offsets: np.ndarray,   # (n_queries + 1,)
        srch_ids: np.ndarray,  # (n_queries,)
        num: np.ndarray,
        cats: dict[str, np.ndarray],
        relevance: np.ndarray | None,
    ):
        self.offsets = offsets
        self.srch_ids = srch_ids
        self.num = num
        self.cats = cats
        self.relevance = relevance

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int):
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return {
            "num": self.num[a:b],
            "cats": {k: v[a:b] for k, v in self.cats.items()},
            "rel": self.relevance[a:b] if self.relevance is not None else None,
            "srch_id": int(self.srch_ids[i]),
            "n": b - a,
        }


def make_collate(max_list_len: int, n_num: int, cat_keys: list[str]):
    def collate(batch):
        B = len(batch)
        L = min(max_list_len, max(s["n"] for s in batch))
        num = np.zeros((B, L, n_num), dtype=np.float32)
        cats = {k: np.zeros((B, L), dtype=np.int32) for k in cat_keys}
        rel = np.zeros((B, L), dtype=np.int8)
        mask = np.zeros((B, L), dtype=bool)
        srch_ids = np.zeros(B, dtype=np.int64)
        actual_lens = np.zeros(B, dtype=np.int32)
        for i, s in enumerate(batch):
            n = min(s["n"], L)
            num[i, :n] = s["num"][:n]
            for k in cat_keys:
                cats[k][i, :n] = s["cats"][k][:n]
            if s["rel"] is not None:
                rel[i, :n] = s["rel"][:n]
            mask[i, :n] = True
            srch_ids[i] = s["srch_id"]
            actual_lens[i] = n
        out = {
            "num": torch.from_numpy(num),
            "cats": {k: torch.from_numpy(v).long() for k, v in cats.items()},
            "rel": torch.from_numpy(rel).long(),
            "mask": torch.from_numpy(mask),
            "srch_ids": srch_ids,
            "actual_lens": actual_lens,
        }
        return out
    return collate


def build_session_arrays(
    df: pd.DataFrame, num: np.ndarray, cats: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (offsets, unique_srch_ids) — df must already be sorted by srch_id."""
    srch = df["srch_id"].to_numpy()
    diff = np.concatenate([[True], srch[1:] != srch[:-1]])
    starts = np.where(diff)[0]
    offsets = np.concatenate([starts, [len(srch)]])
    unique_srch = srch[starts]
    return offsets, unique_srch


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def lr_schedule(
    step: int, warmup: int, total: int, base_lr: float, n_cycles: int = 1
) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if n_cycles <= 1:
        progress = (step - warmup) / max(1, total - warmup)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    cycle_len = max(1, (total - warmup) // n_cycles)
    cycle_prog = ((step - warmup) % cycle_len) / cycle_len
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * cycle_prog))


def evaluate(model: RankFormer, loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_scores: list[np.ndarray] = []
    all_qids: list[np.ndarray] = []
    all_rel: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            num = batch["num"].to(DEVICE)
            cats = {k: v.to(DEVICE) for k, v in batch["cats"].items()}
            mask = batch["mask"].to(DEVICE)
            scores = model(num, cats, mask).detach().cpu().numpy()
            for i in range(scores.shape[0]):
                n = int(batch["actual_lens"][i])
                all_scores.append(scores[i, :n])
                all_qids.append(np.full(n, int(batch["srch_ids"][i]), dtype=np.int64))
                all_rel.append(batch["rel"][i, :n].numpy())
    scores = np.concatenate(all_scores)
    qids = np.concatenate(all_qids)
    rel = np.concatenate(all_rel)
    ndcg5 = ndcg_at_k(qids, scores, rel, k=5)
    return ndcg5, scores, qids, rel


def predict_loader(model: RankFormer, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (srch_id_per_row, score_per_row, prop_idx_offset_within_batch_unused)."""
    model.eval()
    all_scores: list[np.ndarray] = []
    all_qids: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            num = batch["num"].to(DEVICE)
            cats = {k: v.to(DEVICE) for k, v in batch["cats"].items()}
            mask = batch["mask"].to(DEVICE)
            scores = model(num, cats, mask).cpu().numpy()
            for i in range(scores.shape[0]):
                n = int(batch["actual_lens"][i])
                all_scores.append(scores[i, :n])
                all_qids.append(np.full(n, int(batch["srch_ids"][i]), dtype=np.int64))
    return np.concatenate(all_qids), np.concatenate(all_scores), None  # type: ignore


def train_loop(
    model: RankFormer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg_train: dict,
    out_path: Path,
) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg_train["lr"], weight_decay=cfg_train["weight_decay"])
    epochs = cfg_train["epochs"]
    total_steps = epochs * len(train_loader)
    log_every = cfg_train.get("log_every_n_batches", 200)
    log: dict = {"train_loss": [], "val_ndcg@5": []}
    best_ndcg = -1.0
    step = 0
    n_batches_per_epoch = len(train_loader)
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            num = batch["num"].to(DEVICE)
            cats = {k: v.to(DEVICE) for k, v in batch["cats"].items()}
            mask = batch["mask"].to(DEVICE)
            rel = batch["rel"].to(DEVICE)

            scores = model(num, cats, mask)
            listnet_w = cfg_train.get("listnet_weight", 1.0)
            hinge_w = cfg_train.get("pairwise_lambda", 0.0)
            lambda_w = cfg_train.get("lambdarank_weight", 0.0)
            loss = scores.sum() * 0.0
            if listnet_w > 0:
                loss = loss + listnet_w * listnet_loss(
                    scores, rel, mask, cfg_train["softmax_temperature"]
                )
            if hinge_w > 0:
                loss = loss + hinge_w * pairwise_hinge(scores, rel, mask)
            if lambda_w > 0:
                loss = loss + lambda_w * lambdarank_loss(scores, rel, mask)

            for pg in opt.param_groups:
                pg["lr"] = lr_schedule(
                    step,
                    cfg_train["warmup_steps"],
                    total_steps,
                    cfg_train["lr"],
                    n_cycles=cfg_train.get("n_cycles", 1),
                )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += float(loss.item())
            n_batches += 1
            step += 1
            if n_batches % log_every == 0:
                elapsed = time.time() - t0
                rate = n_batches / max(elapsed, 1e-9)
                avg_loss = running / max(1, n_batches)
                print(
                    f"  epoch {epoch+1} batch {n_batches}/{n_batches_per_epoch}  "
                    f"loss={avg_loss:.4f}  rate={rate:.1f} batch/s  "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )
        train_loss = running / max(1, n_batches)

        t_eval = time.time()
        ndcg5, _, _, _ = evaluate(model, val_loader)
        eval_time = time.time() - t_eval
        log["train_loss"].append(train_loss)
        log["val_ndcg@5"].append(ndcg5)
        msg = (
            f"epoch {epoch+1}/{epochs}  loss={train_loss:.4f}  val_ndcg@5={ndcg5:.4f}  "
            f"train_time={time.time()-t0-eval_time:.0f}s  eval={eval_time:.0f}s"
        )
        print(msg, flush=True)
        if ndcg5 > best_ndcg:
            best_ndcg = ndcg5
            torch.save({"state_dict": model.state_dict()}, out_path)
    log["best_ndcg@5"] = best_ndcg
    return log
