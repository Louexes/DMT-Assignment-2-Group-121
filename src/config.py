"""Project paths, constants, hyperparameters, and feature lists.

The provided dataset uses the standard naming convention:
- `Data/train.csv`  — labelled training set (with click_bool / booking_bool /
                      position / gross_bookings_usd).
- `Data/test.csv`   — unlabelled test set; this is what we submit predictions
                      for on Kaggle.

NOTE: an earlier iteration of this project incorrectly assumed the files were
swapped. The aliases below reflect the *standard* layout. If your local files
follow a non-standard convention, swap `LABELED_CSV` and `SUBMIT_CSV` here.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
ART_DIR = ROOT / "artifacts"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

# Standard naming: train.csv has labels, test.csv is the unlabelled set we
# submit predictions for.
LABELED_CSV = DATA_DIR / "train.csv"
SUBMIT_CSV = DATA_DIR / "test.csv"
SUBMISSION_SAMPLE = DATA_DIR / "submission_sample.csv"

LABELED_PARQUET = ART_DIR / "labeled.parquet"
SUBMIT_PARQUET = ART_DIR / "submit.parquet"
LABELED_FEAT = ART_DIR / "labeled_feat.parquet"
SUBMIT_FEAT = ART_DIR / "submit_feat.parquet"
PROP_HIST = ART_DIR / "prop_history.parquet"
FEATURE_LIST = ART_DIR / "feature_list.json"

LGBM_MODEL = ART_DIR / "lightgbm.txt"
LGBM_VAL_PREDS = ART_DIR / "lightgbm_val_preds.parquet"

# RankFormer artefacts. The transformer ranker is included as a strong
# secondary model. With the canonical config below (MLP per-item encoder,
# no positional embedding, cosine LR with warm restarts) it reaches val
# NDCG@5 ≈ 0.413 — within ~0.007 of LightGBM. We still submit LightGBM as
# the final predictor; RankFormer is kept for the methods write-up.
RF_MODEL = ART_DIR / "rankformer.pt"
RF_VAL_PREDS = ART_DIR / "rankformer_val_preds.parquet"

SEED = 42

# Columns that are integer IDs / categorical. LightGBM treats these as
# categorical when listed explicitly in `categorical_feature`.
# Note: prop_id and srch_destination_id are *deliberately omitted* — they
# overfit when included raw. We expose them through their count-log
# (frequency encoding) + k-fold target encoding instead.
CATEGORICAL_COLS = [
    "site_id",
    "visitor_location_country_id",
    "prop_country_id",
    "month",
    "day_of_week",
]

# Per-row numeric features that always exist post-engineering.
COMP_INDICES = list(range(1, 9))
COMP_RATE_COLS = [f"comp{i}_rate" for i in COMP_INDICES]
COMP_INV_COLS = [f"comp{i}_inv" for i in COMP_INDICES]
COMP_DIFF_COLS = [f"comp{i}_rate_percent_diff" for i in COMP_INDICES]

# High-missing columns that benefit from a missing-flag binary feature.
# `prop_starrating` and `prop_log_historical_price` are included because the
# zero-as-missing repair in `features.fix_zero_as_missing` turns their 0s
# into nulls.
HIGH_MISSING_COLS = [
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "prop_starrating",
    "prop_review_score",
    "prop_location_score2",
    "srch_query_affinity_score",
    "orig_destination_distance",
    "prop_log_historical_price",
]

# Columns we apply within-search rank/zscore/mean_diff/median_diff to.
WITHIN_SRCH_NUMERIC = [
    "price_usd",
    "log_price",
    "prop_starrating",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "combined_quality",
    "price_per_person",
    "loc2_over_loc1",
]

# LightGBM hyperparameters. With our OOF target-encoded features the
# regularisation regime tolerates slightly more capacity per tree
# (num_leaves=72, min_data_in_leaf=80) and a marginally faster learning
# rate (0.012); we counter this with a heavier path_smooth (1.5) and a
# more frequent bagging cadence (bagging_freq=4). Lambda_l1/l2 tuned
# down modestly because path_smooth carries some of the regularisation
# load now.
LGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5, 10],
    "lambdarank_truncation_level": 5,
    "learning_rate": 0.012,
    "num_leaves": 72,
    "max_depth": 8,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.65,
    "bagging_fraction": 0.65,
    "bagging_freq": 4,
    "lambda_l1": 1.5,
    "lambda_l2": 12.0,
    "path_smooth": 1.5,
    "verbose": -1,
    "seed": SEED,
    "deterministic": True,
    "force_col_wise": True,
}
LGBM_NUM_BOOST = 10000
LGBM_EARLY_STOP = 300

# RankFormer hyperparameters. Canonical config — reaches val NDCG@5 ≈ 0.413.
# Key choices: (i) 2-layer MLP per-item encoder so the transformer sees
# non-linear feature combinations (otherwise it plateaus at ~0.40, since a
# single linear projection loses the interactions GBDTs capture in splits);
# (ii) no positional embedding — the candidates in a search are a set, not
# a sequence, and a learned pos-emb biases the model on row order;
# (iii) cosine LR with warm restarts (2 cycles, peak 6e-4) to escape the
# sub-optimal basin that a single-cycle schedule with smaller LR settles in.
RF_PARAMS = {
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 2,
    "ffn_dim": 256,
    "dropout": 0.1,
    "prop_id_emb": 32,
    "dest_id_emb": 16,
    "country_emb": 8,
    "site_emb": 8,
    "numeric_mlp": True,
    "fuse_mlp": True,
    "use_pos_emb": False,
}
RF_TRAIN = {
    "batch_sessions": 128,
    "lr": 6e-4,
    "weight_decay": 1e-2,
    "warmup_steps": 400,
    "epochs": 12,
    "n_cycles": 2,
    "max_list_len": 40,
    "softmax_temperature": 1.0,
    "listnet_weight": 1.0,
    "pairwise_lambda": 0.3,
    "lambdarank_weight": 0.0,
    "log_every_n_batches": 500,
}
