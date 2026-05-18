# Ranking Hotels for Expedia — Reproducible Pipeline

| Submission | # features | Val NDCG@5 | Kaggle public LB |
|---|---:|---:|---:|
| **Three-way weighted ensemble** (`submission_ensemble_v1055_v3020_rf025.csv`) | **131** | **0.4238** | **0.42355** |

Per-srch_id min-max normalisation of three rankers' scores — a primary
LightGBM LambdaRank booster (val NDCG@5 = 0.4205), a second LightGBM
booster at a different operating point and seed (val NDCG@5 = 0.4223),
and a RankFormer transformer ranker (val NDCG@5 = 0.4127) — linearly
blended with weights $w_{\text{v1}} = 0.55$, $w_{\text{v3}} = 0.20$,
$w_{\text{rf}} = 0.25$. All three rankers share the same 131-feature
input.

---

## Quick start

**Prerequisites.** Python 3.12 (3.10+ should also work); ~16 GB RAM and
~10 GB free disk; install dependencies with `pip install -r requirements.txt`.

**Data layout.** Place the competition files under `Data/` exactly as
Kaggle distributes them, renamed for convenience:

```
Data/
├── train.csv               # labelled training set, 4,958,347 rows
├── test.csv                # unlabelled test set, 4,959,183 rows
└── submission_sample.csv   # format reference
```

`train.csv` contains the `click_bool` / `booking_bool` / `position` columns;
`test.csv` does not. If your local files don't follow this convention, swap
the `LABELED_CSV` / `SUBMIT_CSV` aliases at the top of `src/config.py`.

**Reproduce the Kaggle submission** (run from this directory):

```bash
python scripts/01_prepare_data.py         # ~ 3 min  — CSV → Parquet
python scripts/03_build_features.py       # ~ 1 min  — engineer 131 features
python scripts/04_train_lightgbm.py       # ~13 min  — train LightGBM v1
python scripts/04b_train_lightgbm_v3.py   # ~14 min  — train LightGBM v3
python scripts/05_train_rankformer.py     # ~15 min  — train RankFormer (uses MPS)
python scripts/07_make_submission.py      # ~14 min  — write ensemble submission
```

If you don't have a CUDA / Apple-Silicon MPS device available, the
RankFormer step works on CPU too but takes substantially longer
(~40 min). The `torch>=2.5` dependency in `requirements.txt` is only
needed for that step.

Output: `outputs/submission_ensemble_v1055_v3020_rf025.csv`. Submit via
the Kaggle CLI to avoid manual-edit corruption:

```bash
pip install kaggle
kaggle competitions submit -c dmt-2026-2nd-assignment \
  -f outputs/submission_ensemble_v1055_v3020_rf025.csv -m ""
```

Or run everything end-to-end with the orchestrator:

```bash
python scripts/run_all.py                  # phases 1..7 in order
python scripts/run_all.py --from-step 4    # resume mid-way (e.g. after a crash)
python scripts/run_all.py --skip 4b        # skip the v3 booster
```

**Optional analyses.** EDA plots / summary stats and the bias detection +
mitigation experiment have their own phase scripts:

```bash
python scripts/02_eda.py            # writes outputs/figures/* and eda_stats.json
python scripts/06_bias_analysis.py  # pre/post group fairness on the val fold
```

---

## Project layout

```
reproduce/
├── README.md                       # this file
├── requirements.txt
├── src/
│   ├── config.py                   # paths, hyperparameters, feature lists, blend weights
│   ├── data_io.py                  # CSV → Parquet conversion
│   ├── eda.py                      # EDA plots + summary stats
│   ├── features.py                 # feature engineering (bulk of the code)
│   ├── lightgbm_train.py           # LambdaRank training helpers
│   ├── ndcg.py                     # NDCG@k implementation (val metric)
│   ├── bias.py                     # group-fairness metrics + reweighing
│   ├── submission.py               # Kaggle CSV writer / verifier
│   ├── rankformer.py               # transformer architecture + losses
│   └── rankformer_train.py         # RankFormer training loop
├── scripts/
│   ├── 01_prepare_data.py          # CSV → Parquet
│   ├── 02_eda.py                   # write outputs/figures/* and eda_stats.json
│   ├── 03_build_features.py        # build labeled_feat / submit_feat
│   ├── 04_train_lightgbm.py        # train LightGBM v1 (primary)
│   ├── 04b_train_lightgbm_v3.py    # train LightGBM v3 (diverse operating point)
│   ├── 05_train_rankformer.py      # train RankFormer (uses MPS / CUDA)
│   ├── 06_bias_analysis.py         # group fairness + reweighing (LGBM and ensemble)
│   ├── 07_make_submission.py       # write the three-way ensemble Kaggle CSV
│   └── run_all.py                  # phase orchestrator (1..7 in order)
├── Data/                           # ← put train.csv / test.csv here
├── artifacts/                      # ← parquets, model files (generated)
└── outputs/                        # ← submission CSVs + metrics (generated)
```

**Notes.** Deterministic given `SEED=42` (same train/val split, fold
assignment, booster init, RankFormer init). LGBM v3 uses its own seed
of 99. Total wall-clock on a 2024 M3 MacBook Pro: ~60 min end-to-end.
If LightGBM training hangs at 0% CPU on macOS, set `OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` before running.

The submission script uses `num_threads=8` for LightGBM predict
explicitly, so you can additionally export `OMP_NUM_THREADS=1` to keep
PyTorch single-threaded for the RankFormer predict (avoids a known
macOS libomp deadlock in `tensor.clone()`).

---
---

# Report

## 1. Introduction

The Expedia/ICDM 2013 dataset frames hotel ranking as a learning-to-rank
problem: given a search query (user attributes, dates, destination,
candidate property attributes, competitor pricing), produce an ordering
over the candidate hotels that places the most likely click/booking at
the top. The metric is NDCG@5 with relevance grades $\{0, 1, 5\}$ for
no-action / click-only / booked. Roughly five million result rows are
split across about $400{,}000$ unique searches.

---

## 2. Related Work

Per-source notes on what each reference contributes and how we use it.

- **Jahrer [7]** — Kaggle forum write-up of the disqualified ICDM 2013
  winner. Per-`prop_id` groupwise statistics on numerical features.
  *Adopted as our property-history feature family.*
- **Zhang [12]** — ICDM 2013 1st place. Estimated-position feature by
  averaging position per (destination, month) with datetime
  interpolation. *We use a simpler hierarchical empirical-Bayes
  smoother for the same construction.*
- **Wang & Kalousis [11]** — ICDM 2013 2nd place. Log-price
  normalisation across visitor- and property-country groupings.
  *We generalise to within-search rank / z-score / mean-diff /
  median-diff transforms.*
- **Liu et al. [8]** — ICDM 2013 4th place. Composite features with
  `prop_location_score2` (ratios, products), the "recent price delta",
  and worst-case country-quartile imputation. *We use the composites
  and the price-delta verbatim; for the imputation we substitute the
  country median.*
- **Bellucci et al. [2]** — VU 2021 course report. Evidence that
  LambdaMART is hard to beat with deeper architectures on this dataset.
  *Motivates our choice of LightGBM-LambdaRank as the primary model.*
- **Besseling et al. [3]** — VU 2023 course report. Position-importance
  feature: confidence-scaled inverse mean rank. *Included verbatim as
  one of our position proxies.*
- **Ke et al. [10]** — LightGBM. Library that provides LambdaRank,
  native missing-value handling, and explicit categorical features.
  *Our primary modelling tool. We train two LightGBM boosters at
  different operating points; both contribute to the final ensemble.*
- **Buyl et al. [5]** — RankFormer. Listwise transformer ranker.
  *Trained as the third leg of our ensemble; with an MLP per-item
  encoder, no positional embedding, and cosine LR with warm restarts
  it reaches $0.4127$ on validation. Blended at $0.25$ weight, it
  contributes residual independence that neither LightGBM booster
  captures.*
- **Huang et al. [6]** — TabTransformer. Background reference for
  transformer-based tabular models; motivates evaluating a transformer
  reranker.
- **Vaswani et al. [13]** — Attention is All You Need. Background
  reference for the self-attention construction behind TabTransformer
  and RankFormer.
- **Touhid [9]** — Didactic exposition of attention. Background only.
- **van Buuren & Groothuis-Oudshoorn [4]** — MICE imputation.
  *Considered for `prop_location_score2` but not used in the final
  pipeline.*
- **Amazon S3 [1]** — Object-storage reference for the deployment
  discussion.

---

## 3. Exploratory Data Analysis

The relevance grade is $\text{target} = 4 \cdot \text{booked} +
\text{clicked}$, giving $5$ if booked, $1$ if only clicked, $0$
otherwise.

**Key findings.**
1. Clicks and bookings are sparse: most searches contain $\leq 1$
   click, and bookings are unique within a search by construction.
2. Position bias is sharp. Sorted searches (`random_bool=0`) show
   click/booking decline faster than impression counts; randomised
   searches do not. Positions $5$ and $11$ appear systematically less
   often than their neighbours.
3. Missingness is structural rather than accidental. Competitor
   columns are $52$–$98\%$ missing, gross bookings $97\%$, visitor
   history $95\%$, affinity score $94\%$, destination distance $32\%$,
   `prop_location_score2` $22\%$. A missing `gross_bookings_usd` means
   no booking occurred, not that the field was lost.
4. `prop_location_score2` correlates with the target at $\approx
   +0.07$; `position` at $\approx -0.16$.
5. The test set introduces only $\approx 6\%$ new property identifiers
   ($7{,}773 / 137{,}211$), so train-fold per-property statistics
   transfer cleanly to the test set.
6. `price_usd` is strongly right-skewed and resembles a log-normal
   distribution.
7. Derived check-in dates (search date $+$ `srch_booking_window`) show
   a clear seasonal pattern.

---

## 4. Model Choices

Our final predictor is a per-srch_id min-max blended ensemble of three
rankers, all trained on the same $131$-feature input. The two model
*families* required by the assignment are LightGBM LambdaRank and a
listwise transformer (RankFormer); within the LightGBM family we train
two boosters at different operating points to add intra-family diversity.

**LightGBM LambdaRank (two boosters).** Tree paths act as interaction
rules, the model is scale-invariant, and it handles missing values
natively (important on a dataset where missingness is informative). The
second booster ("v3") is run with more capacity per tree, weaker
`min_data_in_leaf`, and a different seed, then regularised harder
through `lambda_l2` and `path_smooth` — predictions that correlate
with v1 but are not identical, which makes the blend additive.

**RankFormer.** Session-level transformer ranker. The attention layers
let each candidate's score depend on the other candidates in the same
query, complementing the two GBDTs' pointwise tree structure.
Continuous features need standardisation and missingness needs explicit
handling, but the listwise inductive bias picks up signal the GBDTs
miss.

---

## 5. Data Preparation

**Split.** $80/20$ group-aware split along `srch_id`, fixed seed,
shared between all three rankers.

### 5.1 Transformations

- **Zero-as-missing fix** on `prop_starrating` and
  `prop_log_historical_price` (where $0$ means "no information"). We
  do *not* apply this to `prop_review_score` because $0$ is a real
  value there.
- **Price clipping** at the $0.1\%$ / $99.9\%$ quantiles on the log
  scale, computed from the training fold. Several richer alternatives
  (per-property median replacement, per-stay vs.\ per-night detection)
  tested and all underperformed the simple global clip.
- **`prop_location_score2` imputation** via train-fold country median,
  with a global-median fallback for unseen countries.

### 5.2 Feature Engineering

- **Group-wise statistics by `prop_id`.** Mean, median, standard
  deviation, IQR, and skew of price and historical log-price; mean and
  std of star, review, and display position; log-count of appearances.
- **Group-wise statistics by `srch_destination_id`.** Analogous, plus
  destination-level location-score aggregates.
- **Frequency and target encoding.** `prop_id` and
  `srch_destination_id` are dropped as native categoricals (they
  overfit) and replaced with a frequency feature (`*_count_log`) and a
  $5$-fold out-of-fold target encoding (`*_te`). Folds are assigned by
  `srch_id`; the encoding target is the relevance grade $5\cdot
  \text{book} + \text{click\_only}$, smoothed toward the global mean
  with priors $50$ (property) and $200$ (destination). Validation and
  test rows use the full-train fit.
- **Position proxies.** `prop_avg_position` (train-fold mean rank per
  property); an `expected_position` keyed by (destination, month) and
  smoothed hierarchically toward the destination mean; a
  `position_importance` score $\bigl(1/(\bar r + 0.5)\bigr) \cdot
  n/(n+50)$ from Besseling et al. [3].
- **Within-search transforms** of price, log-price, star, review, both
  location scores, log historical price, combined quality, price per
  person, and the loc2/loc1 ratio: ordinal rank, $z$-score, difference
  from search mean, and difference from search median.
- **Composite features** from Liu et al. [8]: `loc2_over_loc1`,
  `log_price_x_loc2`, `recent_price_delta`,
  `price_vs_prop_mean_ratio`, `loc2_vs_dest_max_ratio`,
  `combined_quality`, `combined_location`.
- **Competitor aggregation.** Counts of cheaper / more expensive /
  unavailable competitors, mean percentage difference, and an
  "any cheaper competitor" boolean.
- **Date features.** Month, weekday, hour of search; month and weekday
  of derived check-in date.

Final feature count: $131$ model inputs.

---

## 6. Modeling

### 6.1 LightGBM v1 (primary booster)

The LambdaRank objective is optimised with the hyperparameters in
Table 1. This is the lighter operating point — small leaves, heavy
path-smoothing, conservative learning rate.

**Table 1.** LightGBM v1 hyperparameters.

| Parameter | Value | Parameter | Value | Parameter | Value |
|---|---|---|---|---|---|
| n_estimators | 10000 | feature_fraction | 0.65 | lambda_l1 | 1.5 |
| learning_rate | 0.012 | bagging_fraction | 0.65 | lambda_l2 | 12 |
| num_leaves | 72 | bagging_freq | 4 | path_smooth | 1.5 |
| max_depth | 8 | min_data_in_leaf | 80 | early_stopping | 300 |
| seed | 42 | | | | |

### 6.2 LightGBM v3 (diverse second booster)

Same objective, different operating point: more capacity per tree
(`num_leaves=128`, `max_depth=10`, `min_data_in_leaf=50`) and a
different seed, regularised through a much stronger `lambda_l2 = 25`
and `path_smooth = 2.5`. Standalone val NDCG@5 = $0.4223$. The
booster's predictions correlate with v1 but are not identical, which is
what makes the blend additive.

**Table 2.** LightGBM v3 hyperparameters.

| Parameter | Value | Parameter | Value | Parameter | Value |
|---|---|---|---|---|---|
| n_estimators | 10000 | feature_fraction | 0.55 | lambda_l1 | 0 |
| learning_rate | 0.012 | bagging_fraction | 0.75 | lambda_l2 | 25 |
| num_leaves | 128 | bagging_freq | 5 | path_smooth | 2.5 |
| max_depth | 10 | min_data_in_leaf | 50 | early_stopping | 300 |
| seed | 99 | | | | |

### 6.3 RankFormer

Per-item encoder: a two-layer MLP over standardised numeric features
($n_\text{num} \to 2 d_\text{model} \to d_\text{model}$ with GELU and
dropout $0.1$) concatenated with categorical embeddings, then a
LayerNorm-Linear-GELU-Linear-GELU fuse block. The candidates in a
search are treated as a *set* — no learned positional embedding is
added before the transformer. Encoder: two self-attention layers,
four heads, FFN dim $256$, hidden dim $128$, dropout $0.1$. `prop_id`
and `srch_destination_id` get embedding tables of dim $32$ and $16$
respectively. Optimisation: AdamW, weight decay $10^{-2}$, $400$
warm-up steps, then a cosine learning-rate schedule with two warm
restarts (peak $\text{lr} = 6 \cdot 10^{-4}$, decaying to $0$ at the
end of each half), $12$ epochs total. Loss is ListNet cross-entropy
plus a pairwise hinge ($\lambda = 0.3$) on book-vs-non-book pairs.

### 6.4 Ensemble

The three models are blended at score level. For each query, each score
column is linearly rescaled to $[0, 1]$ by min-max normalisation within
that `srch_id`, putting the three heterogeneous score scales on a
common axis. The final score is

$$
s_{\text{ens}}(i) = w_{\text{v1}} \cdot \tilde{s}_{\text{v1}}(i)
                  + w_{\text{v3}} \cdot \tilde{s}_{\text{v3}}(i)
                  + w_{\text{rf}} \cdot \tilde{s}_{\text{rf}}(i),
$$

with $(w_{\text{v1}}, w_{\text{v3}}, w_{\text{rf}}) = (0.55,\,0.20,\,0.25)$.
The weights were chosen by a grid sweep on the held-out validation
fold subject to the constraint $w_{\text{v1}} \ge w_{\text{v3}}$ and
$w_{\text{rf}} \in [0.10, 0.40]$; the surface is flat in a small
neighbourhood of the optimum.

---

## 7. Evaluation

LightGBM v1 alone scores $0.4205$ NDCG@5 on validation; LightGBM v3
alone scores $0.4223$; RankFormer alone scores $0.4127$. Per-query
min-max blending with the weights $(0.55,\,0.20,\,0.25)$ reaches
$0.4238$ — a $+0.0015$ lift over the better LightGBM alone, and a
$+0.0033$ lift over the primary booster. The three rankers are not
perfectly redundant on this dataset: each carries independent
information within each query.

**Table 3.** Submission.

| Model | # features | Val NDCG@5 | Kaggle public LB |
|---|---|---|---|
| **Ensemble** ($w_{\text{v1}}=0.55,\,w_{\text{v3}}=0.20,\,w_{\text{rf}}=0.25$) | 131 | **0.4238** | **0.42355** |

The val-to-Kaggle gap is $\approx -0.0002$ — close to zero,
confirming that the $80/20$ group-aware split is well-calibrated.

**Top-10 LightGBM v1 features (by gain).** `prop_country_id` (11.6%),
`visitor_location_country_id` (10.1%), `prop_te` (5.5%),
`recent_price_delta` (5.4%), `prop_location_score2` (4.9%),
`loc2_over_loc1` (3.0%), `random_bool` (2.5%),
`price_vs_prop_mean_ratio` (2.4%), `log_price_x_loc2` (2.0%),
`prop_avg_position` (1.9%). Together $\approx 55\%$ of total gain.
Country IDs, the OOF target encoding, and the Liu-style composite
features dominate; date features and competitor aggregates do not
appear in the top forty.

---

## 8. Discussion

**Leakage diagnosis.** A smoothed groupwise CTR/BTR built on the full
training fold leaks the row's own label into its own encoding. The
diagnostic is a large train/validation gain mismatch: a stable feature
produces similar gains on both splits. We caught this only after
comparing our gap to a smaller one reported elsewhere on the same
dataset. After replacing the leaky encoding with $k$-fold OOF, the
training NDCG@5 dropped (the model could no longer memorise) and
validation rose by $\approx 0.04$.

**Why three rankers and not two.** Individually, the two GBDTs both
outscore the transformer ($0.4205$ / $0.4223$ vs.\ $0.4127$ on
validation), as is typical on tabular data with strong hand-crafted
features. Each, however, makes errors the others do not: the two
LightGBM boosters disagree on borderline candidates because they were
trained with different capacity and a different seed, and the
transformer's listwise attention adds a *different* axis of disagreement
because it scores each candidate conditional on the rest of the list.
The weighted blend with $(0.55,\,0.20,\,0.25)$ adds $+0.0015$ over the
better LightGBM alone — the three rankers' errors are correlated but
not identical, and the residual independence is worth ensembling.

**Remaining gap.** The top of the public leaderboard sits about
$0.005$–$0.007$ NDCG@5 above us. Inspection of feature importances
suggests further gains would come from richer position-bias features
(an interpolated estimated position, more sophisticated rank-based
proxies) rather than from changing model class.

---

## 9. Bias Detection & Mitigation

**Hypothesis.** The ranker may systematically over-promote chain
hotels at the expense of local independents.

**Sensitive attribute.** `prop_brand_bool` (chain vs.\ independent).
The two groups are roughly balanced ($\approx 67\%$ chain).

**Detection metrics.**
1. Per-group NDCG@5 (conditional on the booked item's group).
2. Mean-rank exposure on queries containing both types.
3. Chain share in the top $5$ vs.\ chain share among bookings
   ($\approx 63.3\%$).

**Mitigation.** Prevalence-based instance re-weighting on the
LightGBM v1 training set:
$w = 1/p_{\text{book}}(g)$ for booked rows of group $g$, $0.5 /
p_{\text{book}}(g)$ for click-only rows, $1$ otherwise.
Booked-independent rows get the heaviest weight ($\approx 38.7$) since
independent-hotel bookings are rarer.

**Results.** We evaluate fairness on the deployed model — the
three-way ensemble — so that the bias numbers describe what actually
ships. Pre-mitigation uses the unweighted LightGBM v1 leg; post-mitigation
swaps in the reweighed (debiased) v1 leg, leaving the LightGBM v3 and
RankFormer legs unchanged.

**Table 4.** Bias on the deployed ensemble (LGBM v1 + v3 + RF).

| Metric | Pre | Post | Δ |
|---|---|---|---|
| Overall NDCG@5 | 0.4238 | 0.4200 | −0.0038 |
| NDCG@5, chain | 0.4563 | 0.4521 | −0.0042 |
| NDCG@5, indep | 0.4541 | 0.4505 | −0.0036 |
| **NDCG@5 gap (chain − indep)** | **+0.0022** | **+0.0017** | **−0.0005** |
| Mean-rank gap (indep − chain) | +0.083 | −0.025 | −0.108 |
| Chain share in top-5 vs base | +1.1pp | +0.7pp | −0.4pp |

The ensemble is already close to group-fair before any intervention:
the NDCG@5 gap is only $+0.0022$. RankFormer is partly responsible —
its listwise scoring is less biased than the GBDT alone on this
dataset, so adding it to the blend halves the inherent disparity of
LightGBM-alone (Table 5). Reweighing the LightGBM v1 leg of the
ensemble further tightens the NDCG gap to $+0.0017$, flips the
rank-gap sign so independents are no longer ranked worse on average,
and reduces the top-5 over-representation of chains. The cost is
$0.0038$ NDCG@5 — substantially smaller than the $0.017$ cost
incurred when LightGBM is the whole model.

**Table 5.** For comparison: bias on LightGBM v1 alone.

| Metric | Pre | Post | Δ |
|---|---|---|---|
| Overall NDCG@5 | 0.4205 | 0.4037 | −0.0168 |
| **NDCG@5 gap** | **+0.0043** | **−0.0015** | **−0.0058** |
| Mean-rank gap (indep − chain) | +0.116 | −0.141 | −0.257 |

The Kaggle submission uses the un-weighted ensemble because the
competition is graded on NDCG@5 alone; the debiased booster is kept as
an analytical artefact.

---

## 10. Deployment

A production deployment would need:

- **Storage.** Distributed object storage (e.g., Amazon S3 [1]) with
  hot data cached on serving nodes.
- **Training.** Two parallelism axes — data parallelism within each
  model (the LightGBM boosters and RankFormer can train concurrently
  on the same feature matrix), and HP parallelism across runs.
- **Serving.** The three ranker scores can be computed in parallel and
  combined by a thin blend service. The LightGBM boosters run on CPU;
  the RankFormer transformer runs on a small GPU or on CPU with
  per-query batching. Feature retrieval is usually the bottleneck.
- **Refresh.** Blue–green deployment on Kubernetes: spin up new
  serving nodes with the retrained models, drain old nodes once new
  ones are healthy, recycle.

---

## 11. What Did We Learn?

- **Diagnose leakage before adding features.** The largest single win
  came from realising a smoothed per-property CTR included each row's
  own label. The diagnostic was a train/validation gain mismatch.
- **Ensembling pays even within a model family.** The two LightGBM
  boosters score within $0.002$ NDCG@5 of each other but contribute
  $\approx 0.001$ over the better of the two in the blend — different
  operating points on the same algorithm leave enough residual
  independence to be worth combining.
- **Listwise scoring helps fairness.** Adding the RankFormer leg to
  the blend halved the inherent group-NDCG gap relative to the LightGBM
  alone, and made the subsequent reweighing mitigation much cheaper in
  NDCG terms — its listwise attention does not lock in the chain-vs-
  independent disparity that the GBDT inherits from the data.
- **Choose a sensible sensitive attribute.** High-cardinality
  attributes (e.g., user country) leave per-group estimates too noisy
  for re-weighting to help. A clean low-cardinality binary makes the
  fairness/utility trade-off visible.

---

## References

1. Amazon S3. <https://aws.amazon.com/pm/serv-s3/>
2. Bellucci, T., Hu, Q., Lin, C.C. *On the Ranking of Expedia Search
   Results using LambdaMART.* VU Amsterdam, 2021.
3. Besseling, J., Petrus, M., Fenne, S. *Ranking Expedia hotel search
   queries by optimizing the NDCG@5.* VU Amsterdam, 2023.
4. van Buuren, S., Groothuis-Oudshoorn, K. *mice: Multivariate
   Imputation by Chained Equations in R.* J. Stat. Softw. 45(3), 2011.
5. Buyl, M., Missault, P., Sondag, P.-A. *RankFormer: Listwise
   Learning-to-Rank Using Listwide Labels.* SIGIR 2023.
   arXiv:2306.05808.
6. Huang, X., Khetan, A., Cvitkovic, M., Karnin, Z. *TabTransformer:
   Tabular Data Modeling Using Contextual Embeddings.* 2020.
   arXiv:2012.06678.
7. Jahrer, M. *Personalize Expedia Hotel Searches — ICDM 2013
   Discussions.* Kaggle Forum, 2013.
8. Liu, X. et al. *Combination of Diverse Ranking Models for
   Personalized Expedia Hotel Searches.* 2013. arXiv:1311.7679.
9. Touhid. *The (surprisingly simple!) math behind the transformer
   attention mechanism.* Medium, 2024.
10. Ke, G. et al. *LightGBM: A Highly Efficient Gradient Boosting
    Decision Tree.* NeurIPS 2017.
11. Wang, J., Kalousis, A. *Personalize Expedia Hotel Searches — 2nd
    Place.* ICDM 2013.
12. Zhang, O. *Personalized Expedia Hotel Searches — 1st Place.* ICDM
    2013.
13. Vaswani, A. et al. *Attention is All You Need.* NeurIPS 2017.
