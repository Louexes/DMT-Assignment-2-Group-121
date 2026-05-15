# Ranking Hotels for Expedia — Reproducible Pipeline

| Submission | # features | Val NDCG@5 | Kaggle public LB |
|---|---:|---:|---:|
| **Full model** (`submission_v4_lgbm_only.csv`) | **131** | **0.4205** | **0.41995** |
| Minimal model (`submission_top_40.csv`) | 40 | 0.4121 | 0.41245 |

Both are LightGBM LambdaRank models trained on the same engineered feature
matrix; the minimal model is a gain-based subset of the full one. A
RankFormer secondary model (val NDCG@5 ≈ 0.413 with the canonical config
described in §6.2) is included as the listwise neural baseline but is
**not** part of the final submission — LightGBM scores higher.

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

**Reproduce the 0.42 Kaggle submission** (run from this directory):

```bash
python scripts/01_prepare_data.py        # ~ 3 min  — CSV → Parquet
python scripts/03_build_features.py      # ~ 1 min  — engineer 131 features
python scripts/04_train_lightgbm.py      # ~13 min  — train full model
python scripts/07_make_submission.py     # ~ 5 min  — write submission CSV
```

Output: `outputs/submission_v4_lgbm_only.csv`. Submit via the Kaggle CLI to
avoid manual-edit corruption:

```bash
pip install kaggle
kaggle competitions submit -c dmt-2026-2nd-assignment \
  -f outputs/submission_v4_lgbm_only.csv -m ""
```

**Minimal-model variant.** After running the four commands above, also run:

```bash
python scripts/04b_train_lgbm_subset.py top_40
python scripts/07b_submit_subset.py    top_40
```

This produces `outputs/submission_top_40.csv` (40 features, val NDCG@5 =
0.4121).

**Optional secondary model.** RankFormer is in
`scripts/05_train_rankformer.py` (requires `torch`). With the canonical
config (MLP per-item encoder, no positional embedding, cosine LR with
two warm restarts, 12 epochs) it reaches val NDCG@5 ≈ 0.413 in roughly
15 min on an Apple-Silicon MPS device. It is not part of the chain that
produces the 0.42 Kaggle submission.

---

## Project layout

```
reproduce/
├── README.md                       # this file
├── requirements.txt
├── src/
│   ├── config.py                   # paths, hyperparameters, feature lists
│   ├── data_io.py                  # CSV → Parquet conversion
│   ├── features.py                 # feature engineering (bulk of the code)
│   ├── lightgbm_train.py           # LambdaRank training helpers
│   ├── ndcg.py                     # NDCG@k implementation (val metric)
│   ├── submission.py               # Kaggle CSV writer / verifier
│   ├── rankformer.py               # secondary model — not in final pipeline
│   └── rankformer_train.py
├── scripts/
│   ├── 01_prepare_data.py          # CSV → Parquet
│   ├── 03_build_features.py        # build labeled_feat / submit_feat
│   ├── 04_train_lightgbm.py        # full 131-feature LightGBM
│   ├── 04b_train_lgbm_subset.py    # any feature-subset LightGBM
│   ├── 05_train_rankformer.py      # optional, not in final pipeline
│   ├── 07_make_submission.py       # write Kaggle CSV from full model
│   └── 07b_submit_subset.py        # write Kaggle CSV from subset model
├── Data/                           # ← put train.csv / test.csv here
├── artifacts/                      # ← parquets, model files (generated)
└── outputs/                        # ← submission CSVs + metrics (generated)
```

**Notes.** Deterministic given `SEED=42` (same train/val split, fold
assignment, booster init). Total wall-clock on a 2024 M3 MacBook Pro:
~22 min. If LightGBM hangs at 0% CPU on macOS, set `OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` before running.

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
  *Our primary modelling tool.*
- **Buyl et al. [5]** — RankFormer. Listwise transformer ranker with an
  auxiliary listwide loss for empty-interaction sessions. *Trained as
  our listwise neural baseline; with an MLP per-item encoder, no
  positional embedding, and cosine LR with warm restarts it reaches
  $0.413$ — within $0.007$ of LightGBM but still slightly below.*
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

**Primary: LightGBM LambdaRank.** Tree paths act as interaction rules,
the model is scale-invariant, and it handles missing values natively
(important on a dataset where missingness is informative).

**Secondary: RankFormer.** Session-level transformer ranker.
Disadvantages: missing values propagate to NaN gradients, continuous
features need standardisation, and the architecture is data-hungry
relative to its parameter count.

---

## 5. Data Preparation

**Split.** $80/20$ group-aware split along `srch_id`, fixed seed,
shared between LightGBM and RankFormer.

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

### 6.1 LightGBM GBDT

The LambdaRank objective is optimised with the hyperparameters in
Table 1.

**Table 1.** LightGBM hyperparameters.

| Parameter | Value | Parameter | Value | Parameter | Value |
|---|---|---|---|---|---|
| n_estimators | 10000 | feature_fraction | 0.65 | lambda_l1 | 1.5 |
| learning_rate | 0.012 | bagging_fraction | 0.65 | lambda_l2 | 12 |
| num_leaves | 72 | bagging_freq | 4 | path_smooth | 1.5 |
| max_depth | 8 | min_data_in_leaf | 80 | early_stopping | 300 |

### 6.2 RankFormer

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

---

## 7. Evaluation

### 7.1 RankFormer

Validation NDCG@5 $\approx 0.413$ as a single model. Three lever
changes broke the apparent $\approx 0.40$ ceiling we initially hit:
(i) the MLP per-item encoder lets the transformer see the non-linear
feature combinations that a single linear projection collapses;
(ii) removing the learned positional embedding stops the model from
fitting row-order noise (the candidates are a set, not a sequence);
(iii) cosine LR with warm restarts forces the optimiser out of the
sub-optimal basin a low single-cycle LR settles into. Each lever in
isolation moves the curve by $\le +0.001$; the combination is
$+0.013$. LightGBM still scores higher overall ($0.420$ vs.\ $0.413$),
so we submit LightGBM as the final predictor; RankFormer is kept as
the listwise neural baseline required by the assignment.

### 7.2 LightGBM

The largest single improvement to the pipeline came from replacing
in-sample smoothed CTR/BTR features (which include each row's own
label) with a $k$-fold OOF target encoding. The leakage symptom was a
$+0.26$ train/validation NDCG@5 gap; after the fix the gap shrunk to
$+0.13$ and validation NDCG@5 rose by $\approx 0.04$.

**Table 2.** Final submissions.

| Model | # features | Val NDCG@5 | Kaggle public LB |
|---|---|---|---|
| Full | 131 | **0.4205** | **0.41995** |
| Minimal (top-40 by gain) | 40 | 0.4121 | 0.41245 |

The val-to-Kaggle gap is $\approx 0.0002$, confirming the $80/20$
group-aware split is well-calibrated.

**Top-10 features (by gain, full model).** `prop_country_id` (11.6%),
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

**RankFormer vs.\ LightGBM.** With the canonical config (MLP per-item
encoder, no positional embedding, cosine LR with restarts) RankFormer
reaches $0.413$ on validation — within $\approx 0.007$ of LightGBM
($0.420$). The remaining gap reflects the GBDT's natural advantage on
tabular data with strong hand-crafted features: tree splits capture
the kind of sharp non-linear thresholds that a soft attention
mechanism averages out.

**Remaining gap.** The top of the public leaderboard sits about
$0.005$–$0.010$ NDCG@5 above us. Inspection of feature importances
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

**Mitigation.** Prevalence-based instance re-weighting:
$w = 1/p_{\text{book}}(g)$ for booked rows of group $g$, $0.5 /
p_{\text{book}}(g)$ for click-only rows, $1$ otherwise.
Booked-independent rows get the heaviest weight ($\approx 38.7$) since
independent-hotel bookings are rarer.

**Results.**

| Metric | Pre | Post | Δ |
|---|---|---|---|
| Overall NDCG@5 | 0.4205 | 0.4037 | −0.0168 |
| NDCG@5, chain | 0.4531 | 0.4331 | −0.0200 |
| NDCG@5, indep | 0.4488 | 0.4346 | −0.0142 |
| **NDCG@5 gap** | **+0.0043** | **−0.0015** | **−0.0058** |
| Mean-rank gap (chain − indep) | −0.12 | +0.14 | +0.26 |
| Chain share in top-5 | 64.4% | 63.4% | −1.0pp |

All three fairness metrics improve simultaneously: the NDCG gap is
essentially closed (slightly over-corrected to $-0.002$), the rank gap
flips sign (mild over-correction in favour of
independents), and the top-5 share matches the booking baseline. The
cost is $0.017$ NDCG@5 overall — the textbook fairness/utility
trade-off. The Kaggle submission uses the un-weighted booster because
the competition is graded on NDCG@5 alone; the debiased model is kept
as an analytical artefact.

---

## 10. Deployment

A production deployment would need:

- **Storage.** Distributed object storage (e.g., Amazon S3 [1]) with
  hot data cached on serving nodes.
- **Training.** Two parallelism axes — data parallelism within a
  LightGBM run, and HP parallelism across runs.
- **Serving.** A $131$-feature GBDT (or the $40$-feature minimal
  variant) is cheap enough that feature retrieval, not inference, is
  the bottleneck.
- **Refresh.** Blue–green deployment on Kubernetes: spin up new
  serving nodes with the retrained model, drain old nodes once new
  ones are healthy, recycle.

---

## 11. What Did We Learn?

- **Diagnose leakage before adding features.** The largest single win
  came from realising a smoothed per-property CTR included each row's
  own label. The diagnostic was a train/validation gain mismatch.
- **Honest feature selection costs little.** Keeping the top $40$ of
  $131$ features by gain loses only $0.008$ NDCG@5 and removes a long
  tail of zero-gain `*_isnull` flags and redundant variants — useful
  for serving latency, not just leaderboard score.
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
