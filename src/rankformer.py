"""RankFormer — a transformer that scores items conditional on the other
candidates in the same search session.

Architecture (compact, fits comfortably on MPS):
    numeric_features (N_num)  -> Linear(N_num, d_model)
    each categorical feature  -> Embedding(cardinality, d_cat)  -> concat
    fused                     -> LayerNorm -> Linear -> d_model
    + positional encoding (per-item learned position 0..max_list_len-1)
    -> TransformerEncoder (n_layers, n_heads)
    -> Linear(d_model, 1)  -> per-item score

Reference: Buyl, De Backer & Bonastre, "RankFormer: Listwise Learning-to-Rank
Using Listwide Labels", SIGIR 2023 (Booking.com). The implementation here
adapts the core listwise-attention idea; we use a ListNet + pairwise hinge
hybrid loss rather than the listwide-label trick from the paper because our
labels are pointwise (click/book per item).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RankFormerConfig:
    n_num: int
    cat_cardinalities: dict[str, int]
    cat_dims: dict[str, int]
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ffn_dim: int = 256
    dropout: float = 0.1
    max_list_len: int = 40
    numeric_mlp: bool = False     # use 2-layer MLP for numeric features
    fuse_mlp: bool = False        # deeper post-concat fuse block
    use_pos_emb: bool = True      # learned positional embedding


class RankFormer(nn.Module):
    def __init__(self, cfg: RankFormerConfig):
        super().__init__()
        self.cfg = cfg
        d_cat_total = sum(cfg.cat_dims.values())
        self.cat_cols = list(cfg.cat_dims.keys())
        self.embeddings = nn.ModuleDict({
            c: nn.Embedding(cfg.cat_cardinalities[c], cfg.cat_dims[c], padding_idx=0)
            for c in self.cat_cols
        })
        if cfg.numeric_mlp:
            self.numeric_proj = nn.Sequential(
                nn.Linear(cfg.n_num, cfg.d_model * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model * 2, cfg.d_model),
            )
        else:
            self.numeric_proj = nn.Linear(cfg.n_num, cfg.d_model)
        if cfg.fuse_mlp:
            self.fuse = nn.Sequential(
                nn.LayerNorm(cfg.d_model + d_cat_total),
                nn.Linear(cfg.d_model + d_cat_total, cfg.d_model * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model * 2, cfg.d_model),
                nn.GELU(),
            )
        else:
            self.fuse = nn.Sequential(
                nn.LayerNorm(cfg.d_model + d_cat_total),
                nn.Linear(cfg.d_model + d_cat_total, cfg.d_model),
                nn.GELU(),
            )
        self.use_pos_emb = cfg.use_pos_emb
        if cfg.use_pos_emb:
            self.pos_emb = nn.Embedding(cfg.max_list_len, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(
        self,
        x_num: torch.Tensor,    # (B, L, N_num)
        x_cat: dict[str, torch.Tensor],  # each (B, L) integer
        mask: torch.Tensor,     # (B, L) bool, True for real items
    ) -> torch.Tensor:          # (B, L)
        B, L, _ = x_num.shape

        emb_parts = [self.numeric_proj(x_num)]
        for c in self.cat_cols:
            emb_parts.append(self.embeddings[c](x_cat[c]))
        # Concat along feature dim, then fuse to d_model.
        h = self.fuse(torch.cat(emb_parts, dim=-1))

        # Add positional embedding (optional — items are a set when use_pos_emb=False).
        if self.use_pos_emb:
            pos = torch.arange(L, device=h.device).unsqueeze(0).expand(B, -1)
            h = h + self.pos_emb(pos)

        # Transformer expects key_padding_mask True for *padding* (to ignore).
        kpm = ~mask
        h = self.encoder(h, src_key_padding_mask=kpm)

        scores = self.head(h).squeeze(-1)  # (B, L)
        # Mask padded positions with a large negative for downstream softmax/sort.
        scores = scores.masked_fill(~mask, -1e4)
        return scores


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def listnet_loss(
    scores: torch.Tensor,       # (B, L)
    relevances: torch.Tensor,   # (B, L) integer 0/1/5
    mask: torch.Tensor,         # (B, L) bool
    temperature: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy of softmax(scores) vs softmax(relevance/τ), per-query mean.

    Padded positions are nullified by setting both scores and relevance gains
    to -1e4 and -inf respectively before softmax.
    """
    rel_g = (2.0 ** relevances.float() - 1.0)
    rel_g = rel_g.masked_fill(~mask, 0.0)
    target_logits = rel_g / temperature
    # Where rel sum is zero (no positives in query), produce a uniform target
    # over real items so we don't divide by zero — but contribute zero loss
    # by masking those queries out.
    has_pos = rel_g.sum(dim=1) > 0

    # Softmax over real items only.
    masked_scores = scores.masked_fill(~mask, -1e9)
    log_p = F.log_softmax(masked_scores, dim=1)
    masked_target = target_logits.masked_fill(~mask, -1e9)
    target = F.softmax(masked_target, dim=1)
    ce = -(target * log_p).sum(dim=1)
    if has_pos.any():
        return ce[has_pos].mean()
    return ce.sum() * 0.0


def pairwise_hinge(
    scores: torch.Tensor,       # (B, L)
    relevances: torch.Tensor,   # (B, L)
    mask: torch.Tensor,         # (B, L)
    margin: float = 1.0,
) -> torch.Tensor:
    """Hinge loss on pairs (i, j) within the same query where rel_i > rel_j."""
    B, L = scores.shape
    rel = relevances.float()
    s_i = scores.unsqueeze(2)            # (B, L, 1)
    s_j = scores.unsqueeze(1)            # (B, 1, L)
    r_i = rel.unsqueeze(2)
    r_j = rel.unsqueeze(1)
    valid = mask.unsqueeze(2) & mask.unsqueeze(1)
    pair_pos = valid & (r_i > r_j)
    if pair_pos.sum() == 0:
        return scores.sum() * 0.0
    diff = s_j - s_i + margin
    loss = F.relu(diff)[pair_pos]
    return loss.mean()


def lambdarank_loss(
    scores: torch.Tensor,       # (B, L)
    relevances: torch.Tensor,   # (B, L) integer 0/1/5
    mask: torch.Tensor,         # (B, L) bool
    sigma: float = 1.0,
) -> torch.Tensor:
    """LambdaRank loss: NDCG-weighted pairwise logistic loss.

    For each pair (i, j) in the same query with rel_i > rel_j:
        loss_ij = log(1 + exp(-σ(s_i - s_j))) * |ΔNDCG_ij|
    where ΔNDCG_ij is the change in NDCG if items i and j were swapped in the
    current ranking implied by `scores`. Ranks/discounts are detached so
    gradients flow only through the score differences.
    """
    B, L = scores.shape
    rel = relevances.float()
    gain = (2.0 ** rel) - 1.0                              # (B, L)

    with torch.no_grad():
        safe = scores.masked_fill(~mask, -1e9)
        sorted_idx = torch.argsort(safe, dim=1, descending=True)
        ranks = torch.argsort(sorted_idx, dim=1).float()    # (B, L) 0-indexed
        discount = 1.0 / torch.log2(ranks + 2.0)            # (B, L)

        rel_sorted, _ = torch.sort(rel, dim=1, descending=True)
        gain_sorted = (2.0 ** rel_sorted) - 1.0
        pos = torch.arange(1, L + 1, device=scores.device).float()
        ideal_disc = 1.0 / torch.log2(pos + 1.0)            # (L,)
        idcg = (gain_sorted * ideal_disc.unsqueeze(0)).sum(dim=1).clamp(min=1e-9)

        g_i = gain.unsqueeze(2)                             # (B, L, 1)
        g_j = gain.unsqueeze(1)                             # (B, 1, L)
        d_i = discount.unsqueeze(2)
        d_j = discount.unsqueeze(1)
        delta_dcg = torch.abs(g_i - g_j) * torch.abs(d_i - d_j)
        delta_ndcg = delta_dcg / idcg.view(-1, 1, 1)        # (B, L, L)

        r_i = rel.unsqueeze(2)
        r_j = rel.unsqueeze(1)
        valid_pair = (r_i > r_j) & mask.unsqueeze(2) & mask.unsqueeze(1)
        weight = (delta_ndcg * valid_pair.float())          # (B, L, L)

    if not bool(valid_pair.any()):
        return scores.sum() * 0.0

    s_i = scores.unsqueeze(2)
    s_j = scores.unsqueeze(1)
    diff = sigma * (s_i - s_j)
    log_loss = F.softplus(-diff)                            # (B, L, L)

    weighted = (log_loss * weight).sum()
    denom = weight.sum().clamp(min=1e-9)
    return weighted / denom
