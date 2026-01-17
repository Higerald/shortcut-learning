#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, json, random, argparse, ast
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ----------------------- utils -----------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def parse_int_list(s: str) -> List[int]:
    if s is None or s == "": return []
    return [int(x.strip()) for x in s.split(",") if x.strip()!=""]

def parse_str_list(s: str) -> List[str]:
    if s is None or s == "": return []
    return [x.strip() for x in s.split(",") if x.strip()!=""]

def parse_json_or_none(s: Optional[str]):
    if s is None: return None
    s = s.strip()
    if s == "": return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)

def normalize_counts_dict(d):
    if d is None: return None
    out = {}
    for k, v in d.items():
        if isinstance(k, (list, tuple)) and len(k)==2:
            key = (int(k[0]), int(k[1]))
        else:
            ks = str(k).strip()
            if ks.startswith("(") and ks.endswith(")"): ks = ks[1:-1]
            if ks.startswith("[") and ks.endswith("]"): ks = ks[1:-1]
            y,a = ks.split(",")
            key = (int(y.strip()), int(a.strip()))
        out[key] = int(v)
    return out

def _keys_to_str(d: Optional[dict]) -> Optional[dict]:
    if d is None: return None
    return {str(k): int(v) for k, v in d.items()}

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ---------------- synthetic data with spurious attr ----------------
@dataclass
class DataGenConfig:
    d_noise: int = 64          # total D = 2 + d_noise
    sigma_core: float = 0.15
    sigma_spu: float = 0.10
    sigma_noise: float = 1.0
    core_scale: float = 0.5
    B: float = 1.5            # strength of spurious attr a
    # B: float = 0.0           # strength of spurious attr a
    rho: float = 0.9
    pmaj: float = 0.9
    p_pos: float = 0.5
    random_state: int = 0

def sample_groups_counts_default(n: int, pmaj: float) -> Dict[Tuple[int,int], int]:
    nmaj = int(round(pmaj * n)); nmin = n - nmaj
    return {
        (+1,+1): nmaj // 2,
        (-1,-1): nmaj - nmaj // 2,
        (+1,-1): nmin // 2,
        (-1,+1): nmin - nmin // 2,
    }

def generate_dataset(n: int, cfg: DataGenConfig,
                     counts_by_group: Optional[Dict[Tuple[int,int], int]] = None) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.random_state)
    counts = counts_by_group if counts_by_group is not None else sample_groups_counts_default(n, cfg.pmaj)

    Xs, ys, aspu, groups = [], [], [], []
    for (y,a), k in counts.items():
        if k <= 0: continue
        x_core  = rng.normal(loc=cfg.core_scale*y, scale=cfg.sigma_core, size=(k,1))
        x_spu   = rng.normal(loc=cfg.B*a,       scale=cfg.sigma_spu,  size=(k,1))
        x_noise = rng.normal(loc=0.0,           scale=cfg.sigma_noise, size=(k, cfg.d_noise))
        Xg = np.concatenate([x_core, x_spu, x_noise], axis=1).astype(np.float32)
        Xs.append(Xg)
        ys.append(np.full(k, y, dtype=np.int32))
        aspu.append(np.full(k, a, dtype=np.int32))
        groups += [(y,a)]*k
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    a = np.concatenate(aspu, axis=0)
    g = np.array(groups, dtype=object)
    idx = rng.permutation(len(y))
    return dict(X=X[idx], y=y[idx], a=a[idx], g=g[idx])

# ---------------- datasets / regimes ----------------
class FeaturesDataset(Dataset):
    def __init__(self, Z: np.ndarray, y_pm1: np.ndarray, groups):
        self.Z = torch.from_numpy(Z.astype(np.float32))
        self.y = torch.from_numpy((y_pm1 > 0).astype(np.int64))
        self.groups = groups
    def __len__(self): return self.Z.shape[0]
    def __getitem__(self, i):
        return self.Z[i], self.y[i], self.groups[i]

def build_group_weights(groups: np.ndarray) -> np.ndarray:
    labels = np.array([str(g) for g in groups], dtype=np.str_)
    unique, counts = np.unique(labels, return_counts=True)
    freq = dict(zip(unique, counts))
    w = np.array([1.0 / freq[str(g)] for g in groups], dtype=np.float32)
    w *= (len(groups) / w.sum())
    return w

def apply_regime_indices(groups: np.ndarray, regime: str, rng: np.random.default_rng) -> np.ndarray:
    idx_all = np.arange(len(groups))
    if regime == "erm": return idx_all
    buckets = {}
    for i,g in enumerate(groups): buckets.setdefault(tuple(g), []).append(i)
    sizes = {g: len(v) for g,v in buckets.items()}
    min_sz, max_sz = min(sizes.values()), max(sizes.values())
    if regime == "downsample":
        chosen=[]
        for _,ix in buckets.items():
            chosen += list(rng.choice(ix, size=min_sz, replace=False)) if len(ix)>min_sz else ix
        return np.array(chosen, dtype=np.int64)
    if regime == "upsample":
        chosen=[]
        for _,ix in buckets.items():
            chosen += list(ix) + (list(rng.choice(ix, size=max_sz-len(ix), replace=True)) if len(ix)<max_sz else [])
        return np.array(chosen, dtype=np.int64)
    if regime == "reweight":
        return idx_all
    return idx_all

# ---------------- Transformer backbone ----------------
class TabularTransformerBackbone(nn.Module):
    """D scalars -> D tokens. Optional label token for conditional modeling.
       mask_mode="causal" for AR, "none" for classification."""
    def __init__(self, D, d_model=128, nhead=8, num_layers=2, dim_feedforward=512,
                 dropout=0.1, use_label_token=False, ncls=2, mask_mode="causal"):
        super().__init__()
        self.D = D
        self.use_label_token = use_label_token
        self.ncls = ncls
        self.d_model = d_model
        self.mask_mode = mask_mode

        self.in_proj = nn.Linear(1, d_model)
        if use_label_token:
            self.label_proj = nn.Linear(ncls, d_model)

        npos = D + (1 if use_label_token else 0)
        self.pos_emb = nn.Embedding(npos, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.layernorm_out = nn.LayerNorm(d_model)

    def _strict_causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=0)

    def forward(self, x, y_onehot=None):
        B, D = x.shape
        assert D == self.D

        xf = x.view(B, D, 1)
        tok_x = self.in_proj(xf)  # (B, D, d_model)

        toks = []
        if self.use_label_token:
            assert y_onehot is not None
            ytok = self.label_proj(y_onehot).unsqueeze(1)  # (B,1,d_model)
            toks.append(ytok)
        toks.append(tok_x)

        H = torch.cat(toks, dim=1)      # (B, T, d_model)
        T = H.size(1)

        pos_ids = torch.arange(T, device=H.device).unsqueeze(0)
        H = H + self.pos_emb(pos_ids)

        attn_mask = None
        if self.mask_mode == "causal":
            attn_mask = self._strict_causal_mask(T, H.device)
            if self.use_label_token: attn_mask[:, 0] = False  # always allow attending to label

        H = self.encoder(H, mask=attn_mask)
        return self.layernorm_out(H)

# ---------------- Heads ----------------
class DiscTransformerClassifier(nn.Module):
    def __init__(self, D_in, ncls=2, d_model=128, nhead=8, num_layers=2,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.backbone = TabularTransformerBackbone(
            D=D_in, d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            use_label_token=False, ncls=ncls, mask_mode="none"
        )
        self.out = nn.Linear(d_model, ncls)
    def forward(self, x):
        Z = self.backbone(x)      # (B, D, d_model)
        h = Z.mean(dim=1)         # mean pool
        return self.out(h)

class GenTransformerGaussian(nn.Module):
    """Conditional AR Gaussian with teacher-forced shift."""
    def __init__(self, D_in, ncls=2, d_model=128, nhead=8, num_layers=2,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.D = D_in
        self._LOG2PI = math.log(2.0 * math.pi)
        self.backbone = TabularTransformerBackbone(
            D=D_in, d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            use_label_token=True, ncls=ncls, mask_mode="causal"
        )
        self.head = nn.Linear(d_model, 2)  # per token -> [mu, log_sigma]
    def _split_params(self, params):
        mu = params[..., 0]
        log_sig = params[..., 1].clamp(-7.0, 7.0)
        return mu, log_sig
    def forward_params(self, x, y_onehot):
        B, D = x.shape
        x_shift = torch.zeros_like(x)
        x_shift[:, 1:] = x[:, :-1]
        H = self.backbone(x_shift, y_onehot=y_onehot)  # (B,1+D,d_model)
        Hx = H[:, 1:, :]
        params = self.head(Hx)
        return self._split_params(params)
    def nll(self, x, y_onehot):
        mu, log_sig = self.forward_params(x, y_onehot)
        inv_var = torch.exp(-2.0 * log_sig)
        nll = 0.5*self._LOG2PI + log_sig + 0.5*(x - mu)**2 * inv_var
        return nll.sum(dim=1).mean()
    @torch.no_grad()
    def log_prob(self, x, y_onehot):
        mu, log_sig = self.forward_params(x, y_onehot)
        inv_var = torch.exp(-2.0 * log_sig)
        logp = -0.5*self._LOG2PI - log_sig - 0.5*(x - mu)**2 * inv_var
        return logp.sum(dim=1)

# ---------------- collate ----------------
def collate_keep_groups(batch):
    zs, ys, gs = zip(*batch)
    Zb = torch.stack(zs, dim=0)
    yb = torch.stack(ys, dim=0)
    return Zb, yb, list(gs)

# ---------------- training / eval ----------------
def _make_loader(ds_tr, regime, batch_size):
    if regime=="reweight":
        w = build_group_weights(ds_tr.groups)
        return DataLoader(ds_tr, batch_size=batch_size,
                          sampler=WeightedRandomSampler(w, len(w), replacement=True),
                          collate_fn=collate_keep_groups)
    else:
        return DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_keep_groups)

def group_acc(y_true01: np.ndarray, y_pred01: np.ndarray, groups: np.ndarray):
    accs={}
    for g in [(+1,+1),(+1,-1),(-1,+1),(-1,-1)]:
        m = np.array([tuple(gg)==g for gg in groups])
        accs[str(g)] = float((y_true01[m]==y_pred01[m]).mean()) if m.sum()>0 else np.nan
    valid = {k:v for k,v in accs.items() if not np.isnan(v)}
    if not valid: return accs, np.nan, np.nan, [], []
    vals=list(valid.values()); worst, best = min(vals), max(vals)
    return accs, worst, best, [k for k,v in valid.items() if v==worst], [k for k,v in valid.items() if v==best]

@torch.no_grad()
def eval_disc_transformer(model, Z, y_pm1, groups, device="cpu", eval_bs=512):
    device = torch.device(device)
    model.eval()
    N = Z.shape[0]
    ytrue01 = (y_pm1 > 0).astype(int)
    yhat = np.empty(N, dtype=np.int64)

    with torch.inference_mode():
        for s in range(0, N, eval_bs):
            e = min(s + eval_bs, N)
            z = torch.from_numpy(Z[s:e].astype(np.float32)).to(device, non_blocking=True)
            logits = model(z)
            yhat[s:e] = logits.argmax(dim=1).cpu().numpy()
            # free chunk asap
            del z, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    acc_overall = float((yhat == ytrue01).mean())
    accs, worst, best, wgs, bgs = group_acc(ytrue01, yhat, groups)
    return acc_overall, accs, worst, best, wgs, bgs


@torch.no_grad()
def eval_gen_transformer(model, Z, y_pm1, groups, device="cpu", eval_bs=512):
    device = torch.device(device)
    model.eval()
    N = Z.shape[0]
    ytrue01 = (y_pm1 > 0).astype(int)
    prior_pos = float(ytrue01.mean()); prior_neg = 1.0 - prior_pos
    log_prior = (math.log(max(prior_neg, 1e-12)), math.log(max(prior_pos, 1e-12)))
    yhat = np.empty(N, dtype=np.int64)

    with torch.inference_mode():
        for s in range(0, N, eval_bs):
            e = min(s + eval_bs, N)
            z = torch.from_numpy(Z[s:e].astype(np.float32)).to(device, non_blocking=True)
            # y0 = torch.zeros(z.size(0), 2, device=device); y0[:, 0] = 1
            # y1 = torch.zeros(z.size(0), 2, device=device); y1[:, 1] = 1
            y0 = torch.zeros(z.size(0), 2, device=device, dtype=z.dtype); y0[:, 0] = 1
            y1 = torch.zeros(z.size(0), 2, device=device, dtype=z.dtype); y1[:, 1] = 1
            s0 = model.log_prob(z, y0).cpu().numpy()
            s1 = model.log_prob(z, y1).cpu().numpy()
            scores = np.stack([s0 + log_prior[0], s1 + log_prior[1]], axis=1)
            yhat[s:e] = np.argmax(scores, axis=1)
            del z, y0, y1
        if device.type == "cuda":
            torch.cuda.empty_cache()

    acc_overall = float((yhat == ytrue01).mean())
    accs, worst, best, wgs, bgs = group_acc(ytrue01, yhat, groups)
    return acc_overall, accs, worst, best, wgs, bgs


def _metric_from_eval(eval_tuple, key):
    acc_overall, _, acc_worst, _, _, _ = eval_tuple
    return acc_overall if key == "val_acc_overall" else acc_worst

def train_classifier_transformer(
    model, ds_tr, regime, epochs=30, lr=1e-3, batch_size=256, device="cpu",
    val_tuple=None, select_metric="val_acc_overall"
):
    device = torch.device(device); model.to(device)
    loader = _make_loader(ds_tr, regime, batch_size)
    opt = optim.Adam(model.parameters(), lr=lr)

    best = -1e9; best_state=None; best_epoch=-1; best_val_overall=None; best_val_worst=None

    for ep in range(1, epochs+1):
        model.train()
        for z, yb, _ in loader:
            z = z.to(device); yb = yb.to(device)
            loss = nn.functional.cross_entropy(model(z), yb)
            opt.zero_grad(); loss.backward(); opt.step()

        if val_tuple is not None:
            Zval, yval_pm1, gval = val_tuple
            v = eval_disc_transformer(model, Zval, yval_pm1, gval, device=device)
            metric = _metric_from_eval(v, select_metric)
            if metric > best:
                best = metric; best_state = deepcopy(model.state_dict()); best_epoch = ep
                best_val_overall, best_val_worst = v[0], v[2]

    if best_state is not None: model.load_state_dict(best_state)
    return dict(best_epoch=best_epoch, best_val_metric=best,
                best_val_acc_overall=best_val_overall, best_val_acc_worst=best_val_worst,
                selected_by=select_metric)

def train_density_transformer(
    model, ds_tr, regime, epochs=30, lr=1e-3, batch_size=256, device="cpu",
    val_tuple=None, select_metric="val_acc_overall"
):
    device = torch.device(device); model.to(device)
    loader = _make_loader(ds_tr, regime, batch_size)
    opt = optim.Adam(model.parameters(), lr=lr)

    best = -1e9; best_state=None; best_epoch=-1; best_val_overall=None; best_val_worst=None

    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        count = 0
        for z, yb, _ in loader:
            z = z.to(device)
            # y_onehot = torch.zeros(z.size(0), 2, device=device)
            # y_onehot[torch.arange(z.size(0)), yb.to(device)] = 1.0
            y_onehot = torch.zeros(z.size(0), 2, device=device, dtype=z.dtype)
            y_onehot[torch.arange(z.size(0)), yb.to(device)] = 1.0

            loss = model.nll(z, y_onehot)
            opt.zero_grad(); loss.backward(); opt.step()

            running += loss.item() * z.size(0)
            count += z.size(0)

        avg_train_nll = running / max(1, count)
        # print(f"[ep {ep:03d}] gen avg_train_nll = {avg_train_nll:.6f}")

        if val_tuple is not None:
            Zval, yval_pm1, gval = val_tuple
            v = eval_gen_transformer(model, Zval, yval_pm1, gval, device=device)
            metric = _metric_from_eval(v, select_metric)
            if metric > best:
                best = metric; best_state = deepcopy(model.state_dict()); best_epoch = ep
                best_val_overall, best_val_worst = v[0], v[2]

    if best_state is not None: model.load_state_dict(best_state)
    return dict(best_epoch=best_epoch, best_val_metric=best,
                best_val_acc_overall=best_val_overall, best_val_acc_worst=best_val_worst,
                selected_by=select_metric)




# def train_density_transformer(
#     model, ds_tr, regime, epochs=30, lr=1e-3, batch_size=256, device="cpu",
#     val_tuple=None, select_metric="val_acc_overall"
# ):
#     device = torch.device(device); model.to(device)
#     loader = _make_loader(ds_tr, regime, batch_size)
#     opt = optim.Adam(model.parameters(), lr=lr)

#     best = -1e9; best_state=None; best_epoch=-1; best_val_overall=None; best_val_worst=None

#     for ep in range(1, epochs+1):
#         model.train()
#         for z, yb, _ in loader:
#             z = z.to(device)
#             y_onehot = torch.zeros(z.size(0), 2, device=device)
#             y_onehot[torch.arange(z.size(0)), yb.to(device)] = 1.0
#             loss = model.nll(z, y_onehot)
#             opt.zero_grad(); loss.backward(); opt.step()

#         if val_tuple is not None:
#             Zval, yval_pm1, gval = val_tuple
#             v = eval_gen_transformer(model, Zval, yval_pm1, gval, device=device)
#             metric = _metric_from_eval(v, select_metric)
#             if metric > best:
#                 best = metric; best_state = deepcopy(model.state_dict()); best_epoch = ep
#                 best_val_overall, best_val_worst = v[0], v[2]

#     if best_state is not None: model.load_state_dict(best_state)
#     return dict(best_epoch=best_epoch, best_val_metric=best,
#                 best_val_acc_overall=best_val_overall, best_val_acc_worst=best_val_worst,
#                 selected_by=select_metric)

# ---------------- parameter counts (heads only + total) ----------------
def params_disc_head(d_model: int, C: int = 2) -> int:
    return d_model * C + C

def params_gen_head(d_model: int) -> int:
    return d_model * 2 + 2

# ---------------- main experiment ----------------
def split_val_from_id_test(test_id: Dict[str,np.ndarray], frac: float, seed: int):
    """Deterministic split of ID test into (val, final test)."""
    Z = test_id["X"].astype(np.float32)
    y = test_id["y"]; g = test_id["g"]
    n = len(y)
    if frac <= 0 or n < 5:
        return None, (Z, y, g)
    rng = np.random.default_rng(seed + 2027)
    idx = rng.permutation(n)
    n_val = max(1, int(round(frac * n)))
    val_idx, test_idx = idx[:n_val], idx[n_val:]
    val = (Z[val_idx], y[val_idx], g[val_idx])
    test_final = (Z[test_idx], y[test_idx], g[test_idx])
    return val, test_final




############# Sanity Check ##############

def sanity_overfit_one_batch_gen_full(
    ds_tr,
    D_in: int,
    val_tuple=None,
    test_tuple=None,
    d_model: int = 128,
    nhead: int = 2,
    num_layers: int = 1,
    dim_ff: Optional[int] = None,
    dropout: float = 0.0,
    device: str = "cpu",
    steps: int = 400,
    lr: float = 1e-3,
    bs: int = 32,
    print_every: int = 50,
    save_csv: Optional[str] = None,
):
    """
    Train GenTransformerGaussian on ONE minibatch and print:
      - training NLL
      - train accuracy (on same batch)
      - val accuracy (optional)
      - test accuracy (optional)
    """
    if dim_ff is None:
        dim_ff = 4 * d_model

    device = torch.device(device)
    model = GenTransformerGaussian(
        D_in=D_in, ncls=2,
        d_model=d_model, nhead=nhead,
        num_layers=num_layers, dim_feedforward=dim_ff, dropout=dropout
    ).to(device)

    # 1 minibatch from training set
    loader = DataLoader(ds_tr, batch_size=bs, shuffle=True, collate_fn=collate_keep_groups)
    z, yb, _ = next(iter(loader))
    z = z.to(device); yb = yb.to(device)
    # y1h = torch.zeros(z.size(0), 2, device=device)
    # y1h[torch.arange(z.size(0)), yb] = 1.0

    y1h = torch.zeros(z.size(0), 2, device=device, dtype=z.dtype)
    y1h[torch.arange(z.size(0)), yb] = 1.0


    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []

    for t in range(steps):
        model.train()
        loss = model.nll(z, y1h)
        opt.zero_grad(); loss.backward(); opt.step()

        if (t + 1) % print_every == 0 or t == 0:
            # --- train acc (on that same batch)
            model.eval()
            with torch.no_grad():
                # s0 = model.log_prob(z, torch.tensor([[1,0]], device=device).repeat(z.size(0),1))
                # s1 = model.log_prob(z, torch.tensor([[0,1]], device=device).repeat(z.size(0),1))
                s0 = model.log_prob(z, torch.tensor([[1.0, 0.0]], device=device, dtype=z.dtype).repeat(z.size(0), 1))
                s1 = model.log_prob(z, torch.tensor([[0.0, 1.0]], device=device, dtype=z.dtype).repeat(z.size(0), 1))

                scores = torch.stack([s0, s1], dim=1).cpu().numpy()
                yhat = scores.argmax(axis=1)
                train_acc = float((yhat == yb.cpu().numpy()).mean())

                # optional val/test acc
                val_acc = None
                test_acc = None
                if val_tuple is not None:
                    val_acc = eval_gen_transformer(model, *val_tuple, device=device)[0]
                if test_tuple is not None:
                    test_acc = eval_gen_transformer(model, *test_tuple, device=device)[0]

            print_str = f"[{t+1:04d}] NLL={loss.item():.6f}, train_acc={train_acc:.3f}"
            if val_acc is not None: print_str += f", val_acc={val_acc:.3f}"
            if test_acc is not None: print_str += f", test_acc={test_acc:.3f}"
            print(print_str)

            hist.append(dict(step=t+1, nll=float(loss.item()),
                             train_acc=train_acc, val_acc=val_acc, test_acc=test_acc))

    if save_csv:
        pd.DataFrame(hist).to_csv(save_csv, index=False)
        print(f"Saved {save_csv} ({len(hist)} rows)")
    return hist



# def sanity_overfit_one_batch_gen(
#     ds_tr,
#     D_in: int,
#     d_model: int = 128,
#     nhead: int = 2,
#     num_layers: int = 1,
#     dim_ff: Optional[int] = None,
#     dropout: float = 0.0,
#     device: str = "cpu",
#     steps: int = 400,
#     lr: float = 1e-3,
#     bs: int = 32,
#     print_every: int = 50,
# ):
#     """
#     Train GenTransformerGaussian on ONE minibatch and print NLL trending down.
#     Returns dict with the per-step losses list.
#     """
#     if dim_ff is None:
#         dim_ff = 4 * d_model

#     device = torch.device(device)
#     model = GenTransformerGaussian(
#         D_in=D_in, ncls=2,
#         d_model=d_model, nhead=nhead,
#         num_layers=num_layers, dim_feedforward=dim_ff, dropout=dropout
#     ).to(device)

#     loader = DataLoader(ds_tr, batch_size=bs, shuffle=True, collate_fn=collate_keep_groups)
#     z, yb, _ = next(iter(loader))
#     z = z.to(device); yb = yb.to(device)
#     y1h = torch.zeros(z.size(0), 2, device=device); y1h[torch.arange(z.size(0)), yb] = 1.0

#     opt = torch.optim.Adam(model.parameters(), lr=lr)

#     losses = []
#     for t in range(steps):
#         model.train()
#         loss = model.nll(z, y1h)
#         opt.zero_grad(); loss.backward(); opt.step()
#         losses.append(float(loss.item()))
#         if (t + 1) % print_every == 0:
#             print(f"[overfit {t+1:04d}/{steps}] NLL: {loss.item():.6f}")

#     return {"losses": losses, "batch_size": int(bs)}





def run_experiment(
    out_csv="results_transformer_made_vs_disc.csv",
    seeds=[0],
    d_list=[16, 64, 256],
    n_train=2000,
    n_test=8000,
    counts_by_group_train: Optional[Dict[Tuple[int,int],int]] = None,
    head_hidden_list=[128, 256],
    head_layers_list=[1, 2],
    regimes=["erm","downsample","reweight","upsample"],
    id_cfg: DataGenConfig = None,
    ood_cfg: DataGenConfig = None,
    epochs=25, lr=1e-3, batch_size=256,
    device="cpu",
    val_from_id_frac: float = 0.2,           # <---- fraction of ID test used for validation
    select_metric: str = "val_acc_overall"   # or "val_acc_worst"
):
    if id_cfg is None: id_cfg = DataGenConfig()
    if ood_cfg is None: ood_cfg = DataGenConfig()

    rows=[]
    for seed in seeds:
        set_seed(seed)
        for d_noise in d_list:
            cfg_id = DataGenConfig(**{**asdict(id_cfg)}); cfg_id.d_noise = d_noise; cfg_id.random_state=seed
            cfg_ood= DataGenConfig(**{**asdict(ood_cfg)}); cfg_ood.d_noise=d_noise; cfg_ood.random_state=seed+777

            ntr_req = sum(counts_by_group_train.values()) if counts_by_group_train is not None else n_train
            train = generate_dataset(ntr_req, cfg_id, counts_by_group=counts_by_group_train)
            test_id_full = generate_dataset(n_test, cfg_id)
            test_ood= generate_dataset(n_test, cfg_ood)

            # Split validation from ID test
            val_tuple, test_id_final = split_val_from_id_test(test_id_full, val_from_id_frac, seed)
            Zid_val, yid_val, gid_val = (val_tuple if val_tuple is not None else (None, None, None))
            Zid, yid, gid = test_id_final

            Ztr = train["X"].astype(np.float32); D_in = Ztr.shape[1]

            id_cfg_dict  = {f"id_{k}": v for k,v in asdict(cfg_id).items()}
            ood_cfg_dict = {f"ood_{k}": v for k,v in asdict(cfg_ood).items()}
            counts_json = json.dumps(_keys_to_str(counts_by_group_train)) if counts_by_group_train is not None else None

            for H in head_hidden_list:
                d_model = H
                nhead = max(1, d_model // 64)
                ff = 4 * d_model
                for L in head_layers_list:
                    num_layers = L
                    for regime in regimes:
                        rng = np.random.default_rng(seed+42)
                        idx = apply_regime_indices(train["g"], regime, rng)
                        Ztr_reg = Ztr[idx]; ytr_reg = train["y"][idx]; gtr_reg = train["g"][idx]
                        ds_tr  = FeaturesDataset(Ztr_reg, ytr_reg, gtr_reg)

                        eff_counts = {str(ga): int(sum([tuple(gg)==ga for gg in gtr_reg]))
                                      for ga in [(+1,+1),(+1,-1),(-1,+1),(-1,-1)]}
                        eff_counts_json = json.dumps(eff_counts)

                        # ---- Discriminative Transformer ----
                        disc = DiscTransformerClassifier(
                            D_in=D_in, ncls=2, d_model=d_model, nhead=nhead,
                            num_layers=num_layers, dim_feedforward=ff, dropout=0.1
                        )
                        disc_info = train_classifier_transformer(
                            disc, ds_tr, regime, epochs=epochs, lr=lr, batch_size=batch_size,
                            device=device,
                            val_tuple=(Zid_val, yid_val, gid_val) if val_tuple is not None else None,
                            select_metric=select_metric
                        )
                        d_id_acc, d_id_g, d_id_w, d_id_b, d_id_wgs, d_id_bgs = eval_disc_transformer(disc, Zid, yid, gid, device=device)
                        d_od_acc, d_od_g, d_od_w, d_od_b, d_od_wgs, d_od_bgs = eval_disc_transformer(disc, test_ood["X"], test_ood["y"], test_ood["g"], device=device)
                        disc_params_total = count_params(disc)
                        disc_params_head  = params_disc_head(d_model, 2)

                        meta = dict(
                            seed=seed, d_noise=d_noise, D_in=D_in,
                            head_hidden=d_model, head_layers=num_layers, nhead=nhead, dim_ff=ff,
                            regime=regime,
                            n_train_requested=ntr_req, n_train_effective=int(len(Ztr_reg)), n_test=n_test,
                            num_epochs_trained=epochs, lr=lr, batch_size=batch_size, device=device,
                            counts_by_group_train=counts_json, counts_by_group_effective=eff_counts_json,
                            val_from_id_frac=val_from_id_frac, select_metric=select_metric
                        )
                        meta = {**meta, **id_cfg_dict, **ood_cfg_dict}

                        rows.append({**meta,"model":"disc_transformer","split":"ID",
                                     "acc_overall":d_id_acc,"acc_worst_group":d_id_w,"acc_best_group":d_id_b,
                                     "worst_groups":json.dumps(d_id_wgs),"best_groups":json.dumps(d_id_bgs),
                                     "params_total":disc_params_total, "params_head_only":disc_params_head,
                                     "selected_by":disc_info["selected_by"],
                                     "best_epoch":disc_info["best_epoch"],
                                     "best_val_metric":disc_info["best_val_metric"],
                                     "best_val_acc_overall":disc_info["best_val_acc_overall"],
                                     "best_val_acc_worst":disc_info["best_val_acc_worst"]})
                        rows.append({**meta,"model":"disc_transformer","split":"OOD",
                                     "acc_overall":d_od_acc,"acc_worst_group":d_od_w,"acc_best_group":d_od_b,
                                     "worst_groups":json.dumps(d_od_wgs),"best_groups":json.dumps(d_od_bgs),
                                     "params_total":disc_params_total, "params_head_only":disc_params_head,
                                     "selected_by":disc_info["selected_by"],
                                     "best_epoch":disc_info["best_epoch"],
                                     "best_val_metric":disc_info["best_val_metric"],
                                     "best_val_acc_overall":disc_info["best_val_acc_overall"],
                                     "best_val_acc_worst":disc_info["best_val_acc_worst"]})

                        # ---- Generative Transformer ----
                        gen = GenTransformerGaussian(
                            D_in=D_in, ncls=2, d_model=d_model, nhead=nhead,
                            num_layers=num_layers, dim_feedforward=ff, dropout=0.1
                        )
                        gen_info = train_density_transformer(
                            gen, ds_tr, regime, epochs=epochs, lr=lr, batch_size=batch_size,
                            device=device,
                            val_tuple=(Zid_val, yid_val, gid_val) if val_tuple is not None else None,
                            select_metric=select_metric
                        )
                        g_id_acc, g_id_g, g_id_w, g_id_b, g_id_wgs, g_id_bgs = eval_gen_transformer(gen, Zid, yid, gid, device=device)
                        g_od_acc, g_od_g, g_od_w, g_od_b, g_od_wgs, g_od_bgs = eval_gen_transformer(gen, test_ood["X"], test_ood["y"], test_ood["g"], device=device)
                        gen_params_total = count_params(gen)
                        gen_params_head  = params_gen_head(d_model)

                        rows += [
                            {**meta,"model":"gen_transformer_gauss","split":"ID",
                             "acc_overall":g_id_acc,"acc_worst_group":g_id_w,"acc_best_group":g_id_b,
                             "worst_groups":json.dumps(g_id_wgs),"best_groups":json.dumps(g_id_bgs),
                             "params_total":gen_params_total,"params_head_only":gen_params_head,
                             "selected_by":gen_info["selected_by"],
                             "best_epoch":gen_info["best_epoch"],
                             "best_val_metric":gen_info["best_val_metric"],
                             "best_val_acc_overall":gen_info["best_val_acc_overall"],
                             "best_val_acc_worst":gen_info["best_val_acc_worst"]},
                            {**meta,"model":"gen_transformer_gauss","split":"OOD",
                             "acc_overall":g_od_acc,"acc_worst_group":g_od_w,"acc_best_group":g_od_b,
                             "worst_groups":json.dumps(g_od_wgs),"best_groups":json.dumps(g_od_bgs),
                             "params_total":gen_params_total,"params_head_only":gen_params_head,
                             "selected_by":gen_info["selected_by"],
                             "best_epoch":gen_info["best_epoch"],
                             "best_val_metric":gen_info["best_val_metric"],
                             "best_val_acc_overall":gen_info["best_val_acc_overall"],
                             "best_val_acc_worst":gen_info["best_val_acc_worst"]},
                        ]

                        if len(rows) % 100 == 0:
                            pd.DataFrame(rows).to_csv(out_csv, index=False)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with {len(rows)} rows")

# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser(description="Transformer backbone: best-checkpoint via validation from ID test")
    p.add_argument("--out-csv", type=str, default="results_transformer_made_vs_disc.csv")

    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--d-list", type=str, default="16,64")
    p.add_argument("--head-hidden-list", type=str, default="128,256")
    p.add_argument("--head-layers-list", type=str, default="1,2")
    p.add_argument("--regimes", type=str, default="erm,reweight,downsample,upsample")

    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test",  type=int, default=4000)
    p.add_argument("--epochs",  type=int, default=20)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device",  type=str, default="cpu")

    p.add_argument("--counts-by-group-train-json", type=str, default=None,
                   help='JSON dict of per-group counts, e.g. \'{"(1,1)":1000,"(-1,-1)":1000,"(1,-1)":200,"(-1,1)":200}\'')

    p.add_argument("--id-cfg-json",  type=str, default=None)
    p.add_argument("--ood-cfg-json", type=str, default=None)

    p.add_argument("--val-from-id-frac", type=float, default=0.2,
                   help="fraction of ID test used as validation during training (0 disables)")
    p.add_argument("--select-metric", type=str, default="val_acc_overall",
                   choices=["val_acc_overall","val_acc_worst"],
                   help="which validation metric picks the best checkpoint")
    

    p.add_argument("--sanity-gen", action="store_true",
               help="Run overfit-one-batch sanity test for the generative model and exit")
    p.add_argument("--sanity-steps", type=int, default=400)
    p.add_argument("--sanity-bs", type=int, default=32)
    p.add_argument("--sanity-dmodel", type=int, default=128)
    p.add_argument("--sanity-layers", type=int, default=1)
    p.add_argument("--sanity-nhead", type=int, default=2)
    p.add_argument("--sanity-dropout", type=float, default=0.0)
    p.add_argument("--sanity-save-csv", type=str, default="sanity_overfit_gen.csv")

    args = p.parse_args()

    seeds = parse_int_list(args.seeds)
    d_list = parse_int_list(args.d_list)
    head_hidden_list = parse_int_list(args.head_hidden_list)
    head_layers_list = parse_int_list(args.head_layers_list)
    regimes = parse_str_list(args.regimes)

    counts_raw = parse_json_or_none(args.counts_by_group_train_json)
    counts_by_group_train = normalize_counts_dict(counts_raw) if counts_raw is not None else None

    id_over = parse_json_or_none(args.id_cfg_json) or {}
    ood_over= parse_json_or_none(args.ood_cfg_json) or {}

    id_cfg = DataGenConfig(**{**asdict(DataGenConfig()), **id_over})
    ood_cfg= DataGenConfig(**{**asdict(DataGenConfig()), **ood_over})


    # ----- optional: sanity overfit one batch for the generative model -----



    if args.sanity_gen:
        # build dataset & splits
        d_list = parse_int_list(args.d_list)
        first_d = d_list[0] if len(d_list) > 0 else 64
        cfg = DataGenConfig(**{**asdict(DataGenConfig()), **(parse_json_or_none(args.id_cfg_json) or {})})
        cfg.d_noise = first_d
        cfg.random_state = parse_int_list(args.seeds)[0] if args.seeds else 0

        train = generate_dataset(args.n_train, cfg)
        test_id_full = generate_dataset(args.n_test, cfg)
        val_tuple, test_id_final = split_val_from_id_test(test_id_full, 0.2, cfg.random_state)
        Zid_val, yid_val, gid_val = val_tuple
        Zid, yid, gid = test_id_final

        ds_tr = FeaturesDataset(train["X"], train["y"], train["g"])
        D_in = train["X"].shape[1]

        sanity_overfit_one_batch_gen_full(
            ds_tr, D_in,
            val_tuple=(Zid_val, yid_val, gid_val),
            test_tuple=(Zid, yid, gid),
            d_model=args.sanity_dmodel,
            nhead=args.sanity_nhead,
            num_layers=args.sanity_layers,
            dropout=args.sanity_dropout,
            device=args.device,
            steps=args.sanity_steps,
            lr=args.lr,
            bs=args.sanity_bs,
            print_every=50,
            save_csv=args.sanity_save_csv
        )
        return

    # if args.sanity_gen:
    #     # Build a small training set using the first d in d_list
    #     d_list = parse_int_list(args.d_list)
    #     first_d = d_list[0] if len(d_list) > 0 else 64
    #     cfg = DataGenConfig(**{**asdict(DataGenConfig()), **(parse_json_or_none(args.id_cfg_json) or {})})
    #     cfg.d_noise = first_d
    #     cfg.random_state = parse_int_list(args.seeds)[0] if args.seeds else 0

    #     train = generate_dataset(args.n_train, cfg,
    #                             counts_by_group=normalize_counts_dict(parse_json_or_none(args.counts_by_group_train_json))
    #                             if args.counts_by_group_train_json else None)
    #     ds_tr = FeaturesDataset(train["X"], train["y"], train["g"])
    #     D_in = train["X"].shape[1]

    #     res = sanity_overfit_one_batch_gen(
    #         ds_tr, D_in,
    #         d_model=args.sanity_dmodel,
    #         nhead=args.sanity_nhead,
    #         num_layers=args.sanity_layers,
    #         dropout=args.sanity_dropout,
    #         device=args.device,
    #         steps=args.sanity_steps,
    #         lr=args.lr,
    #         bs=args.sanity_bs,
    #     )
    #     # save losses to CSV for a quick plot if you want
    #     pd.DataFrame({"step": np.arange(1, len(res["losses"])+1),
    #                 "nll": res["losses"]}).to_csv(args.sanity_save_csv, index=False)
    #     print(f"Saved {args.sanity_save_csv} with {len(res['losses'])} rows")
    #     return



    run_experiment(
        out_csv=args.out_csv,
        seeds=seeds,
        d_list=d_list,
        n_train=args.n_train,
        n_test=args.n_test,
        counts_by_group_train=counts_by_group_train,
        head_hidden_list=head_hidden_list,
        head_layers_list=head_layers_list,
        regimes=regimes,
        id_cfg=id_cfg,
        ood_cfg=ood_cfg,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        device=args.device,
        val_from_id_frac=args.val_from_id_frac,
        select_metric=args.select_metric,
    )

if __name__ == "__main__":
    main()
