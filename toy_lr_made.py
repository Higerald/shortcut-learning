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

def parse_float(s: str) -> float:
    return float(s)

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
    """Accept keys '(1,1)', '1,1', [1,1], (1,1); return Dict[(int,int), int]."""
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
    B: float = 1.5             # strength of spurious attr a
    rho: float = 0.9           # corr between y and a in majority bucket
    pmaj: float = 0.9          # fraction of samples in majority buckets
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

def apply_regime_indices(groups: np.ndarray, regime: str, rng: np.random.Generator) -> np.ndarray:
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

# ---------------- MADE (Gaussian) ----------------
def create_masks(D: int, hidden_dims: List[int], ncls:int=2, seed:int=0):
    """
    Inputs degrees: data dims (1..D), class one-hot gets degree 0 (always visible).
    Hidden degrees: Uniform{1,...,D-1}.
    """
    rng = np.random.default_rng(seed)
    in_deg = np.concatenate([np.arange(1, D+1), np.zeros(ncls, dtype=int)], axis=0)
    degrees = [in_deg]
    for h in hidden_dims:
        degrees.append(rng.integers(low=1, high=D, size=h))
    return degrees

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.register_buffer("mask", torch.ones(out_features, in_features))
    def set_mask(self, mask: torch.Tensor): self.mask.data.copy_(mask)
    def forward(self, x): return nn.functional.linear(x, self.weight * self.mask, self.bias)

class MADEGaussian(nn.Module):
    def __init__(self, D, hidden, n_layers, ncls=2, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.D, self.ncls = D, ncls
        hidden_dims = [hidden]*n_layers
        self.degrees = create_masks(D, hidden_dims, ncls, seed)
        dims = [D+ncls] + hidden_dims + [2*D]
        self.net = nn.ModuleList([MaskedLinear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.act = nn.ReLU(); self._make_masks()
    def _make_masks(self):
        masks=[]; L = len(self.net)
        for l in range(L):
            in_deg = self.degrees[l]
            if l + 1 < len(self.degrees):
                out_deg = self.degrees[l+1]
                m = (out_deg[:, None] >= in_deg[None, :]).astype(np.float32)   # Eq.12
            else:
                out_deg = np.repeat(np.arange(1, self.D+1), 2)                 # (2D,)
                m = (out_deg[:, None] >  in_deg[None, :]).astype(np.float32)   # Eq.13 strict
            masks.append(torch.tensor(m))
        for layer, m in zip(self.net, masks): layer.set_mask(m)
    def forward(self, z, y_onehot):
        h = torch.cat([z, y_onehot], dim=1)
        for lyr in self.net[:-1]: h = self.act(lyr(h))
        out = self.net[-1](h)
        mu, log_sig = out[:,:self.D], torch.clamp(out[:,self.D:], -7.0, 7.0)
        return mu, log_sig
    def nll(self, z, y_onehot):
        mu, log_sig = self.forward(z, y_onehot)
        inv_var = torch.exp(-2.0 * log_sig)
        nll = 0.5*math.log(2*math.pi) + log_sig + 0.5*(z-mu)**2 * inv_var
        return nll.sum(dim=1).mean()
    @torch.no_grad()
    def log_prob(self, z, y_onehot):
        mu, log_sig = self.forward(z, y_onehot)
        inv_var = torch.exp(-2.0 * log_sig)
        logp = -0.5*math.log(2*math.pi) - log_sig - 0.5*(z-mu)**2 * inv_var
        return logp.sum(dim=1)

# ---------------- Disc-MLP (discriminative) ----------------
class DiscMLP(nn.Module):
    def __init__(self, D_in: int, hidden: int, n_layers: int, ncls: int = 2, seed: int = 0):
        super().__init__(); torch.manual_seed(seed)
        dims = [D_in] + [hidden]*n_layers + [ncls]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        return self.net(z)

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
def eval_disc_mlp(model, Z, y_pm1, groups, device="cpu"):
    device = torch.device(device); model.eval()
    z = torch.from_numpy(Z.astype(np.float32)).to(device)
    yhat = model(z).argmax(dim=1).cpu().numpy()
    ytrue01 = (y_pm1>0).astype(int)
    acc_overall = float((yhat==ytrue01).mean())
    accs, worst, best, wgs, bgs = group_acc(ytrue01, yhat, groups)
    return acc_overall, accs, worst, best, wgs, bgs

@torch.no_grad()
def eval_density(model, Z, y_pm1, groups, device="cpu"):
    device = torch.device(device); model.eval()
    z = torch.from_numpy(Z.astype(np.float32)).to(device)
    ytrue01 = (y_pm1>0).astype(int)
    y0 = torch.zeros(z.size(0),2,device=device); y0[:,0]=1
    y1 = torch.zeros(z.size(0),2,device=device); y1[:,1]=1
    s0 = model.log_prob(z, y0).cpu().numpy()
    s1 = model.log_prob(z, y1).cpu().numpy()
    prior_pos = float(ytrue01.mean()); prior_neg = 1.0 - prior_pos
    scores = np.stack([s0+np.log(prior_neg+1e-12), s1+np.log(prior_pos+1e-12)], axis=1)
    yhat = np.argmax(scores, axis=1)
    acc_overall = float((yhat==ytrue01).mean())
    accs, worst, best, wgs, bgs = group_acc(ytrue01, yhat, groups)
    return acc_overall, accs, worst, best, wgs, bgs

def _metric_from_eval(eval_tuple, key):
    acc_overall, _, acc_worst, _, _, _ = eval_tuple
    return acc_overall if key == "val_acc_overall" else acc_worst

def train_classifier(
    model, ds_tr, regime, epochs=30, lr=1e-3, batch_size=256,
    device="cpu", val_tuple=None, select_metric="val_acc_overall"
):
    device = torch.device(device); model.to(device)
    loader = _make_loader(ds_tr, regime, batch_size)
    opt = optim.Adam(model.parameters(), lr=lr)

    best = -1e9; best_state=None; best_epoch=-1
    best_val_overall=None; best_val_worst=None

    for ep in range(1, epochs+1):
        model.train()
        for z, yb, _ in loader:
            z = z.to(device); yb = yb.to(device)
            loss = nn.functional.cross_entropy(model(z), yb)
            opt.zero_grad(); loss.backward(); opt.step()

        if val_tuple is not None:
            Zval, yval, gval = val_tuple
            v = eval_disc_mlp(model, Zval, yval, gval, device=device)
            metric = _metric_from_eval(v, select_metric)
            if metric > best:
                best = metric
                best_state = deepcopy(model.state_dict())
                best_epoch = ep
                best_val_overall, best_val_worst = v[0], v[2]

    if best_state is not None:
        model.load_state_dict(best_state)

    return dict(best_epoch=best_epoch, best_val_metric=best,
                best_val_acc_overall=best_val_overall, best_val_acc_worst=best_val_worst,
                selected_by=select_metric)

def train_density(
    model, ds_tr, regime, epochs=30, lr=1e-3, batch_size=256, seed=0,
    device="cpu", val_tuple=None, select_metric="val_acc_overall"
):
    device = torch.device(device); model.to(device)
    loader = _make_loader(ds_tr, regime, batch_size)
    opt = optim.Adam(model.parameters(), lr=lr)

    best = -1e9; best_state=None; best_epoch=-1
    best_val_overall=None; best_val_worst=None

    for ep in range(1, epochs+1):
        model.train()
        for z, yb, _ in loader:
            z = z.to(device)
            y_onehot = torch.zeros(z.size(0), 2, device=device)
            y_onehot[torch.arange(z.size(0)), yb] = 1.0
            loss = model.nll(z, y_onehot)
            opt.zero_grad(); loss.backward(); opt.step()

        if val_tuple is not None:
            Zval, yval, gval = val_tuple
            v = eval_density(model, Zval, yval, gval, device=device)
            metric = _metric_from_eval(v, select_metric)
            if metric > best:
                best = metric
                best_state = deepcopy(model.state_dict())
                best_epoch = ep
                best_val_overall, best_val_worst = v[0], v[2]

    if best_state is not None:
        model.load_state_dict(best_state)

    return dict(best_epoch=best_epoch, best_val_metric=best,
                best_val_acc_overall=best_val_overall, best_val_acc_worst=best_val_worst,
                selected_by=select_metric)

# ---------------- parameter counts ----------------
def params_disc_exact(D: int, H: int, L: int, C: int = 2) -> int:
    return (D*H + H) + (L-1)*(H*H + H) + (H*C + C)

def params_made_raw(D: int, H: int, L: int, C: int = 2) -> int:
    return ((D+C)*H + H) + (L-1)*(H*H + H) + (H*(2*D) + 2*D)

def params_made_expected_effective(D: int, H: int, L: int, C: int = 2) -> int:
    weff = H*(C + 0.5*D) + 0.5*(L-1)*(H*H) + H*D
    beff = L*H + 2*D
    return int(round(weff + beff))

# ---------------- val split ----------------
def split_val_from_id_test(test_id: Dict[str, np.ndarray], frac: float, seed: int):
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

# ---------------- main experiment ----------------
def run_experiment(
    out_csv="results_made_vs_disc_identity.csv",
    seeds=[0],
    d_list=[16, 64, 256],      # D = 2 + d
    n_train=2000,
    n_test=8000,
    counts_by_group_train: Optional[Dict[Tuple[int,int],int]] = None,
    head_hidden_list=[32, 128],
    head_layers_list=[1, 2],
    regimes=["erm","downsample","reweight","upsample"],
    id_cfg: DataGenConfig = None,
    ood_cfg: DataGenConfig = None,
    epochs=100, lr=1e-3, batch_size=256,         # <-- default 100 epochs now
    device="cpu",
    val_from_id_frac: float = 0.2,               # <-- fraction of ID test used for validation
    select_metric: str = "val_acc_overall"       # or "val_acc_worst"
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
            test_ood = generate_dataset(n_test, cfg_ood)

            # Split validation from ID test
            val_tuple, test_id_final = split_val_from_id_test(test_id_full, val_from_id_frac, seed)
            Zid_val, yid_val, gid_val = (val_tuple if val_tuple is not None else (None, None, None))
            Zid, yid, gid = test_id_final

            # identity backbone
            Ztr = train["X"].astype(np.float32)
            D_in = Ztr.shape[1]

            id_cfg_dict  = {f"id_{k}": v for k,v in asdict(cfg_id).items()}
            ood_cfg_dict = {f"ood_{k}": v for k,v in asdict(cfg_ood).items()}
            counts_json = json.dumps(_keys_to_str(counts_by_group_train)) if counts_by_group_train is not None else None

            for H in head_hidden_list:
                for L in head_layers_list:
                    for regime in regimes:
                        rng = np.random.default_rng(seed+42)
                        idx = apply_regime_indices(train["g"], regime, rng)
                        Ztr_reg = Ztr[idx]; ytr_reg = train["y"][idx]; gtr_reg = train["g"][idx]
                        ds_tr  = FeaturesDataset(Ztr_reg, ytr_reg, gtr_reg)

                        # effective group counts actually seen
                        eff_counts = {}
                        for ga in [(+1,+1),(+1,-1),(-1,+1),(-1,-1)]:
                            eff_counts[str(ga)] = int(sum([tuple(gg)==ga for gg in gtr_reg]))
                        eff_counts_json = json.dumps(eff_counts)

                        # ---- Disc-MLP ----
                        disc = DiscMLP(D_in=D_in, hidden=H, n_layers=L, ncls=2, seed=seed)
                        disc_info = train_classifier(
                            disc, ds_tr, regime, epochs=epochs, lr=lr, batch_size=batch_size,
                            device=device,
                            val_tuple=(Zid_val, yid_val, gid_val) if val_tuple is not None else None,
                            select_metric=select_metric
                        )
                        d_id_acc, d_id_g, d_id_w, d_id_b, d_id_wgs, d_id_bgs = eval_disc_mlp(disc, Zid, yid, gid, device=device)
                        d_od_acc, d_od_g, d_od_w, d_od_b, d_od_wgs, d_od_bgs = eval_disc_mlp(disc, test_ood["X"], test_ood["y"], test_ood["g"], device=device)
                        disc_params = params_disc_exact(D_in, H, L, 2)

                        meta = dict(
                            seed=seed, d_noise=d_noise, D_in=D_in,
                            head_hidden=H, head_layers=L, regime=regime,
                            n_train_requested=ntr_req, n_train_effective=int(len(Ztr_reg)), n_test=n_test,
                            num_epochs_trained=epochs, lr=lr, batch_size=batch_size, device=device,
                            counts_by_group_train=counts_json, counts_by_group_effective=eff_counts_json,
                            val_from_id_frac=val_from_id_frac, select_metric=select_metric
                        )
                        meta = {**meta, **id_cfg_dict, **ood_cfg_dict}

                        rows.append({**meta,"model":"disc_mlp","split":"ID",
                                     "acc_overall":d_id_acc,"acc_worst_group":d_id_w,"acc_best_group":d_id_b,
                                     "worst_groups":json.dumps(d_id_wgs),"best_groups":json.dumps(d_id_bgs),
                                     "params_raw":disc_params,
                                     "selected_by":disc_info["selected_by"],
                                     "best_epoch":disc_info["best_epoch"],
                                     "best_val_metric":disc_info["best_val_metric"],
                                     "best_val_acc_overall":disc_info["best_val_acc_overall"],
                                     "best_val_acc_worst":disc_info["best_val_acc_worst"]})
                        rows.append({**meta,"model":"disc_mlp","split":"OOD",
                                     "acc_overall":d_od_acc,"acc_worst_group":d_od_w,"acc_best_group":d_od_b,
                                     "worst_groups":json.dumps(d_od_wgs),"best_groups":json.dumps(d_od_bgs),
                                     "params_raw":disc_params,
                                     "selected_by":disc_info["selected_by"],
                                     "best_epoch":disc_info["best_epoch"],
                                     "best_val_metric":disc_info["best_val_metric"],
                                     "best_val_acc_overall":disc_info["best_val_acc_overall"],
                                     "best_val_acc_worst":disc_info["best_val_acc_worst"]})

                        # ---- MADE (Gaussian) ----
                        made = MADEGaussian(D=D_in, hidden=H, n_layers=L, ncls=2, seed=seed)
                        made_info = train_density(
                            made, ds_tr, regime, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed,
                            device=device,
                            val_tuple=(Zid_val, yid_val, gid_val) if val_tuple is not None else None,
                            select_metric=select_metric
                        )
                        g_id_acc, g_id_g, g_id_w, g_id_b, g_id_wgs, g_id_bgs = eval_density(made, Zid, yid, gid, device=device)
                        g_od_acc, g_od_g, g_od_w, g_od_b, g_od_wgs, g_od_bgs = eval_density(made, test_ood["X"], test_ood["y"], test_ood["g"], device=device)

                        made_raw = params_made_raw(D_in, H, L, 2)
                        made_eff = params_made_expected_effective(D_in, H, L, 2)

                        rows += [
                            {**meta,"model":"gen_made","split":"ID",
                             "acc_overall":g_id_acc,"acc_worst_group":g_id_w,"acc_best_group":g_id_b,
                             "worst_groups":json.dumps(g_id_wgs),"best_groups":json.dumps(g_id_bgs),
                             "params_raw":made_raw, "params_effective_expected":made_eff,
                             "selected_by":made_info["selected_by"],
                             "best_epoch":made_info["best_epoch"],
                             "best_val_metric":made_info["best_val_metric"],
                             "best_val_acc_overall":made_info["best_val_acc_overall"],
                             "best_val_acc_worst":made_info["best_val_acc_worst"]},
                            {**meta,"model":"gen_made","split":"OOD",
                             "acc_overall":g_od_acc,"acc_worst_group":g_od_w,"acc_best_group":g_od_b,
                             "worst_groups":json.dumps(g_od_wgs),"best_groups":json.dumps(g_od_bgs),
                             "params_raw":made_raw, "params_effective_expected":made_eff,
                             "selected_by":made_info["selected_by"],
                             "best_epoch":made_info["best_epoch"],
                             "best_val_metric":made_info["best_val_metric"],
                             "best_val_acc_overall":made_info["best_val_acc_overall"],
                             "best_val_acc_worst":made_info["best_val_acc_worst"]},
                        ]

                        if len(rows) % 100 == 0:
                            pd.DataFrame(rows).to_csv(out_csv, index=False)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with {len(rows)} rows")

# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser(description="Disc-MLP vs MADE (Gaussian) with ID/OOD; best checkpoint via validation from ID test")
    p.add_argument("--out-csv", type=str, default="results_made_vs_disc_identity.csv")

    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--d-list", type=str, default="16,64")
    p.add_argument("--head-hidden-list", type=str, default="64,256")
    p.add_argument("--head-layers-list", type=str, default="1,2")
    p.add_argument("--regimes", type=str, default="erm,reweight,downsample,upsample")

    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test",  type=int, default=4000)
    p.add_argument("--epochs",  type=int, default=100)          # default 100
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device",  type=str, default="cpu")

    p.add_argument("--counts-by-group-train-json", type=str, default=None,
                   help='JSON dict of per-group counts, e.g. \'{"(1,1)":1000,"(-1,-1)":1000,"(1,-1)":200,"(-1,1)":200}\'')

    p.add_argument("--id-cfg-json",  type=str, default=None,
                   help="JSON to override DataGenConfig for ID")
    p.add_argument("--ood-cfg-json", type=str, default=None,
                   help="JSON to override DataGenConfig for OOD")

    p.add_argument("--val-from-id-frac", type=float, default=0.2,
                   help="fraction of ID test used as validation during training (0 disables)")
    p.add_argument("--select-metric", type=str, default="val_acc_overall",
                   choices=["val_acc_overall","val_acc_worst"],
                   help="validation metric that selects best checkpoint")

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
