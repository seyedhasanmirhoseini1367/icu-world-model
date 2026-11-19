"""
stage13b.py - Stage 13b: diagnostics at vasopressor starts, and start-weighted training.

Nothing in model.py or train.py is changed; this file only imports them.

Diagnostics (evaluation only):
  start_hour_stats   per predicted hour group (start hour +1, start hour +2, all other valid hours):
                     decoder sigma of MAP at the prior mean, 50-sample predictive spread of MAP,
                     KL(posterior || prior) summed over latent dims, residual (measured - predicted)
  start_fit_both     the Stage 13 start-fit (d = -3 .. +3 around real starts) with two point
                     predictions: decoder(prior mean) and the mean of 50 samples
  populations        real start hours vs the Part B moments: MAP level and 3-hour trend before,
                     earlier vasopressor use, starting dose

Training: the reconstruction term of the hours just after a start (start hour +1 and +2)
is weighted by START_WEIGHT; everything else is the act_delta model and the Stage 4 loss.
"""

import numpy as np
import torch

import confounded as C
import train as T
from model import WorldModel

START_WEIGHT = 10.0
N_SAMPLES = 50
MAP = C.MAP
NEE = C.NEE
D_FIT = list(range(-3, 4))


def start_masks(nee):
    """nee (B, T) -> bool (B, T-1) masks over predicted positions i (which predict hour i+1):
    plus1 where hour i+1 is a start hour + 1, plus2 where it is a start hour + 2."""
    prev = torch.cat([torch.zeros_like(nee[:, :1]), nee[:, :-1]], 1)
    start = (nee > 0) & (prev <= 0)                                   # (B, T): start at hour s
    plus1 = start[:, :-1]                                             # position s predicts hour s+1
    plus2 = torch.cat([torch.zeros_like(start[:, :1]), start[:, :-2]], 1)   # position s+1 predicts s+2
    return plus1, plus2


# ---------------------------------------------------------------- start-weighted training

class StartWeightedWorldModel(WorldModel):
    """Same parameters as WorldModel (checkpoints load as a plain WorldModel); its forward also
    returns the per-position reconstruction weight used by weighted_compute_loss."""

    def forward(self, batch, show=False):
        out = super().forward(batch, show)
        plus1, plus2 = start_masks(batch["a"][..., NEE])
        w = torch.ones_like(out["valid_target"])
        w[plus1 | plus2] = START_WEIGHT
        out["recon_weight"] = w
        return out


def weighted_compute_loss(out):
    """train.compute_loss with the reconstruction NLL of each predicted hour multiplied by its weight.
    The sum is still divided by the number of real predicted hours, as in train.compute_loss."""
    valid = out["valid_target"]
    n = valid.sum()
    w = out.get("recon_weight", torch.ones_like(valid))
    nll = T.gaussian_nll(out["x_target"], out["dec_mu"], out["dec_log_sigma"])
    recon = ((nll * out["mask_target"]).sum(-1) * valid * w).sum() / n
    kl_dims = T.gaussian_kl(out["post_mu"], out["post_log_sigma"], out["prior_mu"], out["prior_log_sigma"])
    kl_dims = kl_dims * valid[..., None]
    kl = kl_dims.sum(-1).sum() / n
    loss = recon + T.BETA_KL * torch.clamp(kl, min=T.FREE_BITS)
    return loss, recon, kl, kl_dims.sum(dim=(0, 1)) / n


def train_start_weighted(name, seed, loaders, model_kwargs, wandb_config):
    """train.train with the weighted loss: train.py's WorldModel and compute_loss are swapped for
    the duration of the call only, and restored afterwards (train.py itself is not edited)."""
    saved = T.WorldModel, T.compute_loss
    T.WorldModel, T.compute_loss = StartWeightedWorldModel, weighted_compute_loss
    try:
        return T.train(name, seed=seed, loaders=loaders, model_kwargs=model_kwargs,
                       wandb_config={**(wandb_config or {}), "start_weight": START_WEIGHT,
                                     "start_weight_hours": "start hour +1 and +2 (reconstruction only)"})
    finally:
        T.WorldModel, T.compute_loss = saved


# ---------------------------------------------------------------- diagnostics

@torch.no_grad()
def _batch_predictions(model, batch, n_samples=N_SAMPLES):
    """Prior one-step predictions for MAP (z-units): decoder mean and sigma at the prior mean,
    50-sample mean and spread; plus KL per position."""
    out = model(batch)
    h_now = out["h"][:, :-1]
    mu, ls = out["prior_mu"], out["prior_log_sigma"]
    dec_mu, dec_ls = model.decoder(torch.cat([mu, h_now], -1))                        # (B, T-1, V)
    z = mu + ls.exp() * torch.randn(n_samples, *mu.shape, device=mu.device)          # (S, B, T-1, Z)
    s_mu, s_ls = model.decoder(torch.cat([z, h_now.expand(n_samples, *h_now.shape)], -1))
    x = s_mu + s_ls.exp() * torch.randn_like(s_mu)                                    # (S, B, T-1, V)
    kl = T.gaussian_kl(out["post_mu"], out["post_log_sigma"], mu, ls).sum(-1)        # (B, T-1)
    return dict(point_mean=dec_mu[..., MAP], sigma_dec=dec_ls[..., MAP].exp(),
                point_samples=x[..., MAP].mean(0), sd_samples=x[..., MAP].std(0), kl=kl)


@torch.no_grad()
def start_hour_stats(model, ds, stats, device="cpu", batch_size=256, seed=0):
    torch.manual_seed(seed)
    std = float(stats["std"][MAP])
    model.eval()
    groups = ("start_plus1", "start_plus2", "other")
    acc = {g: dict(n=0, n_measured=0, sigma_dec=0.0, sd_samples=0.0, kl=0.0, resid=0.0, abs_resid=0.0) for g in groups}
    for lo in range(0, len(ds), batch_size):
        sl = slice(lo, lo + batch_size)
        batch = {k: getattr(ds, k)[sl].to(device) for k in ("x", "mask", "delta", "a", "valid", "idx")}
        p = _batch_predictions(model, batch)
        valid = batch["valid"][:, 1:] > 0
        measured = (batch["mask"][:, 1:, MAP] > 0) & valid
        plus1, plus2 = start_masks(batch["a"][..., NEE])
        resid = (batch["x"][:, 1:, MAP] - p["point_mean"]) * std
        for g, sel in (("start_plus1", plus1 & valid), ("start_plus2", plus2 & ~plus1 & valid),
                       ("other", valid & ~plus1 & ~plus2)):
            a = acc[g]
            a["n"] += int(sel.sum())
            a["sigma_dec"] += float((p["sigma_dec"][sel] * std).sum())
            a["sd_samples"] += float((p["sd_samples"][sel] * std).sum())
            a["kl"] += float(p["kl"][sel].sum())
            m = sel & measured
            a["n_measured"] += int(m.sum())
            a["resid"] += float(resid[m].sum())
            a["abs_resid"] += float(resid[m].abs().sum())
    out = {}
    for g, a in acc.items():
        n, nm = max(a["n"], 1), max(a["n_measured"], 1)
        out[g] = dict(n=a["n"], n_measured=a["n_measured"], sigma_dec_mmhg=a["sigma_dec"] / n,
                      predictive_sd_mmhg=a["sd_samples"] / n, kl_nats=a["kl"] / n,
                      residual_mmhg=a["resid"] / nm, abs_residual_mmhg=a["abs_resid"] / nm)
    for g in ("start_plus1", "start_plus2"):
        out[g]["sd_ratio_vs_other"] = out[g]["predictive_sd_mmhg"] / max(out["other"]["predictive_sd_mmhg"], 1e-9)
        out[g]["sigma_dec_ratio_vs_other"] = out[g]["sigma_dec_mmhg"] / max(out["other"]["sigma_dec_mmhg"], 1e-9)
    return out


@torch.no_grad()
def start_fit_both(model, ds, stats, device="cpu", batch_size=256, seed=0):
    """Stage 13 start-fit with two point predictions (decoder at the prior mean; 50-sample mean)."""
    torch.manual_seed(seed)
    std = float(stats["std"][MAP])
    starts = C.vaso_starts(ds.a[..., NEE].numpy())
    sums = {d: dict(n=0, measured=0.0, pred_mean=0.0, pred_samples=0.0) for d in D_FIT}
    model.eval()
    for lo in range(0, len(ds), batch_size):
        sl = slice(lo, lo + batch_size)
        batch = {k: getattr(ds, k)[sl].to(device) for k in ("x", "mask", "delta", "a", "valid", "idx")}
        p = _batch_predictions(model, batch)
        x = batch["x"][..., MAP].cpu().numpy()
        m = (batch["mask"][..., MAP] > 0).cpu().numpy() & (batch["valid"] > 0).cpu().numpy()
        pm, ps = p["point_mean"].cpu().numpy(), p["point_samples"].cpu().numpy()
        T_ = x.shape[1]
        rows, cols = np.nonzero(starts[sl])
        for d in D_FIT:
            j = cols + d
            ok = (j >= 1) & (j < T_)
            r, jj = rows[ok], j[ok]
            ok2 = m[r, jj] & m[r, jj - 1]
            r, jj = r[ok2], jj[ok2]
            s = sums[d]
            s["n"] += len(r)
            s["measured"] += float(((x[r, jj] - x[r, jj - 1]) * std).sum())
            s["pred_mean"] += float(((pm[r, jj - 1] - x[r, jj - 1]) * std).sum())
            s["pred_samples"] += float(((ps[r, jj - 1] - x[r, jj - 1]) * std).sum())
    n = [max(sums[d]["n"], 1) for d in D_FIT]
    return dict(d=D_FIT, n=[sums[d]["n"] for d in D_FIT],
                measured_change=[sums[d]["measured"] / k for d, k in zip(D_FIT, n)],
                predicted_change=[sums[d]["pred_mean"] / k for d, k in zip(D_FIT, n)],
                predicted_change_samples=[sums[d]["pred_samples"] / k for d, k in zip(D_FIT, n)])


def populations(ds, stats, moments):
    """Real start hours vs Part B moments, on unmodified data (original units)."""
    value = ds.x[..., MAP].numpy() * stats["std"][MAP] + stats["mean"][MAP]
    measured = ds.mask[..., MAP].numpy() > 0
    nee = ds.a[..., NEE].numpy()
    valid = ds.valid.numpy() > 0

    def describe(points, with_dose):
        level, trend_mean, trend_3h, earlier, dose, hour = [], [], [], [], [], []
        for i, t in points:
            if measured[i, t]:
                level.append(value[i, t])
                prev = [value[i, s] for s in range(max(0, t - 3), t) if measured[i, s]]
                if prev:
                    trend_mean.append(value[i, t] - np.mean(prev))
                if t >= 3 and measured[i, t - 3]:
                    trend_3h.append(value[i, t] - value[i, t - 3])
            earlier.append(bool(nee[i, :t].sum() > 0))
            hour.append(t)
            if with_dose:
                dose.append(nee[i, t])
        f = lambda v: float(np.mean(v)) if len(v) else None     # noqa: E731
        return dict(n=len(points), n_map_measured=len(level), map_level=f(level),
                    map_minus_mean_prev3h=f(trend_mean), map_minus_map_3h_before=f(trend_3h),
                    share_with_earlier_vasopressor=f(earlier), mean_hour=f(hour),
                    mean_start_dose=f(dose) if with_dose else None,
                    share_map_below_65=f([v < 65 for v in level]))

    starts = [(i, t) for i, t in zip(*np.nonzero(C.vaso_starts(nee))) if valid[i, t]]
    first_low = [(i, t) for i, t in starts if t >= 2 and nee[i, :t].sum() == 0 and measured[i, t] and value[i, t] < 65]
    return dict(real_starts=describe(starts, True), first_starts_map_below_65=describe(first_low, True),
                part_b_moments=describe(moments, False))
