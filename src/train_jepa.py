"""
train_jepa.py - Stage 14: train the JEPA-style world model, then its vitals probe.
New file; train.py is only imported (its constants, KL, W&B helpers).

Loss per predicted hour (averaged over real predicted hours):
    pred   = 0.5 * sum over the D embedding dims of (predicted - target embedding)^2
    var    = VICReg variance term on the online context embeddings h_t and on the predictions
    cov    = VICReg covariance term on the same two
    kl     = KL(posterior || prior), summed over the latent dims (as in train.py)
    loss   = pred + VAR_WEIGHT * var + COV_WEIGHT * cov + BETA_KL * max(kl, FREE_BITS)
After every optimizer step the target encoder moves towards the online encoder (EMA).
beta, free bits, optimizer, learning rate, batch size, patience and max epochs are train.py's.

Collapse check, every epoch on the validation split: per-dimension std across real hours of the
target embeddings, the online context embeddings and the predictions; logged as min / median and
the number of dims with std < COLLAPSE_STD.

Probe (after training, model frozen): ProbeHead(predicted embedding) -> Gaussian over the 5
vitals, trained with the masked Gaussian NLL of x_{t+1}, z sampled from the posterior: the same
inputs the VAE decoder is trained on.
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T                                                     # noqa: E402
import model_jepa as MJ                                               # noqa: E402

COLLAPSE_STD = 0.01
PROBE_LR, PROBE_MAX_EPOCHS, PROBE_PATIENCE = 1e-3, 40, 4


def jepa_loss(out):
    valid = out["valid_target"] > 0                                                   # (B, T-1)
    n = valid.sum()
    sq = 0.5 * ((out["pred"] - out["target"]) ** 2).sum(-1)                           # (B, T-1)
    pred_loss = (sq * valid).sum() / n
    h_v, p_v = out["h"][:, :-1][valid], out["pred"][valid]                           # (N, D) each
    var = MJ.variance_term(h_v) + MJ.variance_term(p_v)
    cov = MJ.covariance_term(h_v) + MJ.covariance_term(p_v)
    kl_dims = T.gaussian_kl(out["post_mu"], out["post_log_sigma"], out["prior_mu"], out["prior_log_sigma"])
    kl_dims = kl_dims * valid[..., None]
    kl = kl_dims.sum(-1).sum() / n
    loss = pred_loss + MJ.VAR_WEIGHT * var + MJ.COV_WEIGHT * cov + T.BETA_KL * torch.clamp(kl, min=T.FREE_BITS)
    return loss, dict(pred=pred_loss, var=var, cov=cov, kl=kl), kl_dims.sum(dim=(0, 1)) / n


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    keys = ("loss", "pred", "var", "cov", "kl")
    sums, n, kl_dims_sum = {k: 0.0 for k in keys}, 0, 0.0
    D = model.d
    emb = {k: dict(s=torch.zeros(D, device=device, dtype=torch.float64),
                   sq=torch.zeros(D, device=device, dtype=torch.float64), n=0) for k in ("target", "context", "pred")}
    with torch.random.fork_rng(enabled=not training), torch.set_grad_enabled(training):
        if not training:
            torch.manual_seed(123)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            loss, parts, kl_dims = jepa_loss(out)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.ema_update()
            else:
                valid = out["valid_target"] > 0
                for k, e in (("target", out["target"]), ("context", out["h"][:, :-1]), ("pred", out["pred"])):
                    v = e[valid].detach().double()
                    emb[k]["s"] += v.sum(0)
                    emb[k]["sq"] += (v ** 2).sum(0)
                    emb[k]["n"] += len(v)
            b = len(batch["idx"])
            sums["loss"] += loss.item() * b
            for k in ("pred", "var", "cov", "kl"):
                sums[k] += parts[k].item() * b
            kl_dims_sum = kl_dims_sum + kl_dims.detach() * b
            n += b
    res = {k: v / n for k, v in sums.items()}
    res["kl_dims"] = (kl_dims_sum / n).tolist()
    if not training:
        res["embedding_std"] = {}
        for k, a in emb.items():
            m = a["s"] / max(a["n"], 1)
            std = (a["sq"] / max(a["n"], 1) - m ** 2).clamp(min=0).sqrt().cpu()
            res["embedding_std"][k] = dict(per_dim=std.tolist(), min=float(std.min()), median=float(std.median()),
                                           n_collapsed=int((std < COLLAPSE_STD).sum()))
    return res


def train_jepa(name, seed=T.SEED, loaders=None, model_kwargs=None, wandb_config=None):
    """Same loop as train.train (early stopping on validation loss, checkpoint on every best,
    time budget), with the JEPA loss, the EMA update and the collapse check."""
    tag = name if seed == T.SEED else f"{name}_s{seed}"
    torch.manual_seed(seed)
    model = MJ.JEPAWorldModel(**(model_kwargs or {})).to(T.DEVICE)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=T.LR,
                                  weight_decay=T.WEIGHT_DECAY)
    history, best_val, best_state, bad = [], float("inf"), None, 0
    run = T._wandb_start(tag, seed, loaders, model_kwargs, dict(
        model="JEPA", ema_momentum=MJ.EMA_MOMENTUM, var_weight=MJ.VAR_WEIGHT, cov_weight=MJ.COV_WEIGHT,
        prediction_loss="0.5 * squared error summed over embedding dims", **(wandb_config or {})))
    start = time.time()
    for epoch in range(1, T.MAX_EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(model, loaders["train"], optimizer)
        va = run_epoch(model, loaders["val"])
        seconds = time.time() - t0
        is_best = va["loss"] < best_val - 1e-4
        history.append(dict(epoch=epoch, train=tr, val=va, seconds=seconds))
        if is_best:
            best_val, best_state, bad = va["loss"], copy.deepcopy(model.state_dict()), 0
            torch.save(best_state, T.OUT / f"model_{tag}.pt")
        else:
            bad += 1
        (T.OUT / f"history_{tag}.json").write_text(json.dumps(history))
        es = va["embedding_std"]
        print(f"  [{tag}] epoch {epoch:3d}  train loss {tr['loss']:.3f}  val loss {va['loss']:.3f} "
              f"(pred {va['pred']:.3f}, var {va['var']:.3f}, cov {va['cov']:.3f}, KL {va['kl']:.3f})  "
              f"emb std min/med target {es['target']['min']:.3f}/{es['target']['median']:.3f} "
              f"pred {es['pred']['min']:.3f}/{es['pred']['median']:.3f}  {seconds:.0f}s"
              + ("  *best" if is_best else ""), flush=True)
        if run is not None:
            log = dict(epoch=epoch, train_loss=tr["loss"], val_loss=va["loss"], val_pred=va["pred"], val_var=va["var"],
                       val_cov=va["cov"], val_kl=va["kl"], seconds_per_epoch=seconds, is_best=int(is_best),
                       **{f"val_kl_dim_{i}": k for i, k in enumerate(va["kl_dims"])})
            for k, s in es.items():
                log.update({f"emb_std/{k}_min": s["min"], f"emb_std/{k}_median": s["median"],
                            f"emb_std/{k}_n_collapsed": s["n_collapsed"]})
            run.log(log, step=epoch)
        if bad >= T.PATIENCE:
            print(f"  [{tag}] early stop at epoch {epoch}")
            break
        if T.TIME_BUDGET_S and time.time() - start + seconds > T.TIME_BUDGET_S:
            print(f"  [{tag}] time budget of {T.TIME_BUDGET_S / 3600:.2f} h reached after epoch {epoch}")
            break
    model.load_state_dict(best_state)
    return model, history, tag


# ---------------------------------------------------------------- probe

def _probe_epoch(model, probe, loader, optimizer=None):
    training = optimizer is not None
    probe.train(training)
    model.eval()
    device = next(probe.parameters()).device
    total, n_obs = 0.0, 0.0
    with torch.random.fork_rng(enabled=not training):
        if not training:
            torch.manual_seed(123)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                out = model(batch)                                         # z ~ posterior, model frozen
            with torch.set_grad_enabled(training):
                mu, ls = probe(out["pred"])
                nll = T.gaussian_nll(out["x_target"], mu, ls) * out["mask_target"] * out["valid_target"][..., None]
                n = out["valid_target"].sum()
                loss = nll.sum() / n                                       # nats per real predicted hour
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total += float(nll.sum())
            n_obs += float(n)
    return total / max(n_obs, 1)


def train_probe(model, loaders, tag, first_step=0):
    """Train the vitals probe on the frozen model; best by validation NLL. Saves model_{tag}_probe.pt."""
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(0)
    probe = MJ.ProbeHead(model.d, model.d, model.n_vitals).to(next(model.parameters()).device)
    opt = torch.optim.AdamW(probe.parameters(), lr=PROBE_LR, weight_decay=T.WEIGHT_DECAY)
    hist, best, best_state, bad = [], float("inf"), None, 0
    for epoch in range(1, PROBE_MAX_EPOCHS + 1):
        t0 = time.time()
        tr = _probe_epoch(model, probe, loaders["train"], opt)
        va = _probe_epoch(model, probe, loaders["val"])
        is_best = va < best - 1e-4
        hist.append(dict(epoch=epoch, train_nll=tr, val_nll=va, seconds=time.time() - t0))
        if is_best:
            best, best_state, bad = va, copy.deepcopy(probe.state_dict()), 0
            torch.save(best_state, T.OUT / f"model_{tag}_probe.pt")
        else:
            bad += 1
        print(f"  [{tag} probe] epoch {epoch:2d}  train NLL {tr:.3f}  val NLL {va:.3f}  "
              f"{hist[-1]['seconds']:.0f}s" + ("  *best" if is_best else ""), flush=True)
        if T.WANDB_RUN is not None:
            T.WANDB_RUN.log(dict(probe_epoch=epoch, probe_train_nll=tr, probe_val_nll=va), step=first_step + epoch)
        if bad >= PROBE_PATIENCE:
            break
    probe.load_state_dict(best_state)
    (T.OUT / f"history_{tag}_probe.json").write_text(json.dumps(hist))
    for p in model.parameters():
        if not any(p is q for q in model.target.parameters()):
            p.requires_grad_(True)
    return probe.eval(), hist


def summary_of(history):
    """Kept epoch (improvement > 1e-4, as train() keeps it) and its validation numbers."""
    best, ep = float("inf"), None
    for h in history:
        if h["val"]["loss"] < best - 1e-4:
            best, ep = h["val"]["loss"], h["epoch"]
    v = next(h for h in history if h["epoch"] == ep)["val"]
    return dict(checkpoint_epoch=ep, epochs=len(history),
                seconds_per_epoch_median=float(np.median([h["seconds"] for h in history])),
                val_loss=v["loss"], val_pred=v["pred"], val_var=v["var"], val_cov=v["cov"], val_kl=v["kl"],
                val_kl_dims=v["kl_dims"],
                embedding_std={k: dict(min=s["min"], median=s["median"], n_collapsed=s["n_collapsed"])
                               for k, s in v["embedding_std"].items()},
                stopped_early=len(history) - ep >= T.PATIENCE)
