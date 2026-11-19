"""
train_jepa_b.py - Stage 14b: train the leak-fixed JEPA model (model_jepa_b), then its probe.

Same loss, EMA update, collapse check, early stopping, checkpointing and probe as Stage 14
(train_jepa.py, imported only). Added: the KL per predicted hour on the validation split, every
epoch (mean, share of hours below the free-bits floor, mean by hour of stay), and kl_profile(),
which measures the same thing for any trained model (VAE or JEPA) at its kept checkpoint.
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T                 # noqa: E402
import train_jepa as TJ           # noqa: E402
import model_jepa as MJ           # noqa: E402
import model_jepa_b as MJB        # noqa: E402

train_probe = TJ.train_probe


class KLHours:
    """Accumulates the KL (summed over z dims) of every real predicted hour."""

    def __init__(self):
        self.s, self.n, self.below, self.total, self.count, self.dims = None, None, 0.0, 0.0, 0.0, None

    def add(self, out):
        kl = T.gaussian_kl(out["post_mu"], out["post_log_sigma"], out["prior_mu"], out["prior_log_sigma"])  # (B, T-1, Z)
        valid = (out["valid_target"] > 0).double()
        kl_h = kl.sum(-1).double() * valid                                                              # (B, T-1)
        if self.s is None:
            L = kl_h.shape[1]
            self.s, self.n = torch.zeros(L, dtype=torch.float64), torch.zeros(L, dtype=torch.float64)
            self.dims = torch.zeros(kl.shape[-1], dtype=torch.float64)
        self.s += kl_h.sum(0).cpu()
        self.n += valid.sum(0).cpu()
        self.below += float(((kl.sum(-1) < T.FREE_BITS).double() * valid).sum())
        self.total += float(kl_h.sum())
        self.count += float(valid.sum())
        self.dims += (kl.double() * valid[..., None]).sum((0, 1)).cpu()

    def result(self):
        if not self.count:
            return {}
        by_hour = (self.s / self.n.clamp(min=1)).tolist()
        return dict(mean=self.total / self.count, share_below_free_bits=self.below / self.count,
                    free_bits=T.FREE_BITS, by_hour=by_hour, n_by_hour=self.n.tolist(),
                    per_dim=(self.dims / self.count).tolist(), n_hours=self.count)


@torch.no_grad()
def kl_profile(model, loader, device):
    """KL per real predicted hour for any model whose forward returns prior/posterior (VAE or raw JEPA)."""
    model.eval()
    acc = KLHours()
    with torch.random.fork_rng():
        torch.manual_seed(123)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            acc.add(model(batch))
    return acc.result()


def run_epoch(model, loader, optimizer=None):
    """train_jepa.run_epoch plus the per-hour KL on validation."""
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    keys = ("loss", "pred", "var", "cov", "kl")
    sums, n, kl_dims_sum = {k: 0.0 for k in keys}, 0, 0.0
    D = model.d
    emb = {k: dict(s=torch.zeros(D, device=device, dtype=torch.float64),
                   sq=torch.zeros(D, device=device, dtype=torch.float64), n=0) for k in ("target", "context", "pred")}
    klh = KLHours()
    with torch.random.fork_rng(enabled=not training), torch.set_grad_enabled(training):
        if not training:
            torch.manual_seed(123)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            loss, parts, kl_dims = TJ.jepa_loss(out)
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
                klh.add(out)
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
                                           n_collapsed=int((std < TJ.COLLAPSE_STD).sum()))
        res["kl_hours"] = klh.result()
    return res


def train_jepa_b(name, seed=T.SEED, loaders=None, model_kwargs=None, wandb_config=None):
    """train_jepa.train_jepa with the leak-fixed model and the per-hour KL log."""
    tag = name if seed == T.SEED else f"{name}_s{seed}"
    torch.manual_seed(seed)
    model = MJB.JEPANoActTarget(**(model_kwargs or {})).to(T.DEVICE)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=T.LR,
                                  weight_decay=T.WEIGHT_DECAY)
    history, best_val, best_state, bad = [], float("inf"), None, 0
    run = T._wandb_start(tag, seed, loaders, model_kwargs, dict(
        model="JEPA-14b (target without action input)", ema_momentum=MJ.EMA_MOMENTUM, var_weight=MJ.VAR_WEIGHT,
        cov_weight=MJ.COV_WEIGHT, target_action_input=False,
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
        es, kh = va["embedding_std"], va["kl_hours"]
        print(f"  [{tag}] epoch {epoch:3d}  train loss {tr['loss']:.3f}  val loss {va['loss']:.3f} "
              f"(pred {va['pred']:.3f}, var {va['var']:.3f}, cov {va['cov']:.3f}, KL {va['kl']:.3f}; "
              f"hours below free bits {100 * kh['share_below_free_bits']:.0f}%)  "
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
            bh = np.array(kh["by_hour"])
            log.update({"kl_hour/mean": kh["mean"], "kl_hour/share_below_free_bits": kh["share_below_free_bits"],
                        "kl_hour/min_over_hours": float(bh.min()), "kl_hour/max_over_hours": float(bh.max()),
                        "kl_hour/hour0": float(bh[0]), "kl_hour/hour1": float(bh[1]) if len(bh) > 1 else None})
            run.log(log, step=epoch)
        if bad >= T.PATIENCE:
            print(f"  [{tag}] early stop at epoch {epoch}")
            break
        if T.TIME_BUDGET_S and time.time() - start + seconds > T.TIME_BUDGET_S:
            print(f"  [{tag}] time budget of {T.TIME_BUDGET_S / 3600:.2f} h reached after epoch {epoch}")
            break
    model.load_state_dict(best_state)
    return model, history, tag


def summary_of(history):
    """train_jepa.summary_of plus the per-hour KL summary at the kept epoch."""
    s = TJ.summary_of(history)
    kh = next(h for h in history if h["epoch"] == s["checkpoint_epoch"])["val"].get("kl_hours", {})
    s["kl_hours"] = {k: kh.get(k) for k in ("mean", "share_below_free_bits", "per_dim")}
    return s
