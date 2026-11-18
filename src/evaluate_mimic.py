"""
evaluate_mimic.py - evaluate a trained MIMIC world model on the TEST split. No training.

Scored hours: hour t+1 inside the stay where the vital was observed AND had been observed
at least once before (so the naive baseline exists). The same hours are used for every method.

    prior      forecast: point = decoder mean with z = prior mean (sees hours 0..t and a_t only)
               band  = 10-90% of 50 samples (z ~ prior, then x ~ decoder)
    posterior  reconstruction: the same with the posterior, which also sees x_{t+1};
               shown only to measure the gap, it is not a forecast
    naive      next hour = last observed value of that vital at or before hour t

Errors are in original units (bpm, breaths/min, %, mmHg, degrees C).

Run:  python src/evaluate_mimic.py --npz mimic_hourly.npz --model model_mimic.pt --out metrics.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mimic_data import ACTIONS, VITALS, load_splits_mimic   # noqa: E402
from model import load_world_model                           # noqa: E402

UNITS = dict(HR="bpm", RR="/min", SpO2="%", MAP="mmHg", Temp="C")


def carry_forward(x, mask):
    """Last observed value at or before each hour. x, mask (B, T, V) -> last, seen (B, T, V)."""
    last, seen = torch.zeros_like(x[:, 0]), torch.zeros_like(mask[:, 0], dtype=torch.bool)
    lasts, seens = [], []
    for t in range(x.shape[1]):
        m = mask[:, t].bool()                                         # (B, V)
        last = torch.where(m, x[:, t], last)
        seen = seen | m
        lasts.append(last)
        seens.append(seen)
    return torch.stack(lasts, 1), torch.stack(seens, 1)              # (B, T, V) each


@torch.no_grad()
def evaluate(npz, model_path, n_samples=50, batch_size=256, seed=0):
    torch.manual_seed(seed)
    datasets, stats = load_splits_mimic(npz)
    test = datasets["test"]
    model = load_world_model(model_path, len(VITALS), len(ACTIONS))
    std = torch.tensor(stats["std"], dtype=torch.float32)            # (V,) z-units -> original units
    V = len(VITALS)
    sums = {k: dict(n=torch.zeros(V), abs=torch.zeros(V), sq=torch.zeros(V), inside=torch.zeros(V))
            for k in ("prior", "posterior", "naive")}

    for batch in DataLoader(test, batch_size=batch_size):
        out = model(batch)
        h_now = out["h"][:, :-1]                                      # (B, T-1, D)
        target = batch["x"][:, 1:]                                    # (B, T-1, V)
        last, seen = carry_forward(batch["x"], batch["mask"])         # (B, T, V)
        scored = (batch["mask"][:, 1:].bool() & seen[:, :-1]
                  & batch["valid"][:, 1:].bool()[..., None])          # (B, T-1, V)

        preds = {"naive": (last[:, :-1], None)}
        for kind, key in (("prior", "prior"), ("posterior", "post")):
            mu, log_sigma = out[f"{key}_mu"], out[f"{key}_log_sigma"]     # (B, T-1, Z)
            point, _ = model.decoder(torch.cat([mu, h_now], -1))          # (B, T-1, V)
            z = mu + log_sigma.exp() * torch.randn(n_samples, *mu.shape)  # (S, B, T-1, Z)
            dec_mu, dec_ls = model.decoder(torch.cat([z, h_now.expand(n_samples, *h_now.shape)], -1))
            samples = dec_mu + dec_ls.exp() * torch.randn_like(dec_mu)    # (S, B, T-1, V)
            lo, hi = torch.quantile(samples, torch.tensor([0.1, 0.9]), dim=0)  # (B, T-1, V) each
            preds[kind] = (point, (target >= lo) & (target <= hi))

        for kind, (point, inside) in preds.items():
            err = (point - target) * std                              # (B, T-1, V), original units
            s = sums[kind]
            s["n"] += scored.sum((0, 1))
            s["abs"] += (err.abs() * scored).sum((0, 1))
            s["sq"] += (err ** 2 * scored).sum((0, 1))
            if inside is not None:
                s["inside"] += (inside & scored).sum((0, 1))

    metrics = {}
    for kind, s in sums.items():
        for j, vital in enumerate(VITALS):
            n = s["n"][j].item()
            if n == 0:                                                # vital never scored
                metrics.setdefault(vital, {})[kind] = dict(n=0, mae=None, rmse=None, coverage=None)
                continue
            metrics.setdefault(vital, {})[kind] = dict(
                n=int(n), mae=s["abs"][j].item() / n, rmse=(s["sq"][j].item() / n) ** 0.5,
                coverage=None if kind == "naive" else s["inside"][j].item() / n)
    return dict(n_test_stays=len(test), n_samples=n_samples, units=UNITS, metrics=metrics)


def table(result):
    lines = [f"{'vital':<5} {'unit':<5} {'scored hours':>12} | {'prior MAE / RMSE / in 10-90%':>30} | "
             f"{'naive MAE / RMSE':>17} | {'posterior MAE / RMSE / in 10-90%':>33}"]
    for vital, m in result["metrics"].items():
        p, n, q = m["prior"], m["naive"], m["posterior"]
        if p["n"] == 0:
            lines.append(f"{vital:<5} {UNITS[vital]:<5} {0:>12} | no scored hours")
            continue
        lines.append(f"{vital:<5} {UNITS[vital]:<5} {p['n']:>12,} | "
                     f"{p['mae']:8.2f} / {p['rmse']:6.2f} / {p['coverage']:6.1%}      | "
                     f"{n['mae']:7.2f} / {n['rmse']:6.2f} | "
                     f"{q['mae']:9.2f} / {q['rmse']:6.2f} / {q['coverage']:6.1%}")
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="mimic_test_metrics.json")
    args = p.parse_args()
    result = evaluate(args.npz, args.model)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"test stays: {result['n_test_stays']:,}")
    print(table(result))
    print(f"saved {args.out}")
