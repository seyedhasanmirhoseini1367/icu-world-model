"""
eval_eicu.py - run a frozen MIMIC world model on eICU stays (no training) and score one-step forecasts
exactly as src/evaluate_mimic.py does. New file; evaluate_mimic is imported, not changed: its data
loader is swapped at run time for one that returns ALL eICU stays as the test set.

Normalisation:
    mimic  z-scores with the MIMIC training mean/std (the scale the model was trained on). These are
           read from the Stage 7 extraction log, which prints them to 0.1 (see MIMIC_STATS).
    eicu   z-scores with eICU's own mean/std (sensitivity check).

Run:  python src/eval_eicu.py --npz eicu_hourly.npz --model real_actnow_s0.pt --out eicu_metrics.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_mimic as EM                                   # noqa: E402
from data import ICUSequences, hours_since_observed           # noqa: E402
from mimic_data import ACTIONS, DELTA_CAP, VITALS             # noqa: E402

# MIMIC-IV training split, from outputs/kaggle_mimic_v3/icu-world-model-mimic.log (printed to 0.1).
MIMIC_STATS = dict(mean=np.array([84.7, 19.4, 96.7, 78.9, 36.9]), std=np.array([18.1, 5.6, 3.0, 15.1, 0.7]))


def load_all(npz, norm):
    raw = np.load(npz)
    x_raw, a, valid = raw["x"], raw["a"], raw["valid"]
    mask = ~np.isnan(x_raw) & valid[..., None]
    if norm == "mimic":
        stats = MIMIC_STATS
    else:
        xo = np.where(mask, x_raw, np.nan)
        stats = dict(mean=np.nanmean(xo, axis=(0, 1)), std=np.nanstd(xo, axis=(0, 1)))
    x = np.where(mask, (x_raw - stats["mean"]) / stats["std"], 0.0)
    delta = np.minimum(hours_since_observed(mask), DELTA_CAP)
    idx = np.arange(len(x))
    return {"test": ICUSequences(x, mask, delta, a, idx, valid=valid)}, stats


def describe(npz):
    raw = np.load(npz)
    x, a, valid = raw["x"], raw["a"], raw["valid"].astype(bool)
    out = dict(stays=int(len(x)), median_valid_hours=float(np.median(valid.sum(1))))
    for j, n in enumerate(VITALS):
        v = x[..., j][valid]
        out[n] = dict(observed_share=float(np.mean(~np.isnan(v))), mean=float(np.nanmean(v)), std=float(np.nanstd(v)))
    for j, n in enumerate(ACTIONS):
        v = a[..., j][valid]
        out[n] = dict(nonzero_hour_share=float(np.mean(v > 0)),
                      stays_with_any=int(((a[..., j] > 0) & valid).any(1).sum()))
    return out


def run(npz, model, norm, n_samples=50):
    EM.load_splits_mimic = lambda path: load_all(path, norm)
    res = EM.evaluate(npz, model, n_samples=n_samples)
    res["normalisation"] = norm
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = dict(data=describe(args.npz), runs={})
    print(json.dumps(result["data"], indent=1))
    for m in args.model:
        for norm in ("mimic", "eicu"):
            key = f"{Path(m).stem}|{norm}"
            r = run(args.npz, m, norm)
            result["runs"][key] = r
            print(f"\n== {key}  (stays {r['n_test_stays']})")
            print(EM.table(r))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"saved {args.out}")
