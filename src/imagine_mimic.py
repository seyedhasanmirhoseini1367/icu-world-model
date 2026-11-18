"""
imagine_mimic.py - Stage 8 on MIMIC: multi-hour imagination, evaluation only (no training).

Part A  Imagination accuracy. For each test stay and origin hour t, imagine hours
        t+1..t+12 with the prior only, feeding the REAL actions that were given
        (rollout.imagine: each imagined hour is fed back as if observed). Compare the
        median of 50 imagined samples with the measured value, against the naive
        "last observed value at or before t", and count how often the measurement falls
        inside the 10-90% band of the samples.

Part B  Vasopressor counterfactual sanity check. Moments: the first hour t (t >= 2) of a
        test stay where MAP was measured below 65 mmHg and no vasopressor had been given
        yet, with the stay lasting at least 6 more hours. From the same history, imagine
        6 hours under two plans: NEE = 0 and NEE = 0.1 mcg/kg/min, fluids and ventilation
        as they really were. Both plans use the same random numbers (paired samples), so
        the difference is the plan, not sampling noise. Also the naive observational
        difference: measured MAP of moments where NEE was started in hour t vs not.

Only aggregate numbers and figures are written (no per-patient values).
Run:  python src/imagine_mimic.py --npz mimic_hourly.npz --model model_mimic.pt --out-dir out [--device cuda]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_mimic import carry_forward                    # noqa: E402
from mimic_data import ACTIONS, VITALS, load_splits_mimic   # noqa: E402
from model import WorldModel                                 # noqa: E402
from rollout import imagine                                  # noqa: E402

N_SAMPLES = 50
ORIGINS_A, HORIZON_A, REPORT_A = (12, 24), 12, (1, 3, 6, 12)
HORIZON_B, REPORT_B = 6, (1, 3, 6)
MAP_LIMIT, DOSE = 65.0, 0.1                 # mmHg, mcg/kg/min norepinephrine equivalent
NEE = ACTIONS.index("NEE mcg/kg/min")


def repeat(t, S, device):
    """(B, ...) -> (B*S, ...): each stay repeated S times, samples of one stay contiguous."""
    return t.repeat_interleave(S, dim=0).to(device)


# ---------------------------------------------------------------- Part A

@torch.no_grad()
def part_a(model, test, stats, device, vitals=("MAP", "HR"), batch_size=128):
    S, H = N_SAMPLES, HORIZON_A
    cols = [VITALS.index(v) for v in vitals]
    acc = {v: dict(n=np.zeros(H), model=np.zeros(H), naive=np.zeros(H), inside=np.zeros(H)) for v in vitals}
    n_windows = 0
    for t0 in ORIGINS_A:
        rows = torch.where(test.valid[:, t0 + H] > 0)[0]                  # stay still running at t0+H
        n_windows += len(rows)
        for chunk in rows.split(batch_size):
            x, mask, delta, a = (getattr(test, k)[chunk] for k in ("x", "mask", "delta", "a"))
            B = len(chunk)
            sim = imagine(model, repeat(x[:, :t0 + 1], S, device), repeat(mask[:, :t0 + 1], S, device),
                          repeat(delta[:, :t0 + 1], S, device), repeat(a[:, :t0], S, device),
                          repeat(a[:, t0:t0 + H], S, device))             # (B*S, H, V) z-units
            sim = sim.view(B, S, H, -1).cpu()
            q10, q50, q90 = torch.quantile(sim, torch.tensor([0.1, 0.5, 0.9]), dim=1)   # (B, H, V) each
            last, seen = carry_forward(x[:, :t0 + 1], mask[:, :t0 + 1])               # (B, t0+1, V)
            naive, has_naive = last[:, -1], seen[:, -1]                               # (B, V)
            target = x[:, t0 + 1:t0 + H + 1]                                          # (B, H, V)
            scored = mask[:, t0 + 1:t0 + H + 1].bool() & has_naive[:, None, :]       # (B, H, V)
            for v, j in zip(vitals, cols):
                o = scored[..., j].float()
                s = acc[v]
                s["n"] += o.sum(0).numpy()
                s["model"] += ((q50[..., j] - target[..., j]).abs() * o).sum(0).numpy() * stats["std"][j]
                s["naive"] += ((naive[:, None, j] - target[..., j]).abs() * o).sum(0).numpy() * stats["std"][j]
                inside = (target[..., j] >= q10[..., j]) & (target[..., j] <= q90[..., j])
                s["inside"] += (inside.float() * o).sum(0).numpy()
    out = dict(origins=list(ORIGINS_A), n_windows=n_windows, horizons=list(range(1, H + 1)))
    for v, s in acc.items():
        n = np.maximum(s["n"], 1)
        out[v] = dict(n=s["n"].astype(int).tolist(), mae_model=(s["model"] / n).tolist(),
                      mae_naive=(s["naive"] / n).tolist(), coverage=(s["inside"] / n).tolist())
    return out


# ---------------------------------------------------------------- Part B

def find_moments(test, stats, vital="MAP", limit=MAP_LIMIT):
    """First hour t >= 2 per test stay with the vital measured below `limit`, no vasopressor
    in hours 0..t-1, and the stay still running at t+HORIZON_B. Returns [(row, t), ...]."""
    j = VITALS.index(vital)
    value = test.x[..., j].numpy() * stats["std"][j] + stats["mean"][j]      # (N, T) original units
    measured = test.mask[..., j].numpy() > 0
    nee = test.a[..., NEE].numpy()
    valid = test.valid.numpy() > 0
    T = value.shape[1]
    moments = []
    for i in range(len(value)):
        for t in range(2, T - HORIZON_B):
            if not valid[i, t + HORIZON_B] or nee[i, :t].sum() > 0:
                break                                                          # stay too short, or already treated
            if measured[i, t] and value[i, t] < limit:
                moments.append((i, t))
                break
    return moments


@torch.no_grad()
def part_b(model, test, stats, device, vital="MAP", limit=MAP_LIMIT, batch_size=128, seed=0):
    S, H = N_SAMPLES, HORIZON_B
    j = VITALS.index(vital)
    moments = find_moments(test, stats, vital, limit)
    diffs, levels = [], {0: [], 1: []}                       # per moment: (S, H) paired differences
    by_t = {}
    for i, t in moments:
        by_t.setdefault(t, []).append(i)
    for t, rows in sorted(by_t.items()):
        for chunk in torch.tensor(rows).split(batch_size):
            x, mask, delta, a = (getattr(test, k)[chunk] for k in ("x", "mask", "delta", "a"))
            B = len(chunk)
            arms = []
            for dose in (0.0, DOSE):
                plan = a[:, t:t + H].clone()                                    # (B, H, A) real fluids/ventilation
                plan[..., NEE] = dose
                torch.manual_seed(seed + t)                                     # same random numbers in both arms
                sim = imagine(model, repeat(x[:, :t + 1], S, device), repeat(mask[:, :t + 1], S, device),
                              repeat(delta[:, :t + 1], S, device), repeat(a[:, :t], S, device),
                              repeat(plan, S, device))                          # (B*S, H, V)
                arms.append(sim.view(B, S, H, -1)[..., j].cpu() * stats["std"][j] + stats["mean"][j])
            diffs.append(arms[1] - arms[0])                                     # (B, S, H) mmHg
            levels[0].append(arms[0].mean(1))
            levels[1].append(arms[1].mean(1))                                   # (B, H)
    out = dict(n_moments=len(moments), dose=DOSE, vital=vital, limit=limit,
               horizons=list(range(1, H + 1)))
    if not moments:
        return out
    d = torch.cat(diffs)                                                        # (M, S, H)
    per_moment = d.mean(1)                                                      # (M, H)
    out.update(
        mean_diff=per_moment.mean(0).tolist(),
        sample_sd=d.std(1).mean(0).tolist(),                                    # spread across the 50 samples
        moment_sd=per_moment.std(0).tolist(),                                   # spread across patients
        moment_p10=per_moment.quantile(0.1, dim=0).tolist(),
        moment_p90=per_moment.quantile(0.9, dim=0).tolist(),
        share_positive=(per_moment > 0).float().mean(0).tolist(),
        imagined_no_dose=torch.cat(levels[0]).mean(0).tolist(),
        imagined_dose=torch.cat(levels[1]).mean(0).tolist())

    # Naive observational comparison on the same moments: NEE started in hour t or not.
    value = test.x[..., j].numpy() * stats["std"][j] + stats["mean"][j]
    measured = test.mask[..., j].numpy() > 0
    started = np.array([test.a[i, t, NEE].item() > 0 for i, t in moments])
    obs = {}
    for h in [0] + list(REPORT_B):
        vals = {g: [value[i, t + h] for (i, t), s in zip(moments, started) if s == g and measured[i, t + h]]
                for g in (True, False)}
        obs[h] = dict(started=float(np.mean(vals[True])) if vals[True] else None,
                      not_started=float(np.mean(vals[False])) if vals[False] else None,
                      n_started=len(vals[True]), n_not_started=len(vals[False]))
    out["observational"] = dict(n_started=int(started.sum()), n_not_started=int((~started).sum()),
                                by_hour=obs)
    return out


# ---------------------------------------------------------------- figures and report

def plot_a(res, path):
    hours = res["horizons"]
    vitals = [v for v in ("MAP", "HR") if v in res]
    fig, axes = plt.subplots(2, len(vitals), figsize=(6 * len(vitals), 7), squeeze=False)
    for col, v in enumerate(vitals):
        r = res[v]
        ax = axes[0, col]
        ax.plot(hours, r["mae_model"], "o-", label="world model (median of 50 imagined samples)")
        ax.plot(hours, r["mae_naive"], "s--", color="0.4", label="naive: last observed value")
        ax.set_title(f"{v}: error vs horizon (test stays, real actions fed)")
        ax.set_xlabel("hours after the last real hour")
        ax.set_ylabel("MAE (" + ("mmHg" if v == "MAP" else "bpm") + ")")
        ax.legend(fontsize=8)
        ax = axes[1, col]
        ax.plot(hours, np.array(r["coverage"]) * 100, "o-", color="tab:green")
        ax.axhline(80, ls=":", color="0.5", label="target 80%")
        ax.set_ylim(0, 100)
        ax.set_title(f"{v}: measurements inside the 10-90% band")
        ax.set_xlabel("hours after the last real hour")
        ax.set_ylabel("%")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_b(res, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if res["n_moments"]:
        h = [0] + res["horizons"]
        mean = [0] + res["mean_diff"]
        ax.fill_between(h, [0] + res["moment_p10"], [0] + res["moment_p90"], alpha=0.2,
                        label="10-90% across patients")
        ax.plot(h, mean, "o-", label=f"imagined: NEE {res['dose']} minus NEE 0 (mean, n={res['n_moments']})")
        by = res["observational"]["by_hour"]                                    # {hour: {...}}
        hs = [h for h, r in by.items() if r["started"] is not None and r["not_started"] is not None]
        ax.plot(hs, [by[h]["started"] - by[h]["not_started"] for h in hs], "s--", color="tab:red",
                label="observed: started NEE minus not started")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("hours after the decision hour")
    ax.set_ylabel("MAP difference (mmHg)")
    ax.set_title("Vasopressor counterfactual vs observational difference (MAP < 65, untreated so far)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(npz, model_path, out_dir, device="cpu", b_vital="MAP", b_limit=MAP_LIMIT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    datasets, stats = load_splits_mimic(npz)
    test = datasets["test"]
    model = WorldModel(n_vitals=len(VITALS), n_actions=len(ACTIONS))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device).eval()

    nee_train = datasets["train"].a[..., NEE][datasets["train"].valid > 0]
    nonzero = nee_train[nee_train > 0]
    dose_context = dict(share_hours_with_nee=float((nee_train > 0).float().mean()),
                        nonzero_p10=float(nonzero.quantile(0.1)), nonzero_median=float(nonzero.median()),
                        nonzero_p90=float(nonzero.quantile(0.9)), share_nonzero_at_or_below_dose=float(
                            (nonzero <= DOSE).float().mean())) if len(nonzero) else {}

    result = dict(n_test_stays=len(test), n_samples=N_SAMPLES, train_nee_doses=dose_context,
                  part_a=part_a(model, test, stats, device),
                  part_b=part_b(model, test, stats, device, vital=b_vital, limit=b_limit))
    (out_dir / "imagination_metrics.json").write_text(json.dumps(result, indent=2))
    plot_a(result["part_a"], out_dir / "mimic_imagination_error.png")
    plot_b(result["part_b"], out_dir / "mimic_vasopressor_counterfactual.png")
    return result


def report(result):
    a, b = result["part_a"], result["part_b"]
    lines = [f"Part A: {a['n_windows']:,} windows (origins {a['origins']}), {result['n_samples']} samples"]
    for v in ("MAP", "HR"):
        r = a[v]
        lines.append(f"  {v}: " + "  ".join(
            f"+{h}h model {r['mae_model'][h - 1]:.2f} naive {r['mae_naive'][h - 1]:.2f} "
            f"band {r['coverage'][h - 1]:.0%} (n {r['n'][h - 1]:,})" for h in REPORT_A))
    lines.append(f"Part B: {b['n_moments']:,} moments ({b['vital']} < {b['limit']}, no vasopressor yet), "
                 f"dose {b['dose']}")
    if b["n_moments"]:
        for h in REPORT_B:
            k = h - 1
            lines.append(f"  +{h}h imagined diff {b['mean_diff'][k]:+.2f} | SD across samples "
                         f"{b['sample_sd'][k]:.2f} | across patients SD {b['moment_sd'][k]:.2f}, "
                         f"p10..p90 {b['moment_p10'][k]:+.2f}..{b['moment_p90'][k]:+.2f} | "
                         f"positive in {b['share_positive'][k]:.0%}")
        o = b["observational"]
        lines.append(f"  observational: started NEE in hour t {o['n_started']}, not {o['n_not_started']}")
        for h, r in o["by_hour"].items():
            if r["started"] is not None and r["not_started"] is not None:
                lines.append(f"    hour +{h}: started {r['started']:.1f} vs not {r['not_started']:.1f} "
                             f"= {r['started'] - r['not_started']:+.1f} mmHg "
                             f"(n {r['n_started']} / {r['n_not_started']})")
    lines.append(f"train NEE doses: {result['train_nee_doses']}")
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir", default="imagination_out")
    p.add_argument("--device", default="cpu")
    p.add_argument("--b-vital", default="MAP", help="testing only: run Part B on another vital")
    p.add_argument("--b-limit", type=float, default=MAP_LIMIT, help="testing only")
    args = p.parse_args()
    res = run(args.npz, args.model, args.out_dir, args.device, args.b_vital, args.b_limit)
    print(report(res))
