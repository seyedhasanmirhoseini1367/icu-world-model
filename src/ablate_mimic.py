"""
ablate_mimic.py - Stage 9 (diagnosis): does the MIMIC world model use the actions at all?
Evaluation only: no training, no architecture change.

1. One-step PRIOR forecast (z = prior mean, decoder mean), MAE for MAP and HR on the
   test split, under four action inputs. The actions enter twice: the token of hour t
   gets a_{t-1}, and the prior for hour t+1 gets a_t; each variant changes both.
       real       the actions that were given
       all_zero   every action set to 0
       shuffled   each stay gets the actions of another test stay (same distribution)
       nee_zero   only the vasopressor (NEE) set to 0; fluids and ventilation real
   Reported on all scored hours, and on "NEE-active" hours: predictions whose a_t has
   NEE > 0 or a NEE different from the hour before.
2. Sensitivity of the prior: with the real history h_t fixed, set a_t's NEE to 0 and to
   0.1 mcg/kg/min and measure how far the prior mean of z moves, per latent dim and as
   an L2 distance, against the typical spread of z (SD of the prior mean across hours,
   and the prior's own sigma). Also the resulting change in the decoded MAP.

Only aggregate numbers are written.
Run:  python src/ablate_mimic.py --npz mimic_hourly.npz --model model_mimic.pt --out ablation.json [--device cuda]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mimic_data import ACTIONS, VITALS, load_splits_mimic   # noqa: E402
from model import WorldModel                                 # noqa: E402

VARIANTS = ("real", "all_zero", "shuffled", "nee_zero")
SUBSETS = ("all_hours", "nee_active_hours")
REPORT_VITALS = ("MAP", "HR")
NEE = ACTIONS.index("NEE mcg/kg/min")
DOSE = 0.1


def variant_actions(name, a, a_shuffled):
    """a, a_shuffled (B, T, A) -> the action tensor for one variant."""
    if name == "real":
        return a
    if name == "all_zero":
        return torch.zeros_like(a)
    if name == "shuffled":
        return a_shuffled
    out = a.clone()                     # nee_zero
    out[..., NEE] = 0.0
    return out


@torch.no_grad()
def run(npz, model_path, device="cpu", batch_size=256, seed=0):
    torch.manual_seed(seed)
    datasets, stats = load_splits_mimic(npz)
    test = datasets["test"]
    model = WorldModel(n_vitals=len(VITALS), n_actions=len(ACTIONS))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device).eval()
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device)     # (V,)
    cols = {v: VITALS.index(v) for v in REPORT_VITALS}
    perm = torch.randperm(len(test), generator=torch.Generator().manual_seed(seed))
    a_shuffled_all = test.a[perm]                                             # (N, T, A)

    err = {var: {sub: {v: [0, 0.0] for v in REPORT_VITALS} for sub in SUBSETS} for var in VARIANTS}
    Z = model.prior.net[-1].out_features // 2
    zs = dict(n=0, abs_d=torch.zeros(Z), sq_d=torch.zeros(Z), l2_d=0.0, mu=torch.zeros(Z),
              mu_sq=torch.zeros(Z), sigma=torch.zeros(Z), dmap=0.0, dmap_abs=0.0)
    n_active = 0

    for start in range(0, len(test), batch_size):
        sl = slice(start, start + batch_size)
        base = {k: getattr(test, k)[sl].to(device) for k in ("x", "mask", "delta", "a", "valid", "idx")}
        a_real = base["a"]                                                    # (B, T, A)
        nee = a_real[..., NEE]                                                # (B, T)
        nee_prev = torch.cat([torch.zeros_like(nee[:, :1]), nee[:, :-1]], 1)  # NEE one hour earlier
        active = ((nee > 0) | (nee != nee_prev))[:, :-1]                      # (B, T-1) for a_t, t = 0..T-2
        scored = base["mask"][:, 1:].bool() & base["valid"][:, 1:].bool()[..., None]   # (B, T-1, V)
        target = base["x"][:, 1:]                                             # (B, T-1, V)
        n_active += int((active & base["valid"][:, 1:].bool()).sum())

        for var in VARIANTS:
            out = model({**base, "a": variant_actions(var, a_real, a_shuffled_all[sl].to(device))})
            point, _ = model.decoder(torch.cat([out["prior_mu"], out["h"][:, :-1]], -1))   # (B, T-1, V)
            abs_err = (point - target).abs() * std                            # original units
            for sub, sel in (("all_hours", None), ("nee_active_hours", active)):
                for v, j in cols.items():
                    m = scored[..., j] if sel is None else scored[..., j] & sel
                    err[var][sub][v][0] += int(m.sum())
                    err[var][sub][v][1] += float(abs_err[..., j][m].sum())
            if var != "real":
                continue
            # Sensitivity of the prior to a_t's NEE, real history fixed.
            h = out["h"][:, :-1]                                              # (B, T-1, D)
            a_t = a_real[:, :-1]
            a0, a1 = a_t.clone(), a_t.clone()
            a0[..., NEE], a1[..., NEE] = 0.0, DOSE
            mu0, ls0 = model.prior(torch.cat([h, a0], -1))                    # (B, T-1, Z)
            mu1, _ = model.prior(torch.cat([h, a1], -1))
            m0, _ = model.decoder(torch.cat([mu0, h], -1))                    # (B, T-1, V)
            m1, _ = model.decoder(torch.cat([mu1, h], -1))
            v = base["valid"][:, 1:].bool()                                   # positions with a real next hour
            d = (mu1 - mu0)[v].cpu()                                          # (n, Z)
            zs["n"] += len(d)
            zs["abs_d"] += d.abs().sum(0)
            zs["sq_d"] += (d ** 2).sum(0)
            zs["l2_d"] += float(d.norm(dim=1).sum())
            zs["mu"] += mu0[v].cpu().sum(0)
            zs["mu_sq"] += (mu0[v].cpu() ** 2).sum(0)
            zs["sigma"] += ls0[v].exp().cpu().sum(0)
            dmap = ((m1 - m0)[..., cols["MAP"]] * std[cols["MAP"]])[v]
            zs["dmap"] += float(dmap.sum())
            zs["dmap_abs"] += float(dmap.abs().sum())

    mae = {var: {sub: {v: dict(n=n, mae=(s / n if n else None)) for v, (n, s) in subs.items()}
                 for sub, subs in by_sub.items()} for var, by_sub in err.items()}
    n = zs["n"]
    mu_mean = zs["mu"] / n
    mu_sd = (zs["mu_sq"] / n - mu_mean ** 2).clamp(min=0).sqrt()             # spread of the prior mean
    z_report = dict(
        n_positions=n, dose=DOSE,
        mean_abs_shift_per_dim=(zs["abs_d"] / n).tolist(),
        rms_shift_per_dim=(zs["sq_d"] / n).sqrt().tolist(),
        prior_mean_sd_per_dim=mu_sd.tolist(),
        prior_sigma_per_dim=(zs["sigma"] / n).tolist(),
        shift_over_spread_per_dim=((zs["abs_d"] / n) / mu_sd.clamp(min=1e-8)).tolist(),
        mean_l2_shift=zs["l2_d"] / n,
        typical_l2_spread=float(mu_sd.norm()),
        decoded_map_change_mean=zs["dmap"] / n,
        decoded_map_change_abs_mean=zs["dmap_abs"] / n)
    return dict(n_test_stays=len(test), n_nee_active_predictions=n_active, mae=mae, z_sensitivity=z_report)


def report(r):
    lines = [f"test stays {r['n_test_stays']:,}; NEE-active predictions {r['n_nee_active_predictions']:,}",
             f"{'variant':<9} " + "  ".join(f"{s + ' ' + v:>26}" for s in SUBSETS for v in REPORT_VITALS)]
    real = r["mae"]["real"]
    for var in VARIANTS:
        cells = []
        for s in SUBSETS:
            for v in REPORT_VITALS:
                m, base = r["mae"][var][s][v], real[s][v]
                if m["mae"] is None:
                    cells.append(f"{'n/a':>26}")
                    continue
                rel = (m["mae"] / base["mae"] - 1) * 100
                cells.append(f"{m['mae']:8.3f} ({rel:+5.1f}%) n={m['n']:>8,}")
        lines.append(f"{var:<9} " + "  ".join(cells))
    z = r["z_sensitivity"]
    lines.append(f"z shift for NEE 0 -> {z['dose']} (n {z['n_positions']:,} positions):")
    lines.append("  dim:             " + " ".join(f"{i:>8}" for i in range(len(z["mean_abs_shift_per_dim"]))))
    for key in ("mean_abs_shift_per_dim", "prior_mean_sd_per_dim", "prior_sigma_per_dim", "shift_over_spread_per_dim"):
        lines.append(f"  {key[:16]:<16} " + " ".join(f"{x:8.4f}" for x in z[key]))
    lines.append(f"  mean L2 shift {z['mean_l2_shift']:.4f} vs typical L2 spread {z['typical_l2_spread']:.4f}"
                 f" | decoded MAP change mean {z['decoded_map_change_mean']:+.3f} mmHg"
                 f" (mean |change| {z['decoded_map_change_abs_mean']:.3f})")
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="ablation_metrics.json")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    res = run(args.npz, args.model, args.device)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(report(res))
