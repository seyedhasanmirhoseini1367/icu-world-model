"""
negative_control.py - Stage 8 Part B as a negative control, all vitals (evaluation only).

Same counterfactual as Stage 8 Part B: the 6,413 test moments (first hour with MAP < 65 and
no vasopressor before), NEE 0 vs 0.1 mcg/kg/min for 6 h, other actions real, 50 paired
samples per arm. But the imagined difference is reported for every vital, not only MAP.
Norepinephrine should have about no direct effect on temperature, so a clear imagined "effect"
on Temp, beyond the noise floor (seed 1 minus seed 0 of the real-data model), points to
confounding the model has absorbed, not to the drug.

Also reported: the raw observed difference, started at t vs not, for the same vitals.
Only aggregates are written.
"""
import numpy as np
import torch

import imagine_mimic as IM
import mimic_data
from overnight_stage11 import Runner
from rollout import imagine
import semisynthetic as SS

VITALS = mimic_data.VITALS
REPORT_HOURS = (1, 3, 6)
OBS_HOURS = (0, 1, 3, 6)


@torch.no_grad()
def per_moment_all_vitals(model, test, stats, device, moments, batch_size=128, seed=0):
    """Paired NEE 0.1 minus 0, all vitals, original units -> {(row, t): (H, V)}."""
    S, H = IM.N_SAMPLES, IM.HORIZON_B
    std = torch.tensor(stats["std"], dtype=torch.float32)
    by_t = {}
    for i, t in moments:
        by_t.setdefault(t, []).append(i)
    out = {}
    for t, rows in sorted(by_t.items()):
        for chunk in torch.tensor(rows).split(batch_size):
            x, mask, delta, a = (getattr(test, k)[chunk] for k in ("x", "mask", "delta", "a"))
            B = len(chunk)
            arms = []
            for dose in (0.0, IM.DOSE):
                plan = a[:, t:t + H].clone()
                plan[..., IM.NEE] = dose
                torch.manual_seed(seed + t)                                     # same random numbers in both arms
                sim = imagine(model, IM.repeat(x[:, :t + 1], S, device), IM.repeat(mask[:, :t + 1], S, device),
                              IM.repeat(delta[:, :t + 1], S, device), IM.repeat(a[:, :t], S, device),
                              IM.repeat(plan, S, device))                       # (B*S, H, V) z-units
                arms.append(sim.view(B, S, H, -1).cpu() * std)                  # differences need std only
            d = (arms[1] - arms[0]).mean(1)                                     # (B, H, V)
            for k, i in enumerate(chunk.tolist()):
                out[(i, t)] = d[k]
    return out


def summarise(per_a, per_b=None):
    """a (or a minus b, paired) -> per vital: mean, p10, p90 per hour."""
    keys = [k for k in per_a if per_b is None or k in per_b]
    if not keys:
        return dict(n=0)
    d = torch.stack([per_a[k] - (per_b[k] if per_b is not None else 0) for k in keys])   # (M, H, V)
    return dict(n=len(keys), **{v: dict(mean=d[..., j].mean(0).tolist(), p10=d[..., j].quantile(0.1, dim=0).tolist(),
                                        p90=d[..., j].quantile(0.9, dim=0).tolist())
                                for j, v in enumerate(VITALS)})


def verdict(effect, noise):
    """Per vital and hour: beyond the noise floor?
    strict: the effect's 10-90% range lies entirely outside the noise floor's 10-90% range.
    mean:   the effect's mean lies outside the noise floor's 10-90% range."""
    out = {}
    for v in VITALS:
        e, n = effect[v], noise[v]
        out[v] = dict(strict=[(elo > nhi) or (ehi < nlo) for elo, ehi, nlo, nhi in zip(e["p10"], e["p90"], n["p10"], n["p90"])],
                      mean=[(m > nhi) or (m < nlo) for m, nlo, nhi in zip(e["mean"], n["p10"], n["p90"])])
    return out


def raw_observed(test, stats, moments):
    out = {}
    nee = test.a[..., IM.NEE].numpy()
    started = np.array([nee[i, t] > 0 for i, t in moments])
    for j, v in enumerate(VITALS):
        value = test.x[..., j].numpy() * stats["std"][j] + stats["mean"][j]
        measured = test.mask[..., j].numpy() > 0
        by_h = {}
        for h in OBS_HOURS:
            vals = {g: [value[i, t + h] for (i, t), s in zip(moments, started) if s == g and measured[i, t + h]]
                    for g in (True, False)}
            s_, n_ = vals[True], vals[False]
            by_h[str(h)] = dict(started=float(np.mean(s_)) if s_ else None, not_started=float(np.mean(n_)) if n_ else None,
                                difference=(float(np.mean(s_)) - float(np.mean(n_))) if s_ and n_ else None,
                                n_started=len(s_), n_not_started=len(n_))
        out[v] = by_h
    return dict(n_started=int(started.sum()), n_not_started=int((~started).sum()), by_vital=out)


def run_all(icu_dir, weights, work, tmp, device="cpu", max_stays=0, b_vital="MAP", b_limit=IM.MAP_LIMIT):
    R = Runner(work, tmp)
    S = dict(per={})

    def extract():
        real = mimic_data.extract(icu_dir, max_stays)
        npz = R.tmp / "mimic_real.npz"
        np.savez_compressed(npz, **real)
        S["sets"], S["stats"] = mimic_data.load_splits_mimic(npz)
        S["moments"] = IM.find_moments(S["sets"]["test"], S["stats"], b_vital, b_limit)
        R.save("extract.json", dict(n_stays=int(real["x"].shape[0]), n_moments=len(S["moments"])))
        return f"{len(S['moments'])} moments"

    R.step("extract", extract)

    for tag in weights:
        def fn(tag=tag):
            S["per"][tag] = per_moment_all_vitals(SS.load_model(weights[tag], device), S["sets"]["test"],
                                                  S["stats"], device, S["moments"])
            R.save(f"effect_{tag}.json", summarise(S["per"][tag]))
        R.step(f"partb_{tag}", fn, needs=["extract"])

    def raw():
        R.save("raw_observed.json", raw_observed(S["sets"]["test"], S["stats"], S["moments"]))

    R.step("raw", raw, needs=["extract"])

    def final():
        noise = summarise(S["per"]["real_s1"], S["per"]["real_s0"])
        res = dict(n_moments=len(S["moments"]), report_hours=list(REPORT_HOURS), vitals=VITALS,
                   noise_floor=noise, effects={}, beyond_noise={})
        for tag in S["per"]:
            if tag == "real_s1":
                continue
            eff = summarise(S["per"][tag])
            res["effects"][tag] = eff
            res["beyond_noise"][tag] = verdict(eff, noise)
        R.save("negative_control.json", res)
        for tag, eff in res["effects"].items():
            for v in VITALS:
                print(f"{tag:8} {v:5} effect 1/3/6 h " + " ".join(f"{eff[v]['mean'][h - 1]:+.3f}" for h in REPORT_HOURS)
                      + " | noise p10..p90 at 6 h " + f"{noise[v]['p10'][5]:+.3f}..{noise[v]['p90'][5]:+.3f}"
                      + " | beyond (mean) " + str([res["beyond_noise"][tag][v]["mean"][h - 1] for h in REPORT_HOURS]))

    R.step("final", final, needs=["partb_real_s0", "partb_real_s1"])
    return R
