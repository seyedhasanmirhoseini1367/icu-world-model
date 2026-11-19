"""
semisynthetic.py - Stage 10: a semi-synthetic MIMIC benchmark with a KNOWN vasopressor effect.

Everything stays real (histories, actions, missingness, patient splits) except observed
MAP, to which a known effect of the vasopressor dose is added:

    e_0     = 0
    e_{t+1} = e_t + (1 - exp(-1/TAU_H)) * (NEE_t - e_t)       first-order lag, exact for hourly steps
    MAP'_t  = MAP_t + K * e_t                                  only where MAP was measured

NEE_t is the hourly norepinephrine-equivalent dose (mcg/kg/min, already capped at 2) given in
hour t, so it first changes MAP at hour t+1, the same timing the model uses (a_t -> x_{t+1}).
With K = 100 mmHg per mcg/kg/min and TAU_H = 1 h, a constant 0.1 adds
K * 0.1 * (1 - exp(-h)) mmHg after h hours: +6.3 at 1 h, +9.5 at 3 h, +10.0 at 6 h.

run_stage10() trains the SAME model with the SAME hyperparameters on this data, then compares
the learned counterfactual effect (Stage 8 Part B) with the real-data model's, against the
injected effect, and reruns the Stage 9 ablation. Only aggregate numbers and figures are kept.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablate_mimic               # noqa: E402
import imagine_mimic              # noqa: E402
import mimic_data                 # noqa: E402
import train as T                 # noqa: E402
from model import load_world_model  # noqa: E402

K_MMHG = 100.0                    # mmHg per mcg/kg/min
TAU_H = 1.0                       # hours
MAP = mimic_data.VITALS.index("MAP")
NEE = mimic_data.ACTIONS.index("NEE mcg/kg/min")


def lagged_dose(nee, tau=TAU_H):
    """nee (N, T) hourly dose -> e (N, T), the dose seen through a first-order lag."""
    alpha = 1.0 - math.exp(-1.0 / tau)
    e = np.zeros_like(nee)
    for t in range(nee.shape[1] - 1):
        e[:, t + 1] = e[:, t] + alpha * (nee[:, t] - e[:, t])
    return e


def inject(data, k=K_MMHG, tau=TAU_H):
    """Copy of an extracted dataset (dict from mimic_data.extract) with the known effect
    added to measured MAP only. Missing MAP stays missing."""
    out = {key: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for key, v in data.items()}
    e = lagged_dose(out["a"][..., NEE], tau)                                  # (N, T)
    out["x"][..., MAP] = out["x"][..., MAP] + k * e                          # NaN + number stays NaN
    out["injected_k"], out["injected_tau_h"] = k, tau
    return out


def injected_effect(dose, hours, k=K_MMHG, tau=TAU_H):
    """Known effect of switching from 0 to a constant `dose` at the decision hour."""
    return [k * dose * (1.0 - math.exp(-h / tau)) for h in hours]


def load_model(path, device):
    return load_world_model(path, len(mimic_data.VITALS), len(mimic_data.ACTIONS), device)


def plot_effects(res, path):
    h = [0] + res["horizons"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(h, [0] + res["injected"], "k--", lw=2, label="injected (known) effect")
    ax.plot(h, [0] + res["semisynthetic_model"]["mean_diff"], "o-", label="learned, semi-synthetic model")
    ax.plot(h, [0] + res["real_model"]["mean_diff"], "s-", label="learned, real-data model")
    ax.plot(h, [0] + res["recovered"], "^:", label="recovered = semi-synthetic minus real-data model")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("hours after the decision hour")
    ax.set_ylabel("MAP difference, NEE 0.1 minus 0 (mmHg)")
    ax.set_title(f"Semi-synthetic benchmark (n = {res['n_moments']} moments)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_stage10(icu_dir, cleaned_model, work, device="cpu", tmp="/tmp", max_stays=0,
                max_epochs=None, time_budget_h=10.0):
    work, tmp = Path(work), Path(tmp)
    (work / "figures").mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    real = mimic_data.extract(icu_dir, max_stays)
    real_npz, semi_npz = tmp / "mimic_real.npz", tmp / "mimic_semisyn.npz"
    np.savez_compressed(real_npz, **real)
    np.savez_compressed(semi_npz, **inject(real))
    clips = mimic_data.clip_counts(real_npz)

    # 1. Same model, same hyperparameters, semi-synthetic data.
    T.OUT, T.FIG, T.DEVICE, T.TIME_BUDGET_S = work, work / "figures", device, time_budget_h * 3600
    if max_epochs:
        T.MAX_EPOCHS = max_epochs
    loaders, _ = mimic_data.make_loaders_mimic(semi_npz, batch_size=T.BATCH_SIZE)
    _, history, _ = T.train(
        "mimic_semisyn", loaders=loaders,
        model_kwargs=dict(n_vitals=len(mimic_data.VITALS), n_actions=len(mimic_data.ACTIONS)),
        wandb_config=dict(semisynthetic=True, injected_k_mmhg_per_dose=K_MMHG, injected_tau_h=TAU_H,
                          nee_clip_mcg_kg_min=mimic_data.NEE_CLIP,
                          fluid_clip_l_per_hour=mimic_data.FLUID_CLIP_L, **clips))
    best = min(history, key=lambda h: h["val"]["loss"])
    T.plot_curves({"mimic_semisyn": history})
    T.plot_kl_dims({"mimic_semisyn": best["val"]["kl_dims"]})

    # 2. Counterfactual check on the same moments for both models.
    real_sets, real_stats = mimic_data.load_splits_mimic(real_npz)
    semi_sets, semi_stats = mimic_data.load_splits_mimic(semi_npz)
    b_real = imagine_mimic.part_b(load_model(cleaned_model, device), real_sets["test"], real_stats, device)
    b_semi = imagine_mimic.part_b(load_model(work / "model_mimic_semisyn.pt", device),
                                  semi_sets["test"], semi_stats, device)
    # Histories before the decision hour have no vasopressor, so both datasets select the same
    # moments; the counts are recorded to confirm it.
    hours = b_semi["horizons"]
    injected = injected_effect(imagine_mimic.DOSE, hours)
    have = b_semi["n_moments"] and b_real["n_moments"]
    recovered = [s - r for s, r in zip(b_semi["mean_diff"], b_real["mean_diff"])] if have else []

    # 3. Stage 9 ablation on the semi-synthetic model.
    ablation = ablate_mimic.run(semi_npz, work / "model_mimic_semisyn.pt", device=device)

    result = dict(injected_k_mmhg_per_dose=K_MMHG, injected_tau_h=TAU_H, dose=imagine_mimic.DOSE,
                  clipped=clips, best_epoch=best["epoch"], epochs=len(history),
                  seconds_per_epoch_median=float(np.median([h["seconds"] for h in history])),
                  best_val=dict(loss=best["val"]["loss"], recon=best["val"]["recon"], kl=best["val"]["kl"],
                                kl_dims=best["val"]["kl_dims"]),
                  n_moments=b_semi["n_moments"], n_moments_real=b_real["n_moments"], horizons=hours,
                  injected=injected, recovered=recovered,
                  recovered_share=[r / i for r, i in zip(recovered, injected)],
                  real_model=b_real, semisynthetic_model=b_semi, ablation_semisynthetic=ablation)
    (work / "stage10_metrics.json").write_text(json.dumps(result, indent=2))
    if recovered:
        plot_effects(result, work / "figures" / "semisynthetic_effect.png")

    summary = dict(best_epoch=best["epoch"], **clips)
    for h, inj, rec in zip(hours, injected, recovered):
        summary[f"effect/injected_{h}h"], summary[f"effect/recovered_{h}h"] = inj, rec
    T.finish_wandb(summary)
    for f in (real_npz, semi_npz):
        f.unlink()
    return result


def report(r):
    lines = [f"semi-synthetic K = {r['injected_k_mmhg_per_dose']} mmHg per mcg/kg/min, tau = {r['injected_tau_h']} h; "
             f"best epoch {r['best_epoch']} of {r['epochs']}, {r['seconds_per_epoch_median']:.0f} s/epoch; "
             f"val recon {r['best_val']['recon']:.3f}, KL {r['best_val']['kl']:.3f}",
             f"Part B on {r['n_moments']:,} moments (real-data selection: {r['n_moments_real']:,}), dose {r['dose']}:",
             f"{'hour':>5} {'injected':>9} {'semi model':>11} {'real model':>11} {'recovered':>10} {'share':>6}"]
    for i, h in enumerate(r["horizons"] if r["recovered"] else []):
        lines.append(f"{h:>5} {r['injected'][i]:9.2f} {r['semisynthetic_model']['mean_diff'][i]:11.2f} "
                     f"{r['real_model']['mean_diff'][i]:11.2f} {r['recovered'][i]:10.2f} {r['recovered_share'][i]:6.0%}")
    s = r["semisynthetic_model"]
    if r["recovered"]:
        lines.append("semi model spread at 1/3/6 h: sample SD " + " ".join(f"{s['sample_sd'][h - 1]:.2f}" for h in (1, 3, 6))
                     + " | patient SD " + " ".join(f"{s['moment_sd'][h - 1]:.2f}" for h in (1, 3, 6)))
    lines.append("ablation on the semi-synthetic model:")
    lines.append(ablate_mimic.report(r["ablation_semisynthetic"]))
    return "\n".join(lines)


# ---------------------------------------------------------------- Stage 10b: detection threshold

def recovered_stats(per_semi, per_real, injected):
    """Pair moments by (row, t) and summarise the recovered effect (semi minus real) per hour.
    Clearly distinguishable at an hour = the 10-90% range across patients lies above 0."""
    keys = [k for k in per_semi if k in per_real]
    if not keys:
        return dict(n_paired=0, injected=injected)
    rec = torch.stack([per_semi[k] - per_real[k] for k in keys])            # (M, H)
    mean = rec.mean(0)
    p10, p90 = rec.quantile(0.1, dim=0), rec.quantile(0.9, dim=0)
    return dict(n_paired=len(keys), injected=injected, recovered_mean=mean.tolist(),
                recovered_p10=p10.tolist(), recovered_p90=p90.tolist(),
                recovered_share=[m / i for m, i in zip(mean.tolist(), injected)],
                share_moments_positive=(rec > 0).float().mean(0).tolist(),
                clearly_distinguishable=(p10 > 0).tolist())


def plot_threshold(res, path):
    ks = res["k_values"]
    hours = [0] + res["horizons"]
    fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4.2), squeeze=False)
    for ax, k in zip(axes[0], ks):
        r = res["by_k"][str(k)]
        ax.plot(hours, [0] + r["injected"], "k--", lw=2, label="injected")
        ax.fill_between(hours, [0] + r["recovered_p10"], [0] + r["recovered_p90"], alpha=0.25,
                        label="recovered, 10-90% across patients")
        ax.plot(hours, [0] + r["recovered_mean"], "o-", label="recovered, mean")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(f"K = {k} mmHg per mcg/kg/min")
        ax.set_xlabel("hours after the decision hour")
        ax.set_ylabel("MAP difference, NEE 0.1 minus 0 (mmHg)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Detection threshold: recovered vs injected effect (n = {res['n_moments']} moments)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_stage10b(icu_dir, real_model, k100_model, work, device="cpu", tmp="/tmp", ks_to_train=(10, 30),
                 max_stays=0, max_epochs=None, time_budget_h=5.0):
    """Train the same model on K = 10 and 30 (same seed, hyperparameters), reuse the K = 100 model,
    and compare the recovered effect on the same moments, paired per moment."""
    work, tmp = Path(work), Path(tmp)
    (work / "figures").mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    real = mimic_data.extract(icu_dir, max_stays)
    real_npz = tmp / "mimic_real.npz"
    np.savez_compressed(real_npz, **real)
    clips = mimic_data.clip_counts(real_npz)
    sets, stats = mimic_data.load_splits_mimic(real_npz)
    b_real = imagine_mimic.part_b(load_model(real_model, device), sets["test"], stats, device, keep_per_moment=True)
    hours = b_real["horizons"]

    T.OUT, T.FIG, T.DEVICE, T.TIME_BUDGET_S = work, work / "figures", device, time_budget_h * 3600
    if max_epochs:
        T.MAX_EPOCHS = max_epochs
    by_k, training = {}, {}
    for k in sorted(set(ks_to_train) | {100}):
        npz = tmp / f"mimic_semisyn_k{int(k)}.npz"
        np.savez_compressed(npz, **inject(real, k=float(k)))
        if k in ks_to_train:
            loaders, _ = mimic_data.make_loaders_mimic(npz, batch_size=T.BATCH_SIZE)
            _, history, _ = T.train(
                f"mimic_semisyn_k{int(k)}", loaders=loaders,
                model_kwargs=dict(n_vitals=len(mimic_data.VITALS), n_actions=len(mimic_data.ACTIONS)),
                wandb_config=dict(semisynthetic=True, injected_k_mmhg_per_dose=float(k), injected_tau_h=TAU_H,
                                  nee_clip_mcg_kg_min=mimic_data.NEE_CLIP,
                                  fluid_clip_l_per_hour=mimic_data.FLUID_CLIP_L, **clips))
            best = min(history, key=lambda h: h["val"]["loss"])
            training[str(k)] = dict(best_epoch=best["epoch"], epochs=len(history),
                                    seconds_per_epoch_median=float(np.median([h["seconds"] for h in history])),
                                    val_recon=best["val"]["recon"], val_kl=best["val"]["kl"])
            model_path = work / f"model_mimic_semisyn_k{int(k)}.pt"
        else:
            model_path = k100_model
        k_sets, k_stats = mimic_data.load_splits_mimic(npz)
        b = imagine_mimic.part_b(load_model(model_path, device), k_sets["test"], k_stats, device, keep_per_moment=True)
        injected = injected_effect(imagine_mimic.DOSE, hours, k=float(k))
        r = recovered_stats(b["_per_moment"], b_real["_per_moment"], injected)
        r.update(n_moments=b["n_moments"], learned_mean=b["mean_diff"])
        by_k[str(k)] = r
        if k in ks_to_train:
            summary = {f"effect/recovered_{h}h": r["recovered_mean"][h - 1] for h in (1, 3, 6)}
            summary.update({f"effect/injected_{h}h": injected[h - 1] for h in (1, 3, 6)})
            T.finish_wandb(dict(best_epoch=training[str(k)]["best_epoch"], **summary, **clips))
        npz.unlink()

    b_real.pop("_per_moment")
    result = dict(k_values=sorted(int(k) for k in by_k), dose=imagine_mimic.DOSE, injected_tau_h=TAU_H,
                  horizons=hours, n_moments=b_real["n_moments"], real_model_mean=b_real["mean_diff"],
                  by_k=by_k, training=training, clipped=clips)
    smallest = [k for k in result["k_values"]
                if all(by_k[str(k)]["clearly_distinguishable"][h - 1] for h in (1, 3, 6))]
    result["smallest_k_clearly_distinguishable_at_1_3_6h"] = min(smallest) if smallest else None
    (work / "stage10b_metrics.json").write_text(json.dumps(result, indent=2))
    plot_threshold(result, work / "figures" / "semisynthetic_threshold.png")
    real_npz.unlink()
    return result


def report_threshold(r):
    lines = [f"{r['n_moments']:,} moments, dose {r['dose']}; real-data model effect at 1/3/6 h: "
             + " ".join(f"{r['real_model_mean'][h - 1]:+.2f}" for h in (1, 3, 6)),
             f"{'K':>4} {'hour':>4} {'injected':>9} {'recovered':>10} {'p10..p90 patients':>20} {'share':>6} "
             f"{'>0':>5} {'clear':>6}"]
    for k in r["k_values"]:
        b = r["by_k"][str(k)]
        for h in (1, 3, 6):
            i = h - 1
            lines.append(f"{k:>4} {h:>4} {b['injected'][i]:9.2f} {b['recovered_mean'][i]:10.2f} "
                         f"{b['recovered_p10'][i]:+9.2f}..{b['recovered_p90'][i]:+.2f} "
                         f"{b['recovered_share'][i]:6.0%} {b['share_moments_positive'][i]:5.0%} "
                         f"{str(b['clearly_distinguishable'][i]):>6}")
    for k, t in r["training"].items():
        lines.append(f"training K={k}: best epoch {t['best_epoch']} of {t['epochs']}, "
                     f"{t['seconds_per_epoch_median']:.0f} s/epoch, val recon {t['val_recon']:.3f}, KL {t['val_kl']:.3f}")
    lines.append(f"smallest K clearly distinguishable at 1, 3 and 6 h: {r['smallest_k_clearly_distinguishable_at_1_3_6h']}")
    return "\n".join(lines)
