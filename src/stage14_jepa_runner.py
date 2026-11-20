"""
stage14_jepa_runner.py - one unattended Kaggle GPU run for Stage 14 (pre-registered in LEARNING.md).

A JEPA-style world model (model_jepa.py, train_jepa.py) against the current default (act_now VAE),
on the same benchmarks and the same evaluation code:
  Stage 7   one-step forecast table (test split, real data)
  Stage 8A  imagination error over a 12-hour horizon (real actions fed)
  Stage 9   action ablation (real data, and K = 30 where the drug really acts)
  Stage 10  recovery of a planted effect, K = 10 / 30 / 100, with the noise floor (seed 1 - seed 0)
  Stage 12b hidden confounder G = 15: learned vs true vs observational, confounding shift
The JEPA model is evaluated through JEPAForEval (probe as the "decoder"); the shared evaluators are
called unchanged, with their load_world_model swapped for the duration of one call only.

Steps (each isolated; results saved as soon as a step finishes; summary.json after every step):
  extract, train_real_s0 (+ its evaluations), ref (act_now models), train_k30, train_h15,
  train_k100, train_k10, train_real_s1, final
"""
import time
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

import ablate_mimic               # noqa: E402
import confounded as C            # noqa: E402
import evaluate_mimic             # noqa: E402
import imagine_mimic              # noqa: E402
import mimic_data                 # noqa: E402
import model_jepa as MJ           # noqa: E402
import semisynthetic as SS        # noqa: E402
import train as T                 # noqa: E402
import train_jepa as TJ           # noqa: E402
from overnight_stage11 import Runner, spread   # noqa: E402

G_HIDDEN = 15
KS = (10, 30, 100)
REPORT_HOURS = (1, 3, 6)
MAE_TOLERANCE = 0.05          # H1: JEPA one-step MAP MAE within 5% of act_now
ABLATION_FACTOR = 0.8         # Stage 9 "lower": JEPA increase < 0.8 x act_now increase
RECOVERY_MARGIN = 0.10        # Stage 10 "lower": mean K = 30 share at 1/3/6 h at least 10 points lower
DATA_KEYS = ("real", "k10", "k30", "k100", "h15")
JEPA_RUNS = (("real", 0), ("k30", 0), ("h15", 0), ("k100", 0), ("k10", 0), ("real", 1))


@contextmanager
def loaded_as(module, model):
    """Make module.load_world_model return `model` for the duration of one call (module not edited)."""
    saved = module.load_world_model
    module.load_world_model = lambda *a, **k: model
    try:
        yield
    finally:
        module.load_world_model = saved


def run_all(icu_dir, weights, work, tmp, device="cpu", max_stays=0, max_epochs=None,
            b_vital="MAP", b_limit=imagine_mimic.MAP_LIMIT, total_budget_h=10.5, train_budget_h=1.25):
    R = Runner(work, tmp, total_budget_h, train_budget_h)
    T.OUT, T.FIG, T.DEVICE = R.work, R.work / "figures", device
    if max_epochs:
        T.MAX_EPOCHS = max_epochs
    S = dict(npz={}, sets={}, per={}, partb={}, runs={}, jepa={}, ref=dict(weights), moments=None, clips=None,
             onestep={}, part_a={}, ablation={})
    H = list(range(1, imagine_mimic.HORIZON_B + 1))
    dose = imagine_mimic.DOSE
    kw = dict(n_vitals=len(mimic_data.VITALS), n_actions=len(mimic_data.ACTIONS))

    def sets(key):
        if key not in S["sets"]:
            S["sets"][key] = mimic_data.load_splits_mimic(S["npz"][key])
        return S["sets"][key]

    def part_b(tag, model, key):
        ds, stats = sets(key)
        b = imagine_mimic.part_b(model, ds["test"], stats, device, vital=b_vital, limit=b_limit,
                                 keep_per_moment=True, moments=S["moments"])
        S["per"][tag] = b.pop("_per_moment", {})
        S["partb"][tag] = b
        return b

    # ---------------------------------------------------------------- extraction
    def extract():
        real = mimic_data.extract(icu_dir, max_stays)
        arrays = dict(real=real, k10=SS.inject(real, k=10.0), k30=SS.inject(real, k=30.0),
                      k100=SS.inject(real, k=100.0),
                      h15=C.inject_confounded(real, k=C.K_MMHG, g=float(G_HIDDEN), kind="hidden"))
        for key in DATA_KEYS:
            S["npz"][key] = R.tmp / f"mimic_{key}.npz"
            np.savez_compressed(S["npz"][key], **arrays[key])
        del arrays
        S["clips"] = mimic_data.clip_counts(S["npz"]["real"])
        ds, stats = sets("real")
        S["moments"] = imagine_mimic.find_moments(ds["test"], stats, b_vital, b_limit)
        R.save("extract.json", dict(n_stays=int(real["x"].shape[0]), n_moments=len(S["moments"]), clipped=S["clips"]))
        return f"{len(S['moments'])} moments"

    R.step("extract", extract)

    # ---------------------------------------------------------------- full evaluation of one real-data model
    def full_real_eval(tag, model_gpu, model_cpu, npz_key="real"):
        npz = str(S["npz"][npz_key])
        with loaded_as(evaluate_mimic, model_cpu):
            S["onestep"][tag] = evaluate_mimic.evaluate(npz, "unused")
        ds, stats = sets(npz_key)
        S["part_a"][tag] = imagine_mimic.part_a(model_gpu, ds["test"], stats, device)
        with loaded_as(ablate_mimic, model_gpu):
            S["ablation"][f"{tag}@real"] = ablate_mimic.run(npz, "unused", device=device)

    def ablation_k30(tag, model_gpu):
        with loaded_as(ablate_mimic, model_gpu):
            S["ablation"][f"{tag}@k30"] = ablate_mimic.run(str(S["npz"]["k30"]), "unused", device=device)

    # ---------------------------------------------------------------- JEPA trainings
    def train_one(key, seed):
        tag = f"jepa_{key}" + (f"_s{seed}" if seed else "")
        left = R.total_s - (time.time() - R.t_start) - R.reserve_s
        if left < 0.3 * 3600:
            raise RuntimeError(f"not enough time left ({left / 3600:.2f} h) for {tag}")
        T.TIME_BUDGET_S = min(R.train_s, left)
        loaders, _ = mimic_data.make_loaders_mimic(S["npz"][key], batch_size=T.BATCH_SIZE)
        spec = dict(real=None, k10=10.0, k30=30.0, k100=100.0, h15=C.K_MMHG)[key]
        model, history, run_tag = TJ.train_jepa(
            f"mimic_jepa_{key}", seed=seed, loaders=loaders, model_kwargs=kw,
            wandb_config=dict(stage="14", data=key, semisynthetic=spec is not None,
                              injected_k_mmhg_per_dose=spec or 0.0,
                              confounder_g_mmhg=float(G_HIDDEN) if key == "h15" else 0.0,
                              confounder_kind="hidden" if key == "h15" else None,
                              nee_clip_mcg_kg_min=mimic_data.NEE_CLIP, fluid_clip_l_per_hour=mimic_data.FLUID_CLIP_L,
                              **(S["clips"] or {})))
        info = TJ.summary_of(history)
        _, probe_hist = TJ.train_probe(model, loaders, run_tag, first_step=len(history))
        info.update(data=key, seed=seed, model_file=f"model_{run_tag}.pt", probe_file=f"model_{run_tag}_probe.pt",
                    probe_epochs=len(probe_hist), probe_val_nll=min(h["val_nll"] for h in probe_hist))
        S["runs"][tag] = info
        S["jepa"][tag] = (R.work / info["model_file"], R.work / info["probe_file"])
        R.save(f"train_{tag}.json", info)
        print(f"{tag}: kept epoch {info['checkpoint_epoch']} of {info['epochs']}, val pred {info['val_pred']:.3f}, "
              f"KL {info['val_kl']:.2f}, target emb std min {info['embedding_std']['target']['min']:.3f}, "
              f"collapsed dims {info['embedding_std']['target']['n_collapsed']}; probe val NLL {info['probe_val_nll']:.3f}")

        gpu = MJ.load_jepa_for_eval(*S["jepa"][tag], device=device, **kw)
        b = part_b(tag, gpu, key)
        evals = dict(partb=b)
        if key == "real" and seed == 0:
            cpu = MJ.load_jepa_for_eval(*S["jepa"][tag], device="cpu", **kw)
            full_real_eval(tag, gpu, cpu)
            evals.update(onestep=S["onestep"][tag], part_a=S["part_a"][tag], ablation=S["ablation"][f"{tag}@real"])
        if key == "k30":
            ablation_k30(tag, gpu)
            evals["ablation_k30"] = S["ablation"][f"{tag}@k30"]
        R.save(f"eval_{tag}.json", evals)
        summary = dict(checkpoint_epoch=info["checkpoint_epoch"], epochs=info["epochs"], probe_val_nll=info["probe_val_nll"])
        if b.get("mean_diff"):
            summary.update({f"effect/imagined_{h}h": b["mean_diff"][h - 1] for h in REPORT_HOURS})
        T.finish_wandb(summary)

    R.step("train_real_s0", lambda: train_one("real", 0), needs=["extract"])

    # ---------------------------------------------------------------- act_now references
    def ref():
        load = lambda t: SS.load_model(S["ref"][t], device)     # noqa: E731
        for tag, key in (("an_real", "real"), ("an_real_s1", "real"), ("an_k10", "k10"), ("an_k30", "k30"),
                         ("an_k100", "k100"), ("an_g0", "k30"), ("an_h15", "h15")):
            part_b(tag, load(tag), key)
        S["onestep"]["an_real"] = evaluate_mimic.evaluate(str(S["npz"]["real"]), str(S["ref"]["an_real"]))
        ds, stats = sets("real")
        S["part_a"]["an_real"] = imagine_mimic.part_a(load("an_real"), ds["test"], stats, device)
        S["ablation"]["an_real@real"] = ablate_mimic.run(str(S["npz"]["real"]), str(S["ref"]["an_real"]), device=device)
        S["ablation"]["an_k30@k30"] = ablate_mimic.run(str(S["npz"]["k30"]), str(S["ref"]["an_k30"]), device=device)
        R.save("ref_act_now.json", dict(partb={t: S["partb"][t] for t in S["partb"] if t.startswith("an_")},
                                        onestep=S["onestep"]["an_real"], part_a=S["part_a"]["an_real"],
                                        ablation={k: v for k, v in S["ablation"].items() if k.startswith("an_")}))

    R.step("ref", ref, needs=["extract"])
    for key, seed in JEPA_RUNS[1:]:
        R.step(f"train_{key}" + (f"_s{seed}" if seed else ""), (lambda k=key, s=seed: train_one(k, s)), needs=["extract"])

    # ---------------------------------------------------------------- final tables, verdict, figures
    def final():
        res = dict(horizons=H, n_moments=len(S["moments"] or []), runs=S["runs"])
        res["noise_floor"] = dict(act_now=spread(S["per"].get("an_real_s1", {}), S["per"].get("an_real", {})),
                                  jepa=spread(S["per"].get("jepa_real_s1", {}), S["per"].get("jepa_real", {})))
        rec = {}
        for design, p in (("act_now", "an"), ("jepa", "jepa")):
            rec[design] = {}
            for k in KS:
                inj = SS.injected_effect(dose, H, k=float(k))
                r = SS.recovered_stats(S["per"].get(f"{p}_k{k}", {}), S["per"].get(f"{p}_real", {}), inj)
                rec[design][str(k)] = r
        res["recovery"] = rec
        true = C.true_effect(dose, H)
        bias = [-G_HIDDEN * u for u in C.dose_arm_u(H, "hidden")]
        res["stage12b"] = dict(true=true, predicted_bias=bias, predicted_learned=[t + b for t, b in zip(true, bias)],
                               learned=dict(act_now=spread(S["per"].get("an_h15", {}), S["per"].get("an_real", {})),
                                            jepa=spread(S["per"].get("jepa_h15", {}), S["per"].get("jepa_real", {}))),
                               confounding_shift=dict(
                                   act_now=spread(S["per"].get("an_h15", {}), S["per"].get("an_g0", {})),
                                   jepa=spread(S["per"].get("jepa_h15", {}), S["per"].get("jepa_k30", {}))))
        res["onestep"] = dict(act_now=S["onestep"].get("an_real"), jepa=S["onestep"].get("jepa_real"))
        res["part_a"] = dict(act_now=S["part_a"].get("an_real"), jepa=S["part_a"].get("jepa_real"))
        res["ablation"] = dict(act_now_real=S["ablation"].get("an_real@real"), jepa_real=S["ablation"].get("jepa_real@real"),
                               act_now_k30=S["ablation"].get("an_k30@k30"), jepa_k30=S["ablation"].get("jepa_k30@k30"))
        res["verdict"] = verdict(res)
        R.save("stage14_metrics.json", res)
        print("verdict:", res["verdict"])
        plot_recovery(res, R.work / "figures" / "stage14_recovery.png")
        plot_12b(res, R.work / "figures" / "stage14_confounding.png")
        plot_horizon(res, R.work / "figures" / "stage14_horizon.png")
        plot_collapse(R.work, R.work / "figures" / "stage14_collapse.png")

    R.step("final", final, needs=["ref", "train_real_s0"])
    print(f"\nall steps done in {R.elapsed_h():.2f} h")
    return R


# ---------------------------------------------------------------- verdict

def _nee_zero_increase(abl):
    """Stage 9 effect: % increase of MAP MAE on NEE-active hours when NEE is set to 0."""
    if not abl:
        return None
    real, zero = abl["mae"]["real"]["nee_active_hours"]["MAP"]["mae"], abl["mae"]["nee_zero"]["nee_active_hours"]["MAP"]["mae"]
    return None if real is None or zero is None else 100.0 * (zero / real - 1.0)


def _mean_share(r):
    if not r or not r.get("n_paired"):
        return None
    return float(np.mean([r["recovered_share"][h - 1] for h in REPORT_HOURS]))


def verdict(res):
    out = dict(rules=dict(mae_tolerance=MAE_TOLERANCE, ablation_factor=ABLATION_FACTOR, recovery_margin=RECOVERY_MARGIN))
    os_ = res["onestep"]
    mae = {d: (os_[d]["metrics"]["MAP"]["prior"]["mae"] if os_.get(d) else None) for d in ("act_now", "jepa")}
    abl = {d: _nee_zero_increase(res["ablation"].get(f"{d}_k30")) for d in ("act_now", "jepa")}
    share = {d: _mean_share(res["recovery"][d].get("30")) for d in ("act_now", "jepa")}
    out.update(map_mae=mae, ablation_k30_nee_zero_increase_pct=abl, k30_mean_share_1_3_6h=share)
    runs = res["runs"]
    collapsed = [t for t in ("jepa_real", "jepa_k30") if t in runs
                 and runs[t]["embedding_std"]["target"]["n_collapsed"] > 32]
    out["collapsed_runs"] = collapsed
    if collapsed or None in mae.values() or None in abl.values() or None in share.values():
        out["verdict"] = "inconclusive: " + ("embedding collapse" if collapsed else "missing results")
        return out
    h1 = mae["jepa"] <= (1 + MAE_TOLERANCE) * mae["act_now"]
    a_lower = abl["jepa"] < ABLATION_FACTOR * abl["act_now"]
    b_lower = share["jepa"] <= share["act_now"] - RECOVERY_MARGIN
    out.update(h1_forecast_parity=h1, ablation_lower=a_lower, recovery_lower=b_lower)
    if h1 and a_lower and b_lower:
        out["verdict"] = "confirmed"
    elif not a_lower and not b_lower:
        out["verdict"] = "rejected"
    else:
        out["verdict"] = "partly"
    return out


# ---------------------------------------------------------------- figures

DESIGNS = (("act_now", "tab:blue", "act_now VAE (current default)"), ("jepa", "tab:red", "JEPA"))


def plot_recovery(res, path):
    hours = [0] + res["horizons"]
    fig, axes = plt.subplots(1, len(KS), figsize=(5 * len(KS), 4.2))
    for ax, k in zip(axes, KS):
        inj = None
        for d, color, label in DESIGNS:
            r = res["recovery"][d].get(str(k), {})
            inj = r.get("injected", inj)
            if r.get("n_paired"):
                ax.fill_between(hours, [0] + r["recovered_p10"], [0] + r["recovered_p90"], color=color, alpha=0.15)
                ax.plot(hours, [0] + r["recovered_mean"], "o-", color=color, label=f"{label}, recovered")
            nf = res["noise_floor"][d]
            if nf.get("n_paired"):
                ax.plot(hours, [0] + nf["p90"], ":", color=color, lw=1, label=f"{label}, noise floor p90")
        if inj:
            ax.plot(hours, [0] + inj, "k--", lw=2, label="injected")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(f"K = {k}")
        ax.set_xlabel("hours after the decision hour")
        ax.set_ylabel("MAP, NEE 0.1 minus 0 (mmHg)")
        ax.legend(fontsize=7)
    fig.suptitle(f"Recovery of a planted effect, JEPA vs act_now (n = {res['n_moments']} moments)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_12b(res, path):
    s = res["stage12b"]
    hours = [0] + res["horizons"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hours, [0] + s["true"], "k--", lw=2, label="true causal effect")
    ax.plot(hours, [0] + s["predicted_learned"], ":", color="tab:red", lw=2, label="observational (true + bias)")
    for d, color, label in DESIGNS:
        r = s["learned"][d]
        if r.get("n_paired"):
            ax.fill_between(hours, [0] + r["p10"], [0] + r["p90"], color=color, alpha=0.15)
            ax.plot(hours, [0] + r["mean"], "o-", color=color, label=f"{label}, learned (10-90%)")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("hours after the decision hour")
    ax.set_ylabel("MAP, NEE 0.1 minus 0 (mmHg)")
    ax.set_title(f"12b data, G = {G_HIDDEN}: learned vs true vs observational")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_horizon(res, path):
    pa = res["part_a"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, v in enumerate(("MAP", "HR")):
        for d, color, label in DESIGNS:
            r = (pa.get(d) or {}).get(v)
            if not r:
                continue
            axes[0, col].plot(pa[d]["horizons"], r["mae_model"], "o-", color=color, label=label)
            axes[1, col].plot(pa[d]["horizons"], np.array(r["coverage"]) * 100, "o-", color=color, label=label)
            naive = r["mae_naive"]
        if any((pa.get(d) or {}).get(v) for d, _, _ in DESIGNS):
            axes[0, col].plot(range(1, len(naive) + 1), naive, "s--", color="0.4", label="naive: last observed")
        axes[0, col].set_title(f"{v}: imagination error vs horizon")
        axes[0, col].set_ylabel("MAE")
        axes[1, col].axhline(80, ls=":", color="0.5")
        axes[1, col].set_ylim(0, 100)
        axes[1, col].set_title(f"{v}: inside the 10-90% band (%)")
        for ax in axes[:, col]:
            ax.set_xlabel("hours after the last real hour")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_collapse(work, path):
    import json
    files = sorted(work.glob("history_mimic_jepa_*.json"))
    files = [f for f in files if not f.name.endswith("_probe.json")]
    if not files:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for f in files:
        hist = json.loads(f.read_text())
        ep = [h["epoch"] for h in hist]
        name = f.stem.replace("history_mimic_", "")
        axes[0].plot(ep, [h["val"]["embedding_std"]["target"]["min"] for h in hist], label=name)
        axes[1].plot(ep, [h["val"]["embedding_std"]["target"]["median"] for h in hist], label=name)
    for ax, t in zip(axes, ("min", "median")):
        ax.axhline(TJ.COLLAPSE_STD, color="0.5", ls=":")
        ax.set_title(f"target embedding: {t} per-dimension std (validation)")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
