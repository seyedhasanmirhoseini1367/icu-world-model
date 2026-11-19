"""
stage14b_runner.py - one unattended Kaggle GPU run for Stage 14b (pre-registered in LEARNING.md).

The leak-fixed JEPA model (model_jepa_b: target encoder without action input) against act_now
and the Stage 14 JEPA checkpoints, all three measured in this job on the same moments with the
same, unchanged evaluation code (Stages 7, 8A, 9, 10, 12b), plus the KL per predicted hour.

Steps (each isolated; results saved as soon as a step finishes; summary.json after every step):
  extract, train_real_s0 (+ its evaluations), ref (act_now models), ref14 (Stage 14 JEPA checkpoints),
  train_k30, train_h15, train_k100, train_k10, train_real_s1, final
New file; the Stage 14 runner and all shared code are only imported.
"""
import json
import time

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
import model_jepa_b as MJB        # noqa: E402
import semisynthetic as SS        # noqa: E402
import train as T                 # noqa: E402
import train_jepa as TJ           # noqa: E402
import train_jepa_b as TJB        # noqa: E402
from overnight_stage11 import Runner, spread                          # noqa: E402
from stage14_jepa_runner import (G_HIDDEN, KS, REPORT_HOURS, MAE_TOLERANCE, ABLATION_FACTOR,   # noqa: E402
                                 DATA_KEYS, loaded_as, _nee_zero_increase, _mean_share)

JEPA_RUNS = (("real", 0), ("k30", 0), ("h15", 0), ("k100", 0), ("k10", 0), ("real", 1))
# design key, run-tag prefix, colour, label
DESIGNS = (("act_now", "an", "tab:blue", "act_now VAE (current default)"),
           ("jepa", "jepa", "tab:red", "JEPA (14)"),
           ("jepa_b", "jepab", "tab:green", "JEPA-14b (no action in target)"))
JEPA14_TAGS = ("jepa_real", "jepa_real_s1", "jepa_k10", "jepa_k30", "jepa_k100", "jepa_h15")


def run_all(icu_dir, weights, jepa14, work, tmp, device="cpu", max_stays=0, max_epochs=None,
            b_vital="MAP", b_limit=imagine_mimic.MAP_LIMIT, total_budget_h=10.5, train_budget_h=1.25):
    """weights: act_now checkpoints by tag; jepa14: {tag: (model_path, probe_path)} for Stage 14 JEPA."""
    R = Runner(work, tmp, total_budget_h, train_budget_h)
    T.OUT, T.FIG, T.DEVICE = R.work, R.work / "figures", device
    if max_epochs:
        T.MAX_EPOCHS = max_epochs
    S = dict(npz={}, sets={}, loaders={}, per={}, partb={}, runs={}, jepa={}, ref=dict(weights), moments=None,
             clips=None, onestep={}, part_a={}, ablation={}, kl={})
    kw = dict(n_vitals=len(mimic_data.VITALS), n_actions=len(mimic_data.ACTIONS))

    def sets(key):
        if key not in S["sets"]:
            S["sets"][key] = mimic_data.load_splits_mimic(S["npz"][key])
        return S["sets"][key]

    def loaders(key):
        if key not in S["loaders"]:
            S["loaders"][key] = mimic_data.make_loaders_mimic(S["npz"][key], batch_size=T.BATCH_SIZE)[0]
        return S["loaders"][key]

    def part_b(tag, model, key):
        ds, stats = sets(key)
        b = imagine_mimic.part_b(model, ds["test"], stats, device, vital=b_vital, limit=b_limit,
                                 keep_per_moment=True, moments=S["moments"])
        S["per"][tag] = b.pop("_per_moment", {})
        S["partb"][tag] = b
        return b

    def kl_of(tag, raw_model, key):
        S["kl"][tag] = TJB.kl_profile(raw_model, loaders(key)["val"], device)
        return S["kl"][tag]

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

    def full_real_eval(tag, model_gpu, model_cpu):
        npz = str(S["npz"]["real"])
        with loaded_as(evaluate_mimic, model_cpu):
            S["onestep"][tag] = evaluate_mimic.evaluate(npz, "unused")
        ds, stats = sets("real")
        S["part_a"][tag] = imagine_mimic.part_a(model_gpu, ds["test"], stats, device)
        with loaded_as(ablate_mimic, model_gpu):
            S["ablation"][f"{tag}@real"] = ablate_mimic.run(npz, "unused", device=device)

    def ablation_k30(tag, model_gpu):
        with loaded_as(ablate_mimic, model_gpu):
            S["ablation"][f"{tag}@k30"] = ablate_mimic.run(str(S["npz"]["k30"]), "unused", device=device)

    def evaluate_jepa(tag, key, seed, gpu, cpu_loader):
        """The Stage 14 evaluation set for one JEPA-type model (gpu: JEPAForEval on the device)."""
        b = part_b(tag, gpu, key)
        evals = dict(partb=b)
        if key == "real" and seed == 0:
            full_real_eval(tag, gpu, cpu_loader())
            evals.update(onestep=S["onestep"][tag], part_a=S["part_a"][tag], ablation=S["ablation"][f"{tag}@real"])
        if key == "k30":
            ablation_k30(tag, gpu)
            evals["ablation_k30"] = S["ablation"][f"{tag}@k30"]
        if key in ("real", "k30") and seed == 0:
            evals["kl_hours"] = kl_of(tag, gpu.model, key)
        return evals, b

    # ---------------------------------------------------------------- JEPA-14b trainings
    def train_one(key, seed):
        tag = f"jepab_{key}" + (f"_s{seed}" if seed else "")
        left = R.total_s - (time.time() - R.t_start) - R.reserve_s
        if left < 0.3 * 3600:
            raise RuntimeError(f"not enough time left ({left / 3600:.2f} h) for {tag}")
        T.TIME_BUDGET_S = min(R.train_s, left)
        spec = dict(real=None, k10=10.0, k30=30.0, k100=100.0, h15=C.K_MMHG)[key]
        model, history, run_tag = TJB.train_jepa_b(
            f"mimic_jepab_{key}", seed=seed, loaders=loaders(key), model_kwargs=kw,
            wandb_config=dict(stage="14b", data=key, semisynthetic=spec is not None,
                              injected_k_mmhg_per_dose=spec or 0.0,
                              confounder_g_mmhg=float(G_HIDDEN) if key == "h15" else 0.0,
                              confounder_kind="hidden" if key == "h15" else None,
                              nee_clip_mcg_kg_min=mimic_data.NEE_CLIP, fluid_clip_l_per_hour=mimic_data.FLUID_CLIP_L,
                              **(S["clips"] or {})))
        info = TJB.summary_of(history)
        _, probe_hist = TJB.train_probe(model, loaders(key), run_tag, first_step=len(history))
        info.update(data=key, seed=seed, model_file=f"model_{run_tag}.pt", probe_file=f"model_{run_tag}_probe.pt",
                    probe_epochs=len(probe_hist), probe_val_nll=min(h["val_nll"] for h in probe_hist))
        S["runs"][tag] = info
        S["jepa"][tag] = (R.work / info["model_file"], R.work / info["probe_file"])
        R.save(f"train_{tag}.json", info)
        print(f"{tag}: kept epoch {info['checkpoint_epoch']} of {info['epochs']}, val pred {info['val_pred']:.3f}, "
              f"KL {info['val_kl']:.2f} ({100 * info['kl_hours']['share_below_free_bits']:.0f}% of hours below free bits), "
              f"target emb std min {info['embedding_std']['target']['min']:.3f}, "
              f"collapsed dims {info['embedding_std']['target']['n_collapsed']}; probe val NLL {info['probe_val_nll']:.3f}")
        gpu = MJB.load_jepa_b_for_eval(*S["jepa"][tag], device=device, **kw)
        evals, b = evaluate_jepa(tag, key, seed, gpu,
                                 lambda: MJB.load_jepa_b_for_eval(*S["jepa"][tag], device="cpu", **kw))
        R.save(f"eval_{tag}.json", evals)
        summary = dict(checkpoint_epoch=info["checkpoint_epoch"], epochs=info["epochs"], probe_val_nll=info["probe_val_nll"])
        if b.get("mean_diff"):
            summary.update({f"effect/imagined_{h}h": b["mean_diff"][h - 1] for h in REPORT_HOURS})
        T.finish_wandb(summary)

    R.step("train_real_s0", lambda: train_one("real", 0), needs=["extract"])

    # ---------------------------------------------------------------- act_now references (as in Stage 14)
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
        kl_of("an_real", load("an_real"), "real")
        kl_of("an_k30", load("an_k30"), "k30")
        R.save("ref_act_now.json", dict(partb={t: S["partb"][t] for t in S["partb"] if t.startswith("an_")},
                                        onestep=S["onestep"]["an_real"], part_a=S["part_a"]["an_real"],
                                        ablation={k: v for k, v in S["ablation"].items() if k.startswith("an_")},
                                        kl_hours={t: S["kl"][t] for t in ("an_real", "an_k30")}))

    R.step("ref", ref, needs=["extract"])

    # ---------------------------------------------------------------- Stage 14 JEPA checkpoints, re-evaluated here
    def ref14():
        out = {}
        for tag in JEPA14_TAGS:
            key = tag.replace("jepa_", "").replace("_s1", "")
            seed = 1 if tag.endswith("_s1") else 0
            mp, pp = jepa14[tag]
            gpu = MJ.load_jepa_for_eval(mp, pp, device=device, **kw)
            out[tag], _ = evaluate_jepa(tag, key, seed, gpu, lambda: MJ.load_jepa_for_eval(mp, pp, device="cpu", **kw))
        R.save("ref_jepa14.json", out)

    R.step("ref14", ref14, needs=["extract"])
    for key, seed in JEPA_RUNS[1:]:
        R.step(f"train_{key}" + (f"_s{seed}" if seed else ""), (lambda k=key, s=seed: train_one(k, s)), needs=["extract"])

    # ---------------------------------------------------------------- final tables, verdict, figures
    def final():
        H = list(range(1, imagine_mimic.HORIZON_B + 1))
        dose = imagine_mimic.DOSE
        g = lambda t: S["per"].get(t, {})       # noqa: E731
        res = dict(horizons=H, n_moments=len(S["moments"] or []), runs=S["runs"])
        res["noise_floor"] = {d: spread(g(f"{p}_real_s1"), g(f"{p}_real")) for d, p, _, _ in DESIGNS}
        res["recovery"] = {d: {str(k): SS.recovered_stats(g(f"{p}_k{k}"), g(f"{p}_real"),
                                                          SS.injected_effect(dose, H, k=float(k))) for k in KS}
                           for d, p, _, _ in DESIGNS}
        res["k30_hour1_b_minus_14"] = spread(g("jepab_k30"), g("jepa_k30"))
        true = C.true_effect(dose, H)
        bias = [-G_HIDDEN * u for u in C.dose_arm_u(H, "hidden")]
        g0 = dict(act_now="an_g0", jepa="jepa_k30", jepa_b="jepab_k30")
        res["stage12b"] = dict(true=true, predicted_bias=bias, predicted_learned=[t + b for t, b in zip(true, bias)],
                               learned={d: spread(g(f"{p}_h15"), g(f"{p}_real")) for d, p, _, _ in DESIGNS},
                               confounding_shift={d: spread(g(f"{p}_h15"), g(g0[d])) for d, p, _, _ in DESIGNS})
        res["onestep"] = {d: S["onestep"].get(f"{p}_real") for d, p, _, _ in DESIGNS}
        res["part_a"] = {d: S["part_a"].get(f"{p}_real") for d, p, _, _ in DESIGNS}
        res["ablation"] = {f"{d}_{w}": S["ablation"].get(f"{p}_{w}@{w}") for d, p, _, _ in DESIGNS for w in ("real", "k30")}
        res["kl_hours"] = {f"{d}_{w}": S["kl"].get(f"{p}_{w}") for d, p, _, _ in DESIGNS for w in ("real", "k30")}
        res["verdict"] = verdict(res)
        R.save("stage14b_metrics.json", res)
        print("verdict:", json.dumps(res["verdict"]))
        fig = R.work / "figures"
        for fn, name in ((plot_recovery, "recovery"), (plot_12b, "confounding"), (plot_horizon, "horizon"),
                         (plot_kl_hours, "kl_hours")):
            try:
                fn(res, fig / f"stage14b_{name}.png")
            except Exception as e:                       # a figure must not lose the metrics
                print(f"figure {name} failed: {type(e).__name__}: {e}")
        plot_collapse(R.work, fig / "stage14b_collapse.png")

    R.step("final", final, needs=["ref", "ref14", "train_real_s0"])
    print(f"\nall steps done in {R.elapsed_h():.2f} h")
    return R


# ---------------------------------------------------------------- verdict

def verdict(res):
    out = dict(rules=dict(hour1_factor=ABLATION_FACTOR, ablation_factor=ABLATION_FACTOR, mae_tolerance=MAE_TOLERANCE))
    rec = res["recovery"]
    hour1 = {d: (rec[d]["30"]["recovered_mean"][0] if rec[d]["30"].get("n_paired") else None) for d in ("jepa", "jepa_b")}
    abl = {d: _nee_zero_increase(res["ablation"].get(f"{d}_k30")) for d in ("act_now", "jepa", "jepa_b")}
    share = {d: _mean_share(rec[d].get("30")) for d in ("act_now", "jepa", "jepa_b")}
    mae = {d: (res["onestep"][d]["metrics"]["MAP"]["prior"]["mae"] if res["onestep"].get(d) else None)
           for d in ("act_now", "jepa", "jepa_b")}
    out.update(k30_hour1_recovered=hour1, ablation_k30_nee_zero_increase_pct=abl, k30_mean_share_1_3_6h=share,
               map_mae=mae)
    runs = res["runs"]
    collapsed = [t for t in ("jepab_real", "jepab_k30") if t in runs
                 and runs[t]["embedding_std"]["target"]["n_collapsed"] > 32]
    out["collapsed_runs"] = collapsed
    needed = [hour1["jepa"], hour1["jepa_b"], abl["jepa"], abl["jepa_b"], share["act_now"], share["jepa_b"],
              mae["act_now"], mae["jepa_b"]]
    if collapsed or None in needed:
        out["verdict"] = "inconclusive: " + ("embedding collapse" if collapsed else "missing results")
        return out
    if hour1["jepa"] <= 0 or abl["jepa"] <= 0:
        out["verdict"] = "inconclusive: JEPA-14 hour-1 effect or ablation increase not positive"
        return out
    r1a = hour1["jepa_b"] <= ABLATION_FACTOR * hour1["jepa"]
    r1b = abl["jepa_b"] < ABLATION_FACTOR * abl["jepa"]
    r2 = share["jepa_b"] <= share["act_now"]
    r3 = mae["jepa_b"] > (1 + MAE_TOLERANCE) * mae["act_now"]
    out.update(r1a_hour1_recovery_drops=r1a, r1b_ablation_drops=r1b, r1_hour1_response_drops=r1a and r1b,
               r2_recovery_not_above_act_now=r2, r3_forecasts_worse=r3)
    if r1a and r1b and r2 and r3:
        out["verdict"] = "confirmed"
    elif not r1a and not r1b:
        out["verdict"] = "rejected"
    else:
        out["verdict"] = "partly"
    return out


# ---------------------------------------------------------------- figures

def plot_recovery(res, path):
    hours = [0] + res["horizons"]
    fig, axes = plt.subplots(1, len(KS), figsize=(5.5 * len(KS), 4.5))
    for ax, k in zip(axes, KS):
        inj = None
        for d, _, color, label in DESIGNS:
            r = res["recovery"][d].get(str(k), {})
            inj = r.get("injected", inj)
            if r.get("n_paired"):
                ax.fill_between(hours, [0] + r["recovered_p10"], [0] + r["recovered_p90"], color=color, alpha=0.12)
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
        ax.legend(fontsize=6)
    fig.suptitle(f"Recovery of a planted effect: act_now, JEPA (14), JEPA-14b (n = {res['n_moments']} moments)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_12b(res, path):
    s = res["stage12b"]
    hours = [0] + res["horizons"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(hours, [0] + s["true"], "k--", lw=2, label="true causal effect")
    ax.plot(hours, [0] + s["predicted_learned"], ":", color="0.3", lw=2, label="observational (true + bias)")
    for d, _, color, label in DESIGNS:
        r = s["learned"][d]
        if r.get("n_paired"):
            ax.fill_between(hours, [0] + r["p10"], [0] + r["p90"], color=color, alpha=0.12)
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
        naive = None
        for d, _, color, label in DESIGNS:
            r = (pa.get(d) or {}).get(v)
            if not r:
                continue
            axes[0, col].plot(pa[d]["horizons"], r["mae_model"], "o-", color=color, label=label)
            axes[1, col].plot(pa[d]["horizons"], np.array(r["coverage"]) * 100, "o-", color=color, label=label)
            naive = r["mae_naive"]
        if naive:
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


def plot_kl_hours(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, w in zip(axes, ("real", "k30")):
        for d, _, color, label in DESIGNS:
            k = res["kl_hours"].get(f"{d}_{w}")
            if k:
                ax.plot(range(len(k["by_hour"])), k["by_hour"], "-", color=color,
                        label=f"{label}: mean {k['mean']:.2f}, {100 * k['share_below_free_bits']:.0f}% of hours < free bits")
        ax.axhline(T.FREE_BITS, color="0.5", ls=":", label="free-bits floor (1 nat)")
        ax.set_yscale("log")
        ax.set_title(f"KL per predicted hour, by hour of stay ({w} data, validation)")
        ax.set_xlabel("hour of stay t (predicting t + 1)")
        ax.set_ylabel("KL, nats (summed over z dims)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_collapse(work, path):
    files = sorted(f for f in work.glob("history_mimic_jepab_*.json") if not f.name.endswith("_probe.json"))
    if not files:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for f in files:
        hist = json.loads(f.read_text())
        ep = [h["epoch"] for h in hist]
        name = f.stem.replace("history_mimic_", "")
        axes[0].plot(ep, [h["val"]["embedding_std"]["target"]["min"] for h in hist], label=name)
        axes[1].plot(ep, [h["val"]["embedding_std"]["target"]["median"] for h in hist], label=name)
        axes[2].plot(ep, [100 * h["val"]["kl_hours"]["share_below_free_bits"] for h in hist], label=name)
    for ax, t in zip(axes[:2], ("min", "median")):
        ax.axhline(TJ.COLLAPSE_STD, color="0.5", ls=":")
        ax.set_title(f"target embedding: {t} per-dimension std (validation)")
    axes[2].set_title("% of validation hours with KL below free bits")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
