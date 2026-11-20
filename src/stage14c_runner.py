"""
stage14c_runner.py - one unattended Kaggle GPU run for Stage 14c (pre-registered in LEARNING.md).

JEPA-14b (target encoder without action input) with the KL weight beta tuned so that z is used.
  Step 1 calibration: beta in CAL_BETAS on real data, at most CAL_EPOCHS epochs each; a beta fits if
         its validation KL is within KL_RANGE nats/h and fewer than MAX_SHARE_BELOW of hours are below
         the free-bits floor; the fitting beta with KL closest to KL_TARGET is used. If none fits,
         the step fails and every later step is skipped (the job stops, calibration.json is kept).
  Step 2 with that beta: real (seeds 0, 1), K = 10 / 30 / 100, 12b G = 15; the same benchmarks as
         Stage 14b, side by side with act_now, JEPA-14 and JEPA-14b (all re-evaluated in this job).
beta is set on train.BETA_KL for the duration of one training only (as Stage 13c did).
New file; the Stage 14 / 14b runners and all shared code are only imported.
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
from stage14_jepa_runner import (G_HIDDEN, KS, MAE_TOLERANCE, DATA_KEYS, REPORT_HOURS,   # noqa: E402
                                 loaded_as, _nee_zero_increase, _mean_share)

CAL_BETAS = (0.3, 0.1, 0.03)
CAL_EPOCHS = 30
KL_RANGE = (1.5, 4.0)          # nats per predicted hour: "z is used, about 2-3 like the VAE"
KL_TARGET = 2.5
MAX_SHARE_BELOW = 0.25         # share of validation hours with KL below the free-bits floor
JEPA_RUNS = (("real", 0), ("k30", 0), ("h15", 0), ("k100", 0), ("k10", 0), ("real", 1))
DESIGNS = (("act_now", "an", "tab:blue", "act_now VAE (current default)"),
           ("jepa", "jepa", "tab:red", "JEPA (14)"),
           ("jepa_b", "jepab", "tab:green", "JEPA-14b (beta 1)"),
           ("jepa_c", "jepac", "tab:purple", "JEPA-14c (beta tuned)"))
REF_TAGS = ("real", "real_s1", "k10", "k30", "k100", "h15")


def beta_tag(beta):
    return f"{beta:g}".replace(".", "p")


def choose_beta(cal, kl_range=KL_RANGE, max_share=MAX_SHARE_BELOW):
    """cal: list of dict(beta, kl, share_below) -> chosen beta or None (pre-registered rule)."""
    fits = [c for c in cal if c.get("kl") is not None and kl_range[0] <= c["kl"] <= kl_range[1]
            and c["share_below"] < max_share]
    return min(fits, key=lambda c: abs(c["kl"] - KL_TARGET))["beta"] if fits else None


def run_all(icu_dir, weights, jepa14, jepa14b, work, tmp, device="cpu", max_stays=0, max_epochs=None,
            b_vital="MAP", b_limit=imagine_mimic.MAP_LIMIT, total_budget_h=10.5, train_budget_h=1.25,
            kl_range=KL_RANGE, max_share=MAX_SHARE_BELOW):
    """weights: act_now checkpoints by tag; jepa14 / jepa14b: {tag: (model_path, probe_path)}.
    kl_range / max_share are the pre-registered values; they are parameters only for the local smoke test."""
    R = Runner(work, tmp, total_budget_h, train_budget_h)
    T.OUT, T.FIG, T.DEVICE = R.work, R.work / "figures", device
    if max_epochs:
        T.MAX_EPOCHS = max_epochs
    S = dict(npz={}, sets={}, loaders={}, per={}, partb={}, runs={}, jepa={}, ref=dict(weights), moments=None,
             clips=None, onestep={}, part_a={}, ablation={}, kl={}, beta=None, calibration=[])
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

    def wandb_cfg(key, extra):
        spec = dict(real=None, k10=10.0, k30=30.0, k100=100.0, h15=C.K_MMHG)[key]
        return dict(stage="14c", data=key, semisynthetic=spec is not None, injected_k_mmhg_per_dose=spec or 0.0,
                    confounder_g_mmhg=float(G_HIDDEN) if key == "h15" else 0.0,
                    confounder_kind="hidden" if key == "h15" else None,
                    nee_clip_mcg_kg_min=mimic_data.NEE_CLIP, fluid_clip_l_per_hour=mimic_data.FLUID_CLIP_L,
                    **(S["clips"] or {}), **extra)

    def train_with_beta(beta, name, seed, key, extra, max_ep=None):
        left = R.total_s - (time.time() - R.t_start) - R.reserve_s
        if left < 0.3 * 3600:
            raise RuntimeError(f"not enough time left ({left / 3600:.2f} h) for {name}")
        T.TIME_BUDGET_S = min(R.train_s, left)
        saved_beta, saved_max = T.BETA_KL, T.MAX_EPOCHS
        T.BETA_KL = beta
        if max_ep:
            T.MAX_EPOCHS = min(max_ep, T.MAX_EPOCHS)
        try:
            return TJB.train_jepa_b(name, seed=seed, loaders=loaders(key), model_kwargs=kw,
                                    wandb_config=wandb_cfg(key, dict(beta_kl_stage14c=beta, **extra)))
        finally:
            T.BETA_KL, T.MAX_EPOCHS = saved_beta, saved_max

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

    # ---------------------------------------------------------------- step 1: calibration
    def calibrate():
        for beta in CAL_BETAS:
            try:
                _, history, tag = train_with_beta(beta, f"mimic_jepac_cal_b{beta_tag(beta)}", 0, "real",
                                                  dict(calibration=True), max_ep=CAL_EPOCHS)
                s = TJB.summary_of(history)
                rec = dict(beta=beta, tag=tag, kl=s["kl_hours"]["mean"], share_below=s["kl_hours"]["share_below_free_bits"],
                           val_kl=s["val_kl"], val_pred=s["val_pred"], checkpoint_epoch=s["checkpoint_epoch"],
                           epochs=s["epochs"], kl_per_dim=s["kl_hours"]["per_dim"],
                           target_std_min=s["embedding_std"]["target"]["min"],
                           n_collapsed=s["embedding_std"]["target"]["n_collapsed"])
                T.finish_wandb(dict(kl_mean=rec["kl"], share_below_free_bits=rec["share_below"]))
            except Exception as e:                       # one failed beta must not lose the others
                rec = dict(beta=beta, error=f"{type(e).__name__}: {str(e)[:300]}")
                T.finish_wandb()
            S["calibration"].append(rec)
            print(f"calibration beta {beta}: " + (f"KL {rec['kl']:.2f} nats/h, {100 * rec['share_below']:.0f}% of hours "
                                                   f"below free bits (kept epoch {rec['checkpoint_epoch']})"
                                                   if "kl" in rec else rec["error"]), flush=True)
            R.save("calibration.json", dict(rule=dict(kl_range=list(kl_range), kl_target=KL_TARGET, max_share_below=max_share),
                                            runs=S["calibration"], chosen_beta=None))
        S["beta"] = choose_beta(S["calibration"], kl_range, max_share)
        R.save("calibration.json", dict(rule=dict(kl_range=list(kl_range), kl_target=KL_TARGET, max_share_below=max_share),
                                        runs=S["calibration"], chosen_beta=S["beta"]))
        if S["beta"] is None:
            raise RuntimeError("no beta fits the pre-registered calibration rule: stopping (later steps skipped)")
        return f"chosen beta {S['beta']}"

    R.step("calibrate", calibrate, needs=["extract"])

    # ---------------------------------------------------------------- evaluation helpers (as in Stage 14b)
    def full_real_eval(tag, model_gpu, model_cpu):
        npz = str(S["npz"]["real"])
        with loaded_as(evaluate_mimic, model_cpu):
            S["onestep"][tag] = evaluate_mimic.evaluate(npz, "unused")
        ds, stats = sets("real")
        S["part_a"][tag] = imagine_mimic.part_a(model_gpu, ds["test"], stats, device)
        with loaded_as(ablate_mimic, model_gpu):
            S["ablation"][f"{tag}@real"] = ablate_mimic.run(npz, "unused", device=device)

    def evaluate_jepa(tag, key, seed, gpu, cpu_loader):
        b = part_b(tag, gpu, key)
        evals = dict(partb=b)
        if key == "real" and seed == 0:
            full_real_eval(tag, gpu, cpu_loader())
            evals.update(onestep=S["onestep"][tag], part_a=S["part_a"][tag], ablation=S["ablation"][f"{tag}@real"])
        if key == "k30":
            with loaded_as(ablate_mimic, gpu):
                S["ablation"][f"{tag}@k30"] = ablate_mimic.run(str(S["npz"]["k30"]), "unused", device=device)
            evals["ablation_k30"] = S["ablation"][f"{tag}@k30"]
        if key in ("real", "k30") and seed == 0:
            evals["kl_hours"] = kl_of(tag, gpu.model, key)
        return evals, b

    # ---------------------------------------------------------------- step 2: full trainings with the chosen beta
    def train_one(key, seed):
        tag = f"jepac_{key}" + (f"_s{seed}" if seed else "")
        model, history, run_tag = train_with_beta(S["beta"], f"mimic_jepac_{key}", seed, key, dict(calibration=False))
        info = TJB.summary_of(history)
        _, probe_hist = TJB.train_probe(model, loaders(key), run_tag, first_step=len(history))
        info.update(data=key, seed=seed, beta_kl=S["beta"], model_file=f"model_{run_tag}.pt",
                    probe_file=f"model_{run_tag}_probe.pt", probe_epochs=len(probe_hist),
                    probe_val_nll=min(h["val_nll"] for h in probe_hist))
        S["runs"][tag] = info
        S["jepa"][tag] = (R.work / info["model_file"], R.work / info["probe_file"])
        R.save(f"train_{tag}.json", info)
        print(f"{tag}: beta {S['beta']}, kept epoch {info['checkpoint_epoch']} of {info['epochs']}, "
              f"val pred {info['val_pred']:.3f}, KL {info['val_kl']:.2f} "
              f"({100 * info['kl_hours']['share_below_free_bits']:.0f}% of hours below free bits), "
              f"collapsed dims {info['embedding_std']['target']['n_collapsed']}; probe val NLL {info['probe_val_nll']:.3f}")
        gpu = MJB.load_jepa_b_for_eval(*S["jepa"][tag], device=device, **kw)
        evals, b = evaluate_jepa(tag, key, seed, gpu,
                                 lambda: MJB.load_jepa_b_for_eval(*S["jepa"][tag], device="cpu", **kw))
        R.save(f"eval_{tag}.json", evals)
        summary = dict(checkpoint_epoch=info["checkpoint_epoch"], epochs=info["epochs"], probe_val_nll=info["probe_val_nll"])
        if b.get("mean_diff"):
            summary.update({f"effect/imagined_{h}h": b["mean_diff"][h - 1] for h in REPORT_HOURS})
        T.finish_wandb(summary)

    R.step("train_real_s0", lambda: train_one("real", 0), needs=["calibrate"])

    # ---------------------------------------------------------------- references, re-evaluated here
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

    def ref_jepa(prefix, paths, loader, out_name):
        out = {}
        for suffix in REF_TAGS:
            tag = f"{prefix}_{suffix}"
            key, seed = suffix.replace("_s1", ""), (1 if suffix.endswith("_s1") else 0)
            mp, pp = paths[tag]
            gpu = loader(mp, pp, device=device, **kw)
            out[tag], _ = evaluate_jepa(tag, key, seed, gpu, lambda: loader(mp, pp, device="cpu", **kw))
        R.save(out_name, out)

    R.step("ref", ref, needs=["calibrate"])
    R.step("ref14", lambda: ref_jepa("jepa", jepa14, MJ.load_jepa_for_eval, "ref_jepa14.json"), needs=["calibrate"])
    R.step("ref14b", lambda: ref_jepa("jepab", jepa14b, MJB.load_jepa_b_for_eval, "ref_jepa14b.json"), needs=["calibrate"])
    for key, seed in JEPA_RUNS[1:]:
        R.step(f"train_{key}" + (f"_s{seed}" if seed else ""), (lambda k=key, s=seed: train_one(k, s)), needs=["calibrate"])

    # ---------------------------------------------------------------- final tables, verdict, figures
    def final():
        H = list(range(1, imagine_mimic.HORIZON_B + 1))
        dose = imagine_mimic.DOSE
        g = lambda t: S["per"].get(t, {})       # noqa: E731
        res = dict(horizons=H, n_moments=len(S["moments"] or []), beta=S["beta"], calibration=S["calibration"],
                   runs=S["runs"])
        res["noise_floor"] = {d: spread(g(f"{p}_real_s1"), g(f"{p}_real")) for d, p, _, _ in DESIGNS}
        res["recovery"] = {d: {str(k): SS.recovered_stats(g(f"{p}_k{k}"), g(f"{p}_real"),
                                                          SS.injected_effect(dose, H, k=float(k))) for k in KS}
                           for d, p, _, _ in DESIGNS}
        res["k30_c_minus_b"] = spread(g("jepac_k30"), g("jepab_k30"))
        true = C.true_effect(dose, H)
        bias = [-G_HIDDEN * u for u in C.dose_arm_u(H, "hidden")]
        g0 = dict(act_now="an_g0", jepa="jepa_k30", jepa_b="jepab_k30", jepa_c="jepac_k30")
        res["stage12b"] = dict(true=true, predicted_bias=bias, predicted_learned=[t + b for t, b in zip(true, bias)],
                               learned={d: spread(g(f"{p}_h15"), g(f"{p}_real")) for d, p, _, _ in DESIGNS},
                               confounding_shift={d: spread(g(f"{p}_h15"), g(g0[d])) for d, p, _, _ in DESIGNS})
        res["onestep"] = {d: S["onestep"].get(f"{p}_real") for d, p, _, _ in DESIGNS}
        res["part_a"] = {d: S["part_a"].get(f"{p}_real") for d, p, _, _ in DESIGNS}
        res["ablation"] = {f"{d}_{w}": S["ablation"].get(f"{p}_{w}@{w}") for d, p, _, _ in DESIGNS for w in ("real", "k30")}
        res["kl_hours"] = {f"{d}_{w}": S["kl"].get(f"{p}_{w}") for d, p, _, _ in DESIGNS for w in ("real", "k30")}
        res["verdict"] = verdict(res, kl_range, max_share)
        R.save("stage14c_metrics.json", res)
        print("verdict:", json.dumps(res["verdict"]))
        fig = R.work / "figures"
        for fn, name in ((plot_recovery, "recovery"), (plot_12b, "confounding"), (plot_horizon, "horizon"),
                         (plot_kl_hours, "kl_hours"), (plot_calibration, "calibration")):
            try:
                fn(res, fig / f"stage14c_{name}.png")
            except Exception as e:                       # a figure must not lose the metrics
                print(f"figure {name} failed: {type(e).__name__}: {e}")
        try:
            plot_collapse(R.work, fig / "stage14c_collapse.png")
        except Exception as e:
            print(f"figure collapse failed: {type(e).__name__}: {e}")

    R.step("final", final, needs=["ref", "ref14", "ref14b", "train_real_s0"])
    print(f"\nall steps done in {R.elapsed_h():.2f} h")
    return R


# ---------------------------------------------------------------- verdict

def verdict(res, kl_range=KL_RANGE, max_share=MAX_SHARE_BELOW):
    out = dict(rules=dict(kl_range=list(kl_range), max_share_below=max_share, mae_tolerance=MAE_TOLERANCE),
               beta=res.get("beta"))
    rec = res["recovery"]
    share = {d: _mean_share(rec[d].get("30")) for d in ("act_now", "jepa", "jepa_b", "jepa_c")}
    mae = {d: (res["onestep"][d]["metrics"]["MAP"]["prior"]["mae"] if res["onestep"].get(d) else None)
           for d in ("act_now", "jepa", "jepa_b", "jepa_c")}
    hour1 = {d: (rec[d]["30"]["recovered_mean"][0] if rec[d]["30"].get("n_paired") else None)
             for d in ("act_now", "jepa", "jepa_b", "jepa_c")}
    abl = {d: _nee_zero_increase(res["ablation"].get(f"{d}_k30")) for d in ("act_now", "jepa", "jepa_b", "jepa_c")}
    out.update(k30_mean_share_1_3_6h=share, map_mae=mae, k30_hour1_recovered=hour1,
               ablation_k30_nee_zero_increase_pct=abl)
    runs = res["runs"]
    kh = (runs.get("jepac_real") or {}).get("kl_hours") or {}
    out["jepac_real_kl"] = dict(mean=kh.get("mean"), share_below_free_bits=kh.get("share_below_free_bits"))
    collapsed = [t for t in ("jepac_real", "jepac_k30") if t in runs
                 and runs[t]["embedding_std"]["target"]["n_collapsed"] > 32]
    out["collapsed_runs"] = collapsed
    if res.get("beta") is None:
        out["verdict"] = "inconclusive: no beta fits the calibration rule"
        return out
    if collapsed or None in (share["act_now"], share["jepa_c"], mae["act_now"], mae["jepa_c"], kh.get("mean")):
        out["verdict"] = "inconclusive: " + ("embedding collapse" if collapsed else "missing results")
        return out
    pre = kl_range[0] <= kh["mean"] <= kl_range[1] and kh["share_below_free_bits"] < max_share
    out["precondition_z_used"] = pre
    if not pre:
        out["verdict"] = "inconclusive: z off-target in the full run"
        return out
    r2 = share["jepa_c"] < share["act_now"]
    r3 = mae["jepa_c"] > (1 + MAE_TOLERANCE) * mae["act_now"]
    out.update(r2_recovers_less=r2, r3_forecasts_worse=r3)
    out["verdict"] = "confirmed" if r2 and r3 else ("rejected" if not r2 and not r3 else "partly")
    return out


# ---------------------------------------------------------------- figures

def plot_recovery(res, path):
    hours = [0] + res["horizons"]
    fig, axes = plt.subplots(1, len(KS), figsize=(5.5 * len(KS), 4.6))
    for ax, k in zip(axes, KS):
        inj = None
        for d, _, color, label in DESIGNS:
            r = res["recovery"][d].get(str(k), {})
            inj = r.get("injected", inj)
            if r.get("n_paired"):
                ax.fill_between(hours, [0] + r["recovered_p10"], [0] + r["recovered_p90"], color=color, alpha=0.1)
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
    fig.suptitle(f"Recovery of a planted effect: act_now, JEPA-14, 14b, 14c (beta {res.get('beta')}), "
                 f"n = {res['n_moments']} moments")
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
            ax.fill_between(hours, [0] + r["p10"], [0] + r["p90"], color=color, alpha=0.1)
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
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


def plot_calibration(res, path):
    cal = [c for c in res.get("calibration", []) if "kl" in c]
    if not cal:
        return
    rule = res["verdict"]["rules"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    b = [c["beta"] for c in cal]
    axes[0].plot(b, [c["kl"] for c in cal], "o-", color="tab:purple")
    axes[0].axhspan(*rule["kl_range"], color="tab:green", alpha=0.12, label="pre-registered KL range")
    axes[0].axhline(KL_TARGET, color="tab:green", ls=":", label=f"target {KL_TARGET}")
    axes[0].set_ylabel("validation KL, nats per hour")
    axes[1].plot(b, [100 * c["share_below"] for c in cal], "o-", color="tab:purple")
    axes[1].axhline(100 * rule["max_share_below"], color="tab:green", ls=":", label="pre-registered limit")
    axes[1].set_ylabel("% of hours with KL below free bits")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("beta")
        ax.legend(fontsize=7)
    fig.suptitle(f"Stage 14c calibration (at most {CAL_EPOCHS} epochs each); chosen beta: {res.get('beta')}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_collapse(work, path):
    files = sorted(f for f in work.glob("history_mimic_jepac_*.json") if not f.name.endswith("_probe.json"))
    if not files:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for f in files:
        hist = json.loads(f.read_text())
        ep = [h["epoch"] for h in hist]
        name = f.stem.replace("history_mimic_", "")
        axes[0].plot(ep, [h["val"]["embedding_std"]["target"]["min"] for h in hist], label=name)
        axes[1].plot(ep, [h["val"]["kl_hours"]["mean"] for h in hist], label=name)
        axes[2].plot(ep, [100 * h["val"]["kl_hours"]["share_below_free_bits"] for h in hist], label=name)
    axes[0].axhline(TJ.COLLAPSE_STD, color="0.5", ls=":")
    axes[0].set_title("target embedding: min per-dimension std (validation)")
    axes[1].set_title("mean KL per hour (validation)")
    axes[2].set_title("% of validation hours with KL below free bits")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
