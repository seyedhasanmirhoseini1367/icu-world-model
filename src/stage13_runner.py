"""
stage13_runner.py - one unattended Kaggle GPU run for Stage 13: let the model see a start.

Hypothesis (written before running, LEARNING.md Stage 13): if the model is made able to predict
the hour after a vasopressor start, it will become biased under hidden confounding.
One change: act_now gets [a_t, a_t - a_{t-1}] (WorldModel(act_delta=True)).

Steps (each isolated; results saved as soon as a step finishes; summary.json after every step):
  extract     MIMIC extraction once; real data, K = 30 (G = 0) and Stage 12b G = 15 data (tmp only)
  fit_current step 1: one-step fit around real starts for the current models
  ref         Part B of the current models (+ noise floor from the two real-data seeds)
  train_k30 / train_h15 / train_real   act_delta models, seed 0, same hyperparameters,
              each followed by Part B and the start fit
  onestep     one-step error on real data, current vs act_delta real-data model
  final       verdict numbers, tables and figures
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

import confounded as C            # noqa: E402
import evaluate_mimic             # noqa: E402
import imagine_mimic              # noqa: E402
import mimic_data                 # noqa: E402
import semisynthetic as SS        # noqa: E402
import stage12_checks as X        # noqa: E402
import train as T                 # noqa: E402
from overnight_stage11 import Runner, checkpoint_epoch, spread   # noqa: E402

REPORT_HOURS = (1, 3, 6)
G_HIDDEN = 15
DATA = {"real": None, "k30": ("ramp", 0), "h15": ("hidden", G_HIDDEN)}


def run_all(icu_dir, weights, work, tmp, device="cpu", max_stays=0, max_epochs=None,
            b_vital="MAP", b_limit=imagine_mimic.MAP_LIMIT, total_budget_h=10.5, train_budget_h=1.5):
    R = Runner(work, tmp, total_budget_h, train_budget_h)
    T.OUT, T.FIG, T.DEVICE = R.work, R.work / "figures", device
    if max_epochs:
        T.MAX_EPOCHS = max_epochs
    S = dict(npz={}, sets={}, per={}, partb={}, fit={}, models=dict(weights), clips=None, moments=None)
    dose = imagine_mimic.DOSE
    H = list(range(1, imagine_mimic.HORIZON_B + 1))
    true = C.true_effect(dose, H)
    predicted_bias = [-G_HIDDEN * u for u in C.dose_arm_u(H, "hidden")]          # U0 = 0 for this kind

    def sets(key):
        if key not in S["sets"]:
            S["sets"][key] = mimic_data.load_splits_mimic(S["npz"][key])
        return S["sets"][key]

    def part_b(tag, key):
        ds, stats = sets(key)
        b = imagine_mimic.part_b(SS.load_model(S["models"][tag], device), ds["test"], stats, device,
                                 vital=b_vital, limit=b_limit, keep_per_moment=True, moments=S["moments"])
        S["per"][tag] = b.pop("_per_moment", {})
        S["partb"][tag] = b
        return b

    def fit(tag, key):
        ds, stats = sets(key)
        S["fit"][f"{tag}@{key}"] = X.start_fit(SS.load_model(S["models"][tag], device), ds["test"], stats, device)
        return S["fit"][f"{tag}@{key}"]

    # ---------------------------------------------------------------- extraction
    def extract():
        real = mimic_data.extract(icu_dir, max_stays)
        for key, spec in DATA.items():
            S["npz"][key] = R.tmp / f"mimic_{key}.npz"
            arrays = real if spec is None else C.inject_confounded(real, k=C.K_MMHG, g=float(spec[1]), kind=spec[0])
            np.savez_compressed(S["npz"][key], **arrays)
        S["clips"] = mimic_data.clip_counts(S["npz"]["real"])
        ds, stats = sets("real")
        S["moments"] = imagine_mimic.find_moments(ds["test"], stats, b_vital, b_limit)
        R.save("extract.json", dict(n_stays=int(real["x"].shape[0]), n_moments=len(S["moments"]),
                                    clipped=S["clips"], data=list(DATA)))
        return f"{len(S['moments'])} moments"

    R.step("extract", extract)

    def print_fit(name, r):
        print(f"{name}: d " + " ".join(f"{d:+d}" for d in r["d"]))
        print("   measured  " + " ".join(f"{v:+6.2f}" for v in r["measured_change"]))
        print("   predicted " + " ".join(f"{v:+6.2f}" for v in r["predicted_change"]))

    def fit_current():
        for tag, key in (("real_s0", "real"), ("cur_k30", "k30"), ("cur_h15", "h15")):
            print_fit(f"{tag} on {key}", fit(tag, key))
        R.save("step1_start_fit_current.json", S["fit"])

    R.step("fit_current", fit_current, needs=["extract"])

    def ref():
        for tag, key in (("real_s0", "real"), ("real_s1", "real"), ("cur_k30", "k30"), ("cur_h15", "h15")):
            part_b(tag, key)
        S["noise"] = spread(S["per"]["real_s1"], S["per"]["real_s0"])
        R.save("noise_floor.json", S["noise"])
        R.save("ref_partb.json", {t: S["partb"][t] for t in ("real_s0", "real_s1", "cur_k30", "cur_h15")})

    R.step("ref", ref, needs=["extract"])

    # ---------------------------------------------------------------- act_delta trainings
    def train_step(key):
        def fn():
            left = R.total_s - (time.time() - R.t_start) - R.reserve_s
            if left < 0.3 * 3600:
                raise RuntimeError(f"not enough time left ({left / 3600:.2f} h) for {key}")
            T.TIME_BUDGET_S = min(R.train_s, left)
            loaders, _ = mimic_data.make_loaders_mimic(S["npz"][key], batch_size=T.BATCH_SIZE)
            name = f"mimic_s13_delta_{key}"
            spec = DATA[key]
            _, history, _ = T.train(
                name, seed=0, loaders=loaders,
                model_kwargs=dict(n_vitals=len(mimic_data.VITALS), n_actions=len(mimic_data.ACTIONS),
                                  act_now=True, act_delta=True),
                wandb_config=dict(stage="13", variant="act_now + delta (a_t - a_{t-1})", data=key,
                                  semisynthetic=spec is not None,
                                  injected_k_mmhg_per_dose=0.0 if spec is None else C.K_MMHG,
                                  confounder_g_mmhg=0.0 if spec is None else float(spec[1]),
                                  confounder_kind=None if spec is None else spec[0],
                                  nee_clip_mcg_kg_min=mimic_data.NEE_CLIP,
                                  fluid_clip_l_per_hour=mimic_data.FLUID_CLIP_L, **(S["clips"] or {})))
            tag = f"delta_{key}"
            S["models"][tag] = R.work / f"model_{name}.pt"
            ep = checkpoint_epoch(history)
            best = next(h for h in history if h["epoch"] == ep)
            R.save(f"train_{tag}.json", dict(
                data=key, checkpoint_epoch=ep, epochs=len(history),
                seconds_per_epoch_median=float(np.median([h["seconds"] for h in history])),
                val_loss=best["val"]["loss"], val_recon=best["val"]["recon"], val_kl=best["val"]["kl"],
                val_kl_dims=best["val"]["kl_dims"], stopped_early=len(history) - ep >= T.PATIENCE))
            b = part_b(tag, key)
            R.save(f"partb_{tag}.json", b)
            f = fit(tag, key)
            R.save(f"start_fit_{tag}.json", f)
            print_fit(f"{tag} on {key}", f)
            summary = dict(checkpoint_epoch=ep, epochs=len(history))
            if b.get("mean_diff"):
                summary.update({f"effect/imagined_{h}h": b["mean_diff"][h - 1] for h in REPORT_HOURS})
            T.finish_wandb(summary)
            return f"checkpoint epoch {ep} of {len(history)}"
        return fn

    for key in ("k30", "h15", "real"):
        R.step(f"train_{key}", train_step(key), needs=["extract"])

    def onestep():
        out = {}
        for tag in ("real_s0", "delta_real"):
            out[tag] = evaluate_mimic.evaluate(str(S["npz"]["real"]), str(S["models"][tag]))
            print(f"--- one-step, {tag}\n" + evaluate_mimic.table(out[tag]))
        R.save("onestep_real.json", out)

    R.step("onestep", onestep, needs=["train_real"])

    # ---------------------------------------------------------------- final
    def final():
        def eff(a, b):
            return spread(S["per"][a], S["per"][b]) if a in S["per"] and b in S["per"] else None
        res = dict(horizons=H, report_hours=list(REPORT_HOURS), dose=dose, k=C.K_MMHG, g=G_HIDDEN,
                   true=true, predicted_bias=predicted_bias,
                   predicted_learned=[t + b for t, b in zip(true, predicted_bias)],
                   noise_floor=S.get("noise"), n_moments=len(S["moments"] or []),
                   learned=dict(cur_k30=eff("cur_k30", "real_s0"), delta_k30=eff("delta_k30", "real_s0"),
                                delta_k30_own_ref=eff("delta_k30", "delta_real"),
                                cur_h15=eff("cur_h15", "real_s0"), delta_h15=eff("delta_h15", "real_s0"),
                                delta_real_vs_cur_real=eff("delta_real", "real_s0")),
                   confounding_shift=dict(current=eff("cur_h15", "cur_k30"), delta=eff("delta_h15", "delta_k30")),
                   start_fit=S["fit"])
        # verdict inputs, as defined before the run
        f = S["fit"]
        i1 = X.D_FIT.index(1)
        if "delta_h15@h15" in f and "delta_k30@k30" in f:
            meas_drop = f["delta_h15@h15"]["measured_change"][i1] - f["delta_k30@k30"]["measured_change"][i1]
            pred_drop = f["delta_h15@h15"]["predicted_change"][i1] - f["delta_k30@k30"]["predicted_change"][i1]
            res["hour1_drop_after_start"] = dict(measured=meas_drop, predicted_delta=pred_drop,
                                                 captured_share=pred_drop / meas_drop if meas_drop else None)
        if "cur_h15@h15" in f and "cur_k30@k30" in f:
            res["hour1_drop_after_start_current"] = dict(
                measured=f["cur_h15@h15"]["measured_change"][i1] - f["cur_k30@k30"]["measured_change"][i1],
                predicted=f["cur_h15@h15"]["predicted_change"][i1] - f["cur_k30@k30"]["predicted_change"][i1])
        R.save("stage13_metrics.json", res)
        plot_fit(S["fit"], R.work / "figures" / "stage13_start_fit.png")
        plot_effect(res, R.work / "figures" / "stage13_confounding.png")

    R.step("final", final, needs=["ref"])
    print(f"\nall steps done in {R.elapsed_h():.2f} h")
    return R


def plot_fit(fit, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, key, title in ((axes[0], "k30", "K = 30, G = 0 (drug only)"),
                           (axes[1], "h15", "Stage 12b, G = 15 (hidden confounder)")):
        meas = next((v for k, v in fit.items() if k.endswith(f"@{key}")), None)
        if meas is None:
            continue
        ax.plot(meas["d"], meas["measured_change"], "k-o", lw=2, label="measured change")
        for tag, color, label in ((f"cur_{key}", "tab:blue", "current model, predicted"),
                                  (f"delta_{key}", "tab:orange", "act_delta model, predicted")):
            r = fit.get(f"{tag}@{key}")
            if r:
                ax.plot(r["d"], r["predicted_change"], "o--", color=color, label=label)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(0, color="0.5", lw=0.8, ls=":")
        ax.set_title(title)
        ax.set_xlabel("hour relative to the vasopressor start (0 = start hour)")
        ax.set_ylabel("change of MAP' into this hour (mmHg)")
        ax.legend(fontsize=7)
    fig.suptitle("One-step fit around real vasopressor starts (test split)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_effect(res, path):
    hours = [0] + res["horizons"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hours, [0] + res["true"], "k--", lw=2, label="true causal effect K*e")
    ax.plot(hours, [0] + res["predicted_learned"], ":", color="tab:red", lw=2,
            label="predicted if learning E[MAP' | history, a]")
    nf = res.get("noise_floor") or {}
    if nf.get("n_paired"):
        ax.fill_between(hours, [0] + nf["p10"], [0] + nf["p90"], color="0.6", alpha=0.3, label="noise floor, 10-90%")
    for tag, color, label in (("cur_h15", "tab:blue", "current model"), ("delta_h15", "tab:orange", "act_delta model")):
        r = res["learned"].get(tag)
        if r and r.get("n_paired"):
            ax.fill_between(hours, [0] + r["p10"], [0] + r["p90"], color=color, alpha=0.2)
            ax.plot(hours, [0] + r["mean"], "o-", color=color, label=f"{label}, learned (10-90%)")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("hours after the decision hour")
    ax.set_ylabel("MAP, NEE 0.1 minus 0 (mmHg)")
    ax.set_title(f"Stage 12b data, G = {res['g']}: learned vs true vs observational (n = {res['n_moments']})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
