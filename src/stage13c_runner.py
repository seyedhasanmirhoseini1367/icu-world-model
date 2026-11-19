"""
stage13c_runner.py - one unattended Kaggle GPU run for Stage 13c (pre-registered in LEARNING.md).

Control for Stage 13b: act_delta with NO start weighting, beta lowered so that validation KL reaches
the 13b level (target band 5.0-8.5 nats/hour). beta = 0.3 first on K = 30; if its KL is outside the
band, adjust once (beta_new = 0.3 * KL / 6.5, clipped to 0.05-0.9) and retrain K = 30; the final beta
is used for 12b G = 15 and real data. beta is changed in train.py's module globals for the duration of
each training call only (train.py is not edited).

Steps (each isolated; results saved as soon as a step finishes; summary.json after every step):
  extract, ref (existing act_delta and 13b weighted models: Part B, start-fit, start-hour stats),
  train_k30_beta0.3, [train_k30_adjusted], train_h15, train_real, onestep, final
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
import stage13b as B              # noqa: E402
import train as T                 # noqa: E402
from overnight_stage11 import Runner, checkpoint_epoch, spread   # noqa: E402

G_HIDDEN = 15
DATA = {"real": None, "k30": ("ramp", 0), "h15": ("hidden", G_HIDDEN)}
BETA_FIRST, KL_BAND, KL_TARGET, BETA_CLIP = 0.3, (5.0, 8.5), 6.5, (0.05, 0.9)
SPECIFIC_SHIFT, GLOBAL_SHIFT, FIT_TARGET = -1.5, -4.24, 0.5
REPORT_HOURS = (1, 3, 6)


def run_all(icu_dir, weights, work, tmp, device="cpu", max_stays=0, max_epochs=None,
            b_vital="MAP", b_limit=imagine_mimic.MAP_LIMIT, total_budget_h=10.5, train_budget_h=1.5):
    R = Runner(work, tmp, total_budget_h, train_budget_h)
    T.OUT, T.FIG, T.DEVICE = R.work, R.work / "figures", device
    if max_epochs:
        T.MAX_EPOCHS = max_epochs
    S = dict(npz={}, sets={}, per={}, partb={}, fit={}, stats1a={}, models=dict(weights), moments=None, clips=None,
             beta=BETA_FIRST, runs={})
    dose = imagine_mimic.DOSE
    H = list(range(1, imagine_mimic.HORIZON_B + 1))
    true = C.true_effect(dose, H)
    predicted_bias = [-G_HIDDEN * u for u in C.dose_arm_u(H, "hidden")]
    i1 = B.D_FIT.index(1)

    def sets(key):
        if key not in S["sets"]:
            S["sets"][key] = mimic_data.load_splits_mimic(S["npz"][key])
        return S["sets"][key]

    def model(tag):
        return SS.load_model(S["models"][tag], device)

    def evaluate_model(tag, key):
        ds, stats = sets(key)
        m = model(tag)
        b = imagine_mimic.part_b(m, ds["test"], stats, device, vital=b_vital, limit=b_limit,
                                 keep_per_moment=True, moments=S["moments"])
        S["per"][tag] = b.pop("_per_moment", {})
        S["partb"][tag] = b
        S["fit"][f"{tag}@{key}"] = B.start_fit_both(m, ds["test"], stats, device)
        S["stats1a"][f"{tag}@{key}"] = B.start_hour_stats(m, ds["test"], stats, device)

    def fit_numbers(prefix):
        f, out = S["fit"], {}
        for kind, key in (("decoder_at_prior_mean", "predicted_change"), ("mean_of_50_samples", "predicted_change_samples")):
            o = {}
            if f"{prefix}_k30@k30" in f and f"{prefix}_real@real" in f:
                o["drug_part_measured"] = f[f"{prefix}_k30@k30"]["measured_change"][i1] - f[f"{prefix}_real@real"]["measured_change"][i1]
                o["drug_part_predicted"] = f[f"{prefix}_k30@k30"][key][i1] - f[f"{prefix}_real@real"][key][i1]
            if f"{prefix}_h15@h15" in f and f"{prefix}_k30@k30" in f:
                meas = f[f"{prefix}_h15@h15"]["measured_change"][i1] - f[f"{prefix}_k30@k30"]["measured_change"][i1]
                pred = f[f"{prefix}_h15@h15"][key][i1] - f[f"{prefix}_k30@k30"][key][i1]
                o.update(drop_measured=meas, drop_predicted=pred, drop_captured_share=pred / meas if meas else None)
            out[kind] = o
        return out

    # ---------------------------------------------------------------- extraction and references
    def extract():
        real = mimic_data.extract(icu_dir, max_stays)
        for key, spec in DATA.items():
            S["npz"][key] = R.tmp / f"mimic_{key}.npz"
            arrays = real if spec is None else C.inject_confounded(real, k=C.K_MMHG, g=float(spec[1]), kind=spec[0])
            np.savez_compressed(S["npz"][key], **arrays)
        S["clips"] = mimic_data.clip_counts(S["npz"]["real"])
        ds, stats = sets("real")
        S["moments"] = imagine_mimic.find_moments(ds["test"], stats, b_vital, b_limit)
        R.save("extract.json", dict(n_stays=int(real["x"].shape[0]), n_moments=len(S["moments"]), clipped=S["clips"]))
        return f"{len(S['moments'])} moments"

    R.step("extract", extract)

    def ref():
        for tag, key in (("real_s0", "real"), ("real_s1", "real"), ("delta_k30", "k30"), ("delta_h15", "h15"),
                         ("delta_real", "real"), ("w_k30", "k30"), ("w_h15", "h15"), ("w_real", "real")):
            evaluate_model(tag, key)
        S["noise"] = spread(S["per"]["real_s1"], S["per"]["real_s0"])
        R.save("ref.json", dict(noise_floor=S["noise"], start_hour_stats=S["stats1a"],
                                fit_numbers=dict(act_delta=fit_numbers("delta"), weighted=fit_numbers("w"))))

    R.step("ref", ref, needs=["extract"])

    # ---------------------------------------------------------------- control trainings
    def train_control(key, beta, tag):
        left = R.total_s - (time.time() - R.t_start) - R.reserve_s
        if left < 0.3 * 3600:
            raise RuntimeError(f"not enough time left ({left / 3600:.2f} h) for {tag}")
        T.TIME_BUDGET_S = min(R.train_s, left)
        loaders, _ = mimic_data.make_loaders_mimic(S["npz"][key], batch_size=T.BATCH_SIZE)
        name = f"mimic_s13c_beta{beta:.3f}_{key}".replace(".", "p")
        spec = DATA[key]
        saved_beta = T.BETA_KL
        T.BETA_KL = beta
        try:
            _, history, _ = T.train(
                name, seed=0, loaders=loaders,
                model_kwargs=dict(n_vitals=len(mimic_data.VITALS), n_actions=len(mimic_data.ACTIONS),
                                  act_now=True, act_delta=True),
                wandb_config=dict(stage="13c", variant="act_delta, lowered beta, no start weighting", data=key,
                                  semisynthetic=spec is not None,
                                  injected_k_mmhg_per_dose=0.0 if spec is None else C.K_MMHG,
                                  confounder_g_mmhg=0.0 if spec is None else float(spec[1]),
                                  confounder_kind=None if spec is None else spec[0],
                                  nee_clip_mcg_kg_min=mimic_data.NEE_CLIP,
                                  fluid_clip_l_per_hour=mimic_data.FLUID_CLIP_L, **(S["clips"] or {})))
        finally:
            T.BETA_KL = saved_beta
        S["models"][tag] = R.work / f"model_{name}.pt"
        ep = checkpoint_epoch(history)
        best = next(h for h in history if h["epoch"] == ep)
        info = dict(data=key, beta=beta, model_file=S["models"][tag].name, checkpoint_epoch=ep, epochs=len(history),
                    seconds_per_epoch_median=float(np.median([h["seconds"] for h in history])),
                    val_loss=best["val"]["loss"], val_recon=best["val"]["recon"], val_kl=best["val"]["kl"],
                    val_kl_dims=best["val"]["kl_dims"], stopped_early=len(history) - ep >= T.PATIENCE)
        S["runs"][tag] = info
        R.save(f"train_{tag}.json", info)
        evaluate_model(tag, key)
        R.save(f"eval_{tag}.json", dict(partb=S["partb"][tag], start_fit=S["fit"][f"{tag}@{key}"],
                                        start_hour_stats=S["stats1a"][f"{tag}@{key}"]))
        summary = dict(checkpoint_epoch=ep, epochs=len(history), beta=beta)
        if S["partb"][tag].get("mean_diff"):
            summary.update({f"effect/imagined_{h}h": S["partb"][tag]["mean_diff"][h - 1] for h in REPORT_HOURS})
        T.finish_wandb(summary)
        print(f"{tag}: beta {beta:.3f}, val KL {info['val_kl']:.2f} nats/h, kept epoch {ep} of {len(history)}")
        return info

    def first_k30():
        info = train_control("k30", BETA_FIRST, "c_k30_first")
        kl = info["val_kl"]
        if KL_BAND[0] <= kl <= KL_BAND[1]:
            S["beta"], S["models"]["c_k30"] = BETA_FIRST, S["models"]["c_k30_first"]
            S["per"]["c_k30"] = S["per"]["c_k30_first"]
            S["fit"]["c_k30@k30"] = S["fit"]["c_k30_first@k30"]
            S["stats1a"]["c_k30@k30"] = S["stats1a"]["c_k30_first@k30"]
            S["runs"]["c_k30"] = info
            S["adjust"] = False
        else:
            S["beta"] = float(np.clip(BETA_FIRST * kl / KL_TARGET, *BETA_CLIP))
            S["adjust"] = True
        R.save("beta_choice.json", dict(first_beta=BETA_FIRST, first_val_kl=kl, band=KL_BAND, adjusted=S["adjust"],
                                        final_beta=S["beta"]))
        return f"KL {kl:.2f}; final beta {S['beta']:.3f}{' (adjusted)' if S['adjust'] else ''}"

    R.step("train_k30_first", first_k30, needs=["extract"])

    def adjusted_k30():
        if not S.get("adjust"):
            return "not needed"
        train_control("k30", S["beta"], "c_k30")

    R.step("train_k30_adjusted", adjusted_k30, needs=["train_k30_first"])
    R.step("train_h15", lambda: train_control("h15", S["beta"], "c_h15") and None, needs=["train_k30_first"])
    R.step("train_real", lambda: train_control("real", S["beta"], "c_real") and None, needs=["train_k30_first"])

    def onestep():
        out = {}
        for tag in ("real_s0", "delta_real", "w_real", "c_real"):
            out[tag] = evaluate_mimic.evaluate(str(S["npz"]["real"]), str(S["models"][tag]))
        R.save("onestep_real.json", out)
        for tag, r in out.items():
            p = r["metrics"]["MAP"]["prior"]
            if p["mae"] is not None:
                print(f"{tag:10} MAP MAE {p['mae']:.3f} RMSE {p['rmse']:.3f} coverage {100 * p['coverage']:.1f}%")

    R.step("onestep", onestep, needs=["train_real"])

    # ---------------------------------------------------------------- final
    def final():
        def eff(a, b):
            return spread(S["per"][a], S["per"][b]) if a in S["per"] and b in S["per"] else None
        res = dict(horizons=H, true=true, predicted_bias=predicted_bias,
                   predicted_learned=[t + b for t, b in zip(true, predicted_bias)], noise_floor=S.get("noise"),
                   n_moments=len(S["moments"] or []), beta=S.get("beta"), adjusted=S.get("adjust"), runs=S["runs"],
                   learned={f"{p}_{k}": eff(f"{p}_{k}", "real_s0") for p in ("delta", "w", "c") for k in ("k30", "h15")},
                   learned_own_ref={f"{p}_k30": eff(f"{p}_k30", f"{p}_real") for p in ("delta", "w", "c")},
                   confounding_shift={p: eff(f"{p}_h15", f"{p}_k30") for p in ("delta", "w", "c")},
                   fit_numbers={p: fit_numbers(p) for p in ("delta", "w", "c")},
                   start_hour_stats=S["stats1a"])
        sc = res["confounding_shift"]["c"]
        fc = res["fit_numbers"]["c"]["decoder_at_prior_mean"].get("drop_captured_share")
        kls = [S["runs"][t]["val_kl"] for t in ("c_k30", "c_h15") if t in S["runs"]]
        on_target = bool(kls) and all(KL_BAND[0] <= k <= KL_BAND[1] for k in kls)
        s1 = sc["mean"][0] if sc and sc.get("n_paired") else None
        if not on_target:
            verdict = "inconclusive: control off-target"
        elif s1 is None:
            verdict = "not available"
        elif s1 > SPECIFIC_SHIFT and fc is not None and fc < FIT_TARGET:
            verdict = "confirmed: specific to fitting the start hours"
        elif s1 <= GLOBAL_SHIFT:
            verdict = "rejected: the global shift explains the bias"
        else:
            verdict = "partly: both contribute"
        res["verdict"] = dict(control_val_kl=kls, kl_band=KL_BAND, on_target=on_target, control_shift_hour1=s1,
                              control_drop_captured=fc, specific_if_above=SPECIFIC_SHIFT, global_if_at_or_below=GLOBAL_SHIFT,
                              verdict=verdict)
        R.save("stage13c_metrics.json", res)
        print("verdict:", res["verdict"])
        plot_fit(S["fit"], R.work / "figures" / "stage13c_start_fit.png")
        plot_effect(res, R.work / "figures" / "stage13c_confounding.png")

    R.step("final", final, needs=["ref"])
    print(f"\nall steps done in {R.elapsed_h():.2f} h")
    return R


LINES = (("delta", "tab:orange", "act_delta"), ("w", "tab:green", "13b: start weight 10"),
         ("c", "tab:purple", "13c control: lowered beta"))


def plot_fit(fit, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, key, title in ((axes[0], "k30", "K = 30, G = 0 (drug only)"),
                           (axes[1], "h15", "Stage 12b, G = 15 (hidden confounder)")):
        meas = next((v for k, v in fit.items() if k.endswith(f"@{key}")), None)
        if meas is None:
            continue
        ax.plot(meas["d"], meas["measured_change"], "k-o", lw=2, label="measured change")
        for p, color, label in LINES:
            r = fit.get(f"{p}_{key}@{key}")
            if r:
                ax.plot(r["d"], r["predicted_change"], "o--", color=color, label=f"{label}, predicted")
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
    for p, color, label in LINES:
        r = res["learned"].get(f"{p}_h15")
        if r and r.get("n_paired"):
            ax.fill_between(hours, [0] + r["p10"], [0] + r["p90"], color=color, alpha=0.2)
            ax.plot(hours, [0] + r["mean"], "o-", color=color, label=f"{label}, learned (10-90%)")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("hours after the decision hour")
    ax.set_ylabel("MAP, NEE 0.1 minus 0 (mmHg)")
    ax.set_title(f"12b data, G = {G_HIDDEN}: learned vs true vs observational (n = {res['n_moments']})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
