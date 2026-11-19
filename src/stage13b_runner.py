"""
stage13b_runner.py - one unattended Kaggle GPU run for Stage 13b (pre-registered in LEARNING.md).

Steps (each isolated; results saved as soon as a step finishes; summary.json after every step):
  extract     MIMIC extraction once: real, K = 30 (G = 0) and 12b G = 15 data (tmp only)
  step1a      sigma, predictive spread, KL and residual at start hour +1 / +2 vs other hours
  step1b      start-fit with two point predictions (decoder at prior mean; 50-sample mean)
  step1c      real start hours vs the Part B moments
  gate        pre-registered rule: train only if sigma widens (>= 1.2x) or residual <= -5 mmHg
              at start hour +1 for either 12b G = 15 model
  ref         Part B of the existing models (noise floor, Stage 13 comparison)
  train_*     start-weighted act_delta models on K = 30, 12b G = 15, real data (seed 0),
              each followed by Part B, start-fit and the 1a statistics
  onestep     one-step error on real data
  negctl      negative control (all vitals) for the real-data models
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
import negative_control as NC     # noqa: E402
import semisynthetic as SS        # noqa: E402
import stage13b as B              # noqa: E402
import train as T                 # noqa: E402
from overnight_stage11 import Runner, checkpoint_epoch, spread   # noqa: E402

REPORT_HOURS = (1, 3, 6)
G_HIDDEN = 15
DATA = {"real": None, "k30": ("ramp", 0), "h15": ("hidden", G_HIDDEN)}
SD_RATIO_GATE, RESID_GATE = 1.2, -5.0
FIT_TARGET, BIAS_TARGET = 0.5, -2.60


def run_all(icu_dir, weights, work, tmp, device="cpu", max_stays=0, max_epochs=None,
            b_vital="MAP", b_limit=imagine_mimic.MAP_LIMIT, total_budget_h=10.5, train_budget_h=1.5,
            force_gate=False):
    R = Runner(work, tmp, total_budget_h, train_budget_h)
    T.OUT, T.FIG, T.DEVICE = R.work, R.work / "figures", device
    if max_epochs:
        T.MAX_EPOCHS = max_epochs
    S = dict(npz={}, sets={}, per={}, partb={}, fit={}, stats1a={}, models=dict(weights), moments=None, clips=None)
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

    def part_b(tag, key):
        ds, stats = sets(key)
        b = imagine_mimic.part_b(model(tag), ds["test"], stats, device, vital=b_vital, limit=b_limit,
                                 keep_per_moment=True, moments=S["moments"])
        S["per"][tag] = b.pop("_per_moment", {})
        S["partb"][tag] = b
        return b

    def fit(tag, key):
        ds, stats = sets(key)
        S["fit"][f"{tag}@{key}"] = B.start_fit_both(model(tag), ds["test"], stats, device)
        return S["fit"][f"{tag}@{key}"]

    def stats1a(tag, key):
        ds, stats = sets(key)
        S["stats1a"][f"{tag}@{key}"] = B.start_hour_stats(model(tag), ds["test"], stats, device)
        return S["stats1a"][f"{tag}@{key}"]

    def print_1a(name, r):
        print(f"{name}: " + " | ".join(
            f"{g}: sd {v['predictive_sd_mmhg']:.2f} sigma {v['sigma_dec_mmhg']:.2f} KL {v['kl_nats']:.2f} "
            f"resid {v['residual_mmhg']:+.2f} (n {v['n_measured']})" for g, v in r.items()))

    def fit_numbers(prefix):
        """Drug part at start hour +1 (K = 30 minus real) and the captured 12b drop, both point types."""
        f = S["fit"]
        out = {}
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
        R.save("extract.json", dict(n_stays=int(real["x"].shape[0]), n_moments=len(S["moments"]), clipped=S["clips"]))
        return f"{len(S['moments'])} moments"

    R.step("extract", extract)

    # ---------------------------------------------------------------- step 1
    def step1a():
        for tag, key in (("cur_k30", "k30"), ("cur_h15", "h15"), ("delta_k30", "k30"), ("delta_h15", "h15")):
            print_1a(f"{tag} on {key}", stats1a(tag, key))
        R.save("step1a_start_hour_stats.json", S["stats1a"])

    def step1b():
        for tag, key in (("real_s0", "real"), ("cur_k30", "k30"), ("cur_h15", "h15"),
                         ("delta_real", "real"), ("delta_k30", "k30"), ("delta_h15", "h15")):
            fit(tag, key)
        S["models"].setdefault("cur_real", S["models"]["real_s0"])
        S["fit"]["cur_real@real"] = S["fit"]["real_s0@real"]
        res = dict(fits={k: v for k, v in S["fit"].items()}, current=fit_numbers("cur"), act_delta=fit_numbers("delta"))
        R.save("step1b_start_fit_two_points.json", res)
        for name in ("current", "act_delta"):
            for kind, o in res[name].items():
                print(f"{name} {kind}: drug part predicted {o.get('drug_part_predicted', float('nan')):+.2f} "
                      f"vs measured {o.get('drug_part_measured', float('nan')):+.2f}; 12b drop captured "
                      f"{100 * (o.get('drop_captured_share') or 0):.0f}%")

    def step1c():
        ds, stats = sets("real")
        r = B.populations(ds["test"], stats, S["moments"])
        R.save("step1c_populations.json", r)
        for k, v in r.items():
            print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})

    R.step("step1a", step1a, needs=["extract"])
    R.step("step1b", step1b, needs=["extract"])
    R.step("step1c", step1c, needs=["extract"])

    def gate():
        reasons = []
        for tag in ("cur_h15@h15", "delta_h15@h15"):
            r = S["stats1a"].get(tag)
            if not r:
                continue
            p1 = r["start_plus1"]
            if p1["sd_ratio_vs_other"] >= SD_RATIO_GATE:
                reasons.append(f"{tag}: predictive spread {p1['sd_ratio_vs_other']:.2f}x other hours")
            if p1["residual_mmhg"] <= RESID_GATE:
                reasons.append(f"{tag}: residual {p1['residual_mmhg']:+.2f} mmHg")
        if force_gate:                                   # local smoke tests only
            reasons.append("forced open for a smoke test")
        S["gate_open"] = bool(reasons)
        R.save("gate.json", dict(open=S["gate_open"], reasons=reasons, rule=dict(sd_ratio=SD_RATIO_GATE, residual=RESID_GATE)))
        print("gate:", "OPEN" if reasons else "CLOSED", reasons)

    R.step("gate", gate, needs=["step1a"])

    def gate_open():
        if not S.get("gate_open"):
            raise RuntimeError("pre-registered gate closed: no sigma widening and no clear underfit at starts")

    R.step("gate_open", gate_open, needs=["gate"])

    def ref():
        for tag, key in (("real_s0", "real"), ("real_s1", "real"), ("cur_k30", "k30"), ("cur_h15", "h15"),
                         ("delta_k30", "k30"), ("delta_h15", "h15"), ("delta_real", "real")):
            part_b(tag, key)
        S["noise"] = spread(S["per"]["real_s1"], S["per"]["real_s0"])
        R.save("noise_floor.json", S["noise"])

    R.step("ref", ref, needs=["extract", "gate_open"])

    # ---------------------------------------------------------------- step 2: start-weighted training
    def train_step(key):
        def fn():
            left = R.total_s - (time.time() - R.t_start) - R.reserve_s
            if left < 0.3 * 3600:
                raise RuntimeError(f"not enough time left ({left / 3600:.2f} h) for {key}")
            T.TIME_BUDGET_S = min(R.train_s, left)
            loaders, _ = mimic_data.make_loaders_mimic(S["npz"][key], batch_size=T.BATCH_SIZE)
            name = f"mimic_s13b_weighted_{key}"
            spec = DATA[key]
            _, history, _ = B.train_start_weighted(
                name, seed=0, loaders=loaders,
                model_kwargs=dict(n_vitals=len(mimic_data.VITALS), n_actions=len(mimic_data.ACTIONS),
                                  act_now=True, act_delta=True),
                wandb_config=dict(stage="13b", variant="act_delta + start-weighted reconstruction", data=key,
                                  semisynthetic=spec is not None,
                                  injected_k_mmhg_per_dose=0.0 if spec is None else C.K_MMHG,
                                  confounder_g_mmhg=0.0 if spec is None else float(spec[1]),
                                  confounder_kind=None if spec is None else spec[0],
                                  nee_clip_mcg_kg_min=mimic_data.NEE_CLIP,
                                  fluid_clip_l_per_hour=mimic_data.FLUID_CLIP_L, **(S["clips"] or {})))
            tag = f"w_{key}"
            S["models"][tag] = R.work / f"model_{name}.pt"
            ep = checkpoint_epoch(history)
            best = next(h for h in history if h["epoch"] == ep)
            R.save(f"train_{tag}.json", dict(
                data=key, checkpoint_epoch=ep, epochs=len(history),
                seconds_per_epoch_median=float(np.median([h["seconds"] for h in history])),
                val_loss_weighted=best["val"]["loss"], val_recon_weighted=best["val"]["recon"], val_kl=best["val"]["kl"],
                val_kl_dims=best["val"]["kl_dims"], stopped_early=len(history) - ep >= T.PATIENCE))
            b = part_b(tag, key)
            R.save(f"partb_{tag}.json", b)
            f = fit(tag, key)
            s1 = stats1a(tag, key)
            R.save(f"start_{tag}.json", dict(start_fit=f, start_hour_stats=s1))
            print_1a(f"{tag} on {key}", s1)
            summary = dict(checkpoint_epoch=ep, epochs=len(history))
            if b.get("mean_diff"):
                summary.update({f"effect/imagined_{h}h": b["mean_diff"][h - 1] for h in REPORT_HOURS})
            T.finish_wandb(summary)
            return f"checkpoint epoch {ep} of {len(history)}"
        return fn

    for key in ("k30", "h15", "real"):
        R.step(f"train_{key}", train_step(key), needs=["extract", "gate_open"])

    def onestep():
        out = {}
        for tag in ("real_s0", "delta_real", "w_real"):
            out[tag] = evaluate_mimic.evaluate(str(S["npz"]["real"]), str(S["models"][tag]))
            print(f"--- one-step, {tag}\n" + evaluate_mimic.table(out[tag]))
        R.save("onestep_real.json", out)

    R.step("onestep", onestep, needs=["train_real"])

    def negctl():
        ds, stats = sets("real")
        per = {tag: NC.per_moment_all_vitals(model(tag), ds["test"], stats, device, S["moments"])
               for tag in ("real_s0", "real_s1", "delta_real", "w_real")}
        noise = NC.summarise(per["real_s1"], per["real_s0"])
        res = dict(noise_floor=noise, effects={}, beyond_noise={})
        for tag in ("real_s0", "delta_real", "w_real"):
            res["effects"][tag] = NC.summarise(per[tag])
            res["beyond_noise"][tag] = NC.verdict(res["effects"][tag], noise)
        R.save("negative_control.json", res)
        for tag in ("real_s0", "delta_real", "w_real"):
            for v in ("MAP", "Temp"):
                e = res["effects"][tag][v]
                print(f"{tag:10} {v:4} " + " ".join(f"{e['mean'][h - 1]:+.3f}" for h in REPORT_HOURS)
                      + " strict " + str([res["beyond_noise"][tag][v]["strict"][h - 1] for h in REPORT_HOURS]))

    R.step("negctl", negctl, needs=["train_real"])

    # ---------------------------------------------------------------- final
    def final():
        def eff(a, b):
            return spread(S["per"][a], S["per"][b]) if a in S["per"] and b in S["per"] else None
        res = dict(horizons=H, true=true, predicted_bias=predicted_bias,
                   predicted_learned=[t + b for t, b in zip(true, predicted_bias)], noise_floor=S.get("noise"),
                   n_moments=len(S["moments"] or []), gate_open=S.get("gate_open"),
                   learned=dict(cur_k30=eff("cur_k30", "real_s0"), delta_k30=eff("delta_k30", "real_s0"),
                                w_k30=eff("w_k30", "real_s0"), w_k30_own_ref=eff("w_k30", "w_real"),
                                cur_h15=eff("cur_h15", "real_s0"), delta_h15=eff("delta_h15", "real_s0"),
                                w_h15=eff("w_h15", "real_s0"), w_real_vs_cur_real=eff("w_real", "real_s0")),
                   confounding_shift=dict(current=eff("cur_h15", "cur_k30"), act_delta=eff("delta_h15", "delta_k30"),
                                          weighted=eff("w_h15", "w_k30")),
                   fit_numbers=dict(current=fit_numbers("cur"), act_delta=fit_numbers("delta"), weighted=fit_numbers("w")))
        fw = res["fit_numbers"]["weighted"]["decoder_at_prior_mean"]
        lw = res["learned"]["w_h15"]
        crit_i = fw.get("drop_captured_share") is not None and fw["drop_captured_share"] >= FIT_TARGET
        crit_ii = bool(lw and lw.get("n_paired")) and lw["mean"][0] <= BIAS_TARGET
        res["verdict"] = dict(fit_captured_share=fw.get("drop_captured_share"), fit_target=FIT_TARGET, criterion_i=crit_i,
                              learned_hour1=lw["mean"][0] if lw and lw.get("n_paired") else None, bias_target=BIAS_TARGET,
                              criterion_ii=crit_ii,
                              verdict=("not run: gate closed" if not S.get("gate_open") else
                                       "confirmed" if crit_i and crit_ii else ("partly" if crit_i or crit_ii else "rejected")))
        R.save("stage13b_metrics.json", res)
        print("verdict:", res["verdict"])
        plot_fit(S["fit"], R.work / "figures" / "stage13b_start_fit.png")
        plot_effect(res, R.work / "figures" / "stage13b_confounding.png")

    R.step("final", final, needs=["step1b"])
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
        for tag, color, label in ((f"cur_{key}", "tab:blue", "current"), (f"delta_{key}", "tab:orange", "act_delta"),
                                  (f"w_{key}", "tab:green", "act_delta + start weight 10")):
            r = fit.get(f"{tag}@{key}")
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
    for tag, color, label in (("cur_h15", "tab:blue", "current"), ("delta_h15", "tab:orange", "act_delta"),
                              ("w_h15", "tab:green", "act_delta + start weight 10")):
        r = res["learned"].get(tag)
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
