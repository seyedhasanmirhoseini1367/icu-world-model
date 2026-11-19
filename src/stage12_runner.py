"""
stage12_runner.py - one unattended Kaggle GPU run for Stage 12 (hidden confounding).

Steps (each isolated; results saved as soon as a step finishes; summary.json after every step):
  extract    MIMIC extraction once -> real data + confounded copies G = 0, 5, 15 (K = 30), in tmp only;
             the Part B moments are chosen once, on the real test split, and reused everywhere
  u_stats    U0_h (expected hidden u without a dose, from the training split) -> predicted bias per G
  ref        Part B of the act_now real-data models (seed 0, seed 1) -> noise floor;
             and the Stage 11 K = 30 model on the G = 0 data (should match the new G = 0 run)
  train_g0 / train_g5 / train_g15   act_now model (the default), seed 0, same hyperparameters,
             then Part B on its own data at the fixed moments: learned, true, bias, predicted bias
  final      one table and one figure
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

import confounded as C            # noqa: E402
import imagine_mimic              # noqa: E402
import mimic_data                 # noqa: E402
import semisynthetic as SS        # noqa: E402
import train as T                 # noqa: E402
from overnight_stage11 import Runner, checkpoint_epoch, spread   # noqa: E402

GS = (0, 5, 15)
REPORT_HOURS = (1, 3, 6)


def run_all(icu_dir, weights, work, tmp, device="cpu", max_stays=0, max_epochs=None,
            b_vital="MAP", b_limit=imagine_mimic.MAP_LIMIT, total_budget_h=10.5, train_budget_h=1.5):
    R = Runner(work, tmp, total_budget_h, train_budget_h)
    T.OUT, T.FIG, T.DEVICE = R.work, R.work / "figures", device
    if max_epochs:
        T.MAX_EPOCHS = max_epochs
    S = dict(npz={}, sets={}, per={}, partb={}, models=dict(weights), clips=None, moments=None)
    dose = imagine_mimic.DOSE
    H = list(range(1, imagine_mimic.HORIZON_B + 1))

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

    # ---------------------------------------------------------------- extraction
    def extract():
        real = mimic_data.extract(icu_dir, max_stays)
        S["npz"]["real"] = R.tmp / "mimic_real.npz"
        np.savez_compressed(S["npz"]["real"], **real)
        for g in GS:
            S["npz"][g] = R.tmp / f"mimic_k30_g{g}.npz"
            np.savez_compressed(S["npz"][g], **C.inject_confounded(real, k=C.K_MMHG, g=float(g)))
        S["clips"] = mimic_data.clip_counts(S["npz"]["real"])
        ds, stats = sets("real")
        S["moments"] = imagine_mimic.find_moments(ds["test"], stats, b_vital, b_limit)
        nee = real["a"][..., C.NEE]
        u = C.hidden_u(nee)
        valid = real["valid"] > 0
        R.save("extract.json", dict(
            n_stays=int(real["x"].shape[0]), n_moments=len(S["moments"]), clipped=S["clips"],
            k=C.K_MMHG, gs=list(GS), ramp_h=C.RAMP_H, decay_tau_h=C.DECAY_TAU_H,
            n_vaso_starts=int(C.vaso_starts(nee).sum()),
            share_hours_u_positive=float((u[valid] > 0).mean()), mean_u_valid_hours=float(u[valid].mean())))
        return f"{len(S['moments'])} moments"

    R.step("extract", extract)

    # ---------------------------------------------------------------- expected u without a dose
    def u_stats():
        out = {}
        for split in ("train", "test"):
            ds, stats = sets("real")
            out[split] = C.no_dose_u(ds[split], stats, H)
        out["predicted_bias"] = {str(g): C.predicted_bias(g, out["train"]) for g in GS}
        S["u_stats"] = out
        R.save("u_stats.json", out)
        print("U0 (train) at 1/3/6 h:", [round(out["train"]["U0"][h - 1], 3) for h in REPORT_HOURS],
              "| exp(-h/4):", [round(out["train"]["U1_formula"][h - 1], 3) for h in REPORT_HOURS])

    R.step("u_stats", u_stats, needs=["extract"])

    # ---------------------------------------------------------------- reference models, noise floor
    def ref():
        for tag in ("real_s0", "real_s1"):
            part_b(tag, "real")
        part_b("stage11_k30", 0)
        nf = spread(S["per"]["real_s1"], S["per"]["real_s0"])
        S["noise"] = nf
        R.save("noise_floor.json", dict(seed_a=1, seed_b=0, **nf))
        R.save("ref_partb.json", {t: S["partb"][t] for t in ("real_s0", "real_s1", "stage11_k30")})
        print("noise floor at 1/3/6 h:", " ".join(
            f"{nf['mean'][h - 1]:+.2f} [{nf['p10'][h - 1]:+.2f}..{nf['p90'][h - 1]:+.2f}]" for h in REPORT_HOURS))

    R.step("ref", ref, needs=["extract"])

    # ---------------------------------------------------------------- one confounded run
    def result_for(tag, g):
        true = C.true_effect(dose, H)
        lr = spread(S["per"][tag], S["per"]["real_s0"])                 # learned = semi minus real model
        if not lr.get("n_paired"):
            return dict(g=g, n_paired=0)
        res = dict(g=g, n_paired=lr["n_paired"], true=true, learned_mean=lr["mean"],
                   learned_p10=lr["p10"], learned_p90=lr["p90"],
                   bias_mean=[m - t for m, t in zip(lr["mean"], true)],
                   bias_p10=[m - t for m, t in zip(lr["p10"], true)],
                   bias_p90=[m - t for m, t in zip(lr["p90"], true)])
        if "u_stats" in S:
            pb = C.predicted_bias(g, S["u_stats"]["train"])
            res.update(predicted_bias=pb, predicted_learned=[t + b for t, b in zip(true, pb)])
        if "noise" in S:
            nf = S["noise"]
            res["bias_outside_noise"] = [abs(b) > max(abs(lo), abs(hi))
                                         for b, lo, hi in zip(res["bias_mean"], nf["p10"], nf["p90"])]
        return res

    def train_step(g):
        def fn():
            left = R.total_s - (time.time() - R.t_start) - R.reserve_s
            if left < 0.3 * 3600:
                raise RuntimeError(f"not enough time left ({left / 3600:.2f} h) for G = {g}")
            T.TIME_BUDGET_S = min(R.train_s, left)
            loaders, _ = mimic_data.make_loaders_mimic(S["npz"][g], batch_size=T.BATCH_SIZE)
            name = f"mimic_s12_g{g}"
            _, history, _ = T.train(
                name, seed=0, loaders=loaders,
                model_kwargs=dict(n_vitals=len(mimic_data.VITALS), n_actions=len(mimic_data.ACTIONS)),
                wandb_config=dict(stage="12", variant="act_now (default)", semisynthetic=True,
                                  injected_k_mmhg_per_dose=C.K_MMHG, injected_tau_h=SS.TAU_H,
                                  confounder_g_mmhg=float(g), confounder_ramp_h=C.RAMP_H,
                                  confounder_tau_h=C.DECAY_TAU_H, nee_clip_mcg_kg_min=mimic_data.NEE_CLIP,
                                  fluid_clip_l_per_hour=mimic_data.FLUID_CLIP_L, **(S["clips"] or {})))
            tag = f"g{g}"
            S["models"][tag] = R.work / f"model_{name}.pt"
            ep = checkpoint_epoch(history)
            best = next(h for h in history if h["epoch"] == ep)
            info = dict(g=g, checkpoint_epoch=ep, epochs=len(history),
                        seconds_per_epoch_median=float(np.median([h["seconds"] for h in history])),
                        val_loss=best["val"]["loss"], val_recon=best["val"]["recon"], val_kl=best["val"]["kl"],
                        val_kl_dims=best["val"]["kl_dims"], stopped_early=len(history) - ep >= T.PATIENCE)
            R.save(f"train_g{g}.json", info)
            b = part_b(tag, g)
            R.save(f"partb_g{g}.json", b)
            res = result_for(tag, g)
            R.save(f"result_g{g}.json", res)
            summary = dict(checkpoint_epoch=ep, epochs=len(history))
            if res.get("n_paired"):
                summary.update({f"effect/learned_{h}h": res["learned_mean"][h - 1] for h in REPORT_HOURS})
                summary.update({f"effect/true_{h}h": res["true"][h - 1] for h in REPORT_HOURS})
                print(f"G={g}: true " + " ".join(f"{res['true'][h - 1]:+.2f}" for h in REPORT_HOURS) +
                      " | learned " + " ".join(f"{res['learned_mean'][h - 1]:+.2f}" for h in REPORT_HOURS) +
                      (" | predicted " + " ".join(f"{res['predicted_learned'][h - 1]:+.2f}" for h in REPORT_HOURS)
                       if "predicted_learned" in res else ""))
            T.finish_wandb(summary)
            return f"checkpoint epoch {ep} of {len(history)}"
        return fn

    for g in GS:
        R.step(f"train_g{g}", train_step(g), needs=["extract"])

    # ---------------------------------------------------------------- final table and figure
    def final():
        rows = {str(g): result_for(f"g{g}", g) for g in GS if f"g{g}" in S["per"]}
        rep = None
        if "stage11_k30" in S["per"] and "g0" in S["per"]:
            rep = spread(S["per"]["g0"], S["per"]["stage11_k30"])     # same data, same seed: how close?
        res = dict(horizons=H, report_hours=list(REPORT_HOURS), dose=dose, k=C.K_MMHG,
                   noise_floor=S.get("noise"), u_stats=S.get("u_stats"), by_g=rows,
                   g0_vs_stage11_k30=rep, n_moments=len(S["moments"] or []))
        R.save("stage12_metrics.json", res)
        plot(res, R.work / "figures" / "stage12_confounding.png")

    R.step("final", final, needs=["ref"])
    print(f"\nall steps done in {R.elapsed_h():.2f} h")
    return R


def plot(res, path):
    hours = [0] + res["horizons"]
    gs = [g for g in GS if str(g) in res["by_g"]]
    fig, axes = plt.subplots(1, max(len(gs), 1), figsize=(5 * max(len(gs), 1), 4.3), squeeze=False)
    nf = res.get("noise_floor") or {}
    for ax, g in zip(axes[0], gs):
        r = res["by_g"][str(g)]
        if not r.get("n_paired"):
            continue
        ax.plot(hours, [0] + r["true"], "k--", lw=2, label="true causal effect K*e")
        if nf.get("n_paired"):
            ax.fill_between(hours, [0] + nf["p10"], [0] + nf["p90"], color="0.6", alpha=0.3,
                            label="noise floor (seed 1 - seed 0), 10-90%")
        ax.fill_between(hours, [0] + r["learned_p10"], [0] + r["learned_p90"], color="tab:orange", alpha=0.25)
        ax.plot(hours, [0] + r["learned_mean"], "o-", color="tab:orange", label="learned, mean (10-90%)")
        if "predicted_learned" in r:
            ax.plot(hours, [0] + r["predicted_learned"], ":", color="tab:red", lw=2,
                    label="predicted if learning E[MAP' | history, a]")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(f"G = {g} mmHg (K = {res['k']:.0f})")
        ax.set_xlabel("hours after the decision hour")
        ax.set_ylabel("MAP, NEE 0.1 minus 0 (mmHg)")
        ax.legend(fontsize=7)
    fig.suptitle(f"Hidden confounding: true vs learned vasopressor effect (n = {res['n_moments']} moments)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
