"""
stage12b_runner.py - one unattended Kaggle GPU run: the checks that explain Stage 12, and
Stage 12b (a truly hidden confounder, u starts the hour AFTER the vasopressor start).

Steps (each isolated; results saved as soon as a step finishes; summary.json after every step):
  extract     MIMIC extraction once; Stage 12 data (ramp, G = 0, 5, 15) and Stage 12b data
              (hidden, G = 5, 15), all K = 30, in tmp only; Part B moments chosen once on real data
  map_gap     read-only count: no MAP but both cuff systolic and diastolic in the same hour
  event       check 1: mean (MAP' - MAP) around real vasopressor starts, d = -6 .. +6 h
  ref         Part B of the real-data models (noise floor) and of the three Stage 12 models
  checks12    check 2 (raw observed MAP' after started vs not) and check 3 (learned effect per
              group: confounder visible in the history or not) for Stage 12
  u_stats     expected u in the no-dose arm for 12b (0 by construction) -> predicted bias
  train_h5 / train_h15   Stage 12b models (act_now default, seed 0, same hyperparameters),
              then Part B, the same table as Stage 12, and the same checks
  final       one table and the figures
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

import confounded as C            # noqa: E402
import imagine_mimic              # noqa: E402
import mimic_data                 # noqa: E402
import semisynthetic as SS        # noqa: E402
import stage12_checks as X        # noqa: E402
import stage12_runner as R12      # noqa: E402
import train as T                 # noqa: E402
from overnight_stage11 import Runner, checkpoint_epoch, spread   # noqa: E402

GS_12 = (0, 5, 15)
GS_12B = (5, 15)
REPORT_HOURS = (1, 3, 6)


def run_all(icu_dir, weights, work, tmp, device="cpu", max_stays=0, max_epochs=None,
            b_vital="MAP", b_limit=imagine_mimic.MAP_LIMIT, total_budget_h=10.5, train_budget_h=1.5):
    R = Runner(work, tmp, total_budget_h, train_budget_h)
    T.OUT, T.FIG, T.DEVICE = R.work, R.work / "figures", device
    if max_epochs:
        T.MAX_EPOCHS = max_epochs
    S = dict(npz={}, sets={}, per={}, partb={}, models=dict(weights), clips=None, moments=None, real=None)
    dose = imagine_mimic.DOSE
    H = list(range(1, imagine_mimic.HORIZON_B + 1))
    true = C.true_effect(dose, H)
    data_keys = [("ramp", g) for g in GS_12] + [("hidden", g) for g in GS_12B]

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

    def result_for(tag, g, u_stats):
        lr = spread(S["per"][tag], S["per"]["real_s0"])
        if not lr.get("n_paired"):
            return dict(g=g, n_paired=0)
        res = dict(g=g, n_paired=lr["n_paired"], true=true, learned_mean=lr["mean"],
                   learned_p10=lr["p10"], learned_p90=lr["p90"],
                   bias_mean=[m - t for m, t in zip(lr["mean"], true)],
                   bias_p10=[m - t for m, t in zip(lr["p10"], true)],
                   bias_p90=[m - t for m, t in zip(lr["p90"], true)])
        if u_stats:
            pb = C.predicted_bias(g, u_stats)
            res.update(predicted_bias=pb, predicted_learned=[t + b for t, b in zip(true, pb)])
        return res

    def checks_for(tag, key):
        """Check 2 and check 3 for one trained model on its own data."""
        ds, stats = sets(key)
        kind = key[0] if isinstance(key, tuple) else "ramp"
        raw = X.raw_observed(ds["test"], stats, S["moments"], kind)
        grp = X.groups(ds["test"], stats, S["moments"])
        eff = X.group_effects(S["per"][tag], S["per"]["real_s0"], grp, true)
        return dict(raw_observed=raw, groups=X.group_overlap(grp), group_effects=eff)

    # ---------------------------------------------------------------- extraction
    def extract():
        real = mimic_data.extract(icu_dir, max_stays)
        S["real"] = real
        S["npz"]["real"] = R.tmp / "mimic_real.npz"
        np.savez_compressed(S["npz"]["real"], **real)
        for kind, g in data_keys:
            S["npz"][(kind, g)] = R.tmp / f"mimic_k30_{kind}_g{g}.npz"
            np.savez_compressed(S["npz"][(kind, g)], **C.inject_confounded(real, k=C.K_MMHG, g=float(g), kind=kind))
        S["clips"] = mimic_data.clip_counts(S["npz"]["real"])
        ds, stats = sets("real")
        S["moments"] = imagine_mimic.find_moments(ds["test"], stats, b_vital, b_limit)
        R.save("extract.json", dict(n_stays=int(real["x"].shape[0]), n_moments=len(S["moments"]),
                                    clipped=S["clips"], data=[f"{k}_g{g}" for k, g in data_keys]))
        return f"{len(S['moments'])} moments"

    R.step("extract", extract)

    def map_gap():
        r = X.map_gap(icu_dir)
        R.save("map_gap.json", r)
        print(f"MAP missing but 220179 and 220180 present: {r['gap_hours_any']:,} hours "
              f"({100 * r['gap_share_any']:.2f}% of {r['valid_hours']:,} valid hours); "
              f"counting only plausible MAP: {r['gap_hours_plausible']:,} ({100 * r['gap_share_plausible']:.2f}%)")

    R.step("map_gap", map_gap)

    def event():
        out = {f"{kind}_g{g}": X.event_study(S["real"], g, kind) for kind, g in data_keys}
        R.save("check1_event_study.json", out)
        plot_event(out, R.work / "figures" / "stage12_event_study.png")
        for name, r in out.items():
            print(name, "mean (MAP' - MAP) at d = -3, 0, +1, +3:",
                  [None if r["mean_diff"][X.D_RANGE.index(d)] is None else round(r["mean_diff"][X.D_RANGE.index(d)], 2)
                   for d in (-3, 0, 1, 3)])

    R.step("event", event, needs=["extract"])

    # ---------------------------------------------------------------- references
    def ref():
        for tag in ("real_s0", "real_s1"):
            part_b(tag, "real")
        for g in GS_12:
            part_b(f"s12_g{g}", ("ramp", g))
        S["noise"] = spread(S["per"]["real_s1"], S["per"]["real_s0"])
        R.save("noise_floor.json", S["noise"])
        R.save("ref_partb.json", {t: S["partb"][t] for t in ("real_s0", "real_s1", "s12_g0", "s12_g5", "s12_g15")})

    R.step("ref", ref, needs=["extract"])

    def checks12():
        ds, stats = sets("real")
        u12 = C.no_dose_u(ds["train"], stats, H, "ramp")
        out = dict(raw_observed_real=X.raw_observed(ds["test"], stats, S["moments"], "ramp"))
        for g in GS_12:
            out[f"g{g}"] = checks_for(f"s12_g{g}", ("ramp", g))
            out[f"g{g}"]["result"] = result_for(f"s12_g{g}", g, u12)
        S["checks12"] = out
        R.save("checks_stage12.json", out)
        for g in GS_12:
            e = out[f"g{g}"]["group_effects"]
            print(f"Stage 12 G={g} learned at 1/3/6 h by group: " + " | ".join(
                f"{k} (n={v['n']}): " + " ".join(f"{v['learned_mean'][h - 1]:+.2f}" for h in REPORT_HOURS)
                for k, v in e.items() if v.get("n")))

    R.step("checks12", checks12, needs=["ref"])

    def u_stats():
        ds, stats = sets("real")
        S["u12b"] = C.no_dose_u(ds["train"], stats, H, "hidden")
        R.save("u_stats_12b.json", dict(u=S["u12b"], predicted_bias={str(g): C.predicted_bias(g, S["u12b"]) for g in GS_12B}))

    R.step("u_stats", u_stats, needs=["extract"])

    # ---------------------------------------------------------------- Stage 12b trainings
    def train_step(g):
        def fn():
            left = R.total_s - (time.time() - R.t_start) - R.reserve_s
            if left < 0.3 * 3600:
                raise RuntimeError(f"not enough time left ({left / 3600:.2f} h) for 12b G = {g}")
            T.TIME_BUDGET_S = min(R.train_s, left)
            key = ("hidden", g)
            loaders, _ = mimic_data.make_loaders_mimic(S["npz"][key], batch_size=T.BATCH_SIZE)
            name = f"mimic_s12b_g{g}"
            _, history, _ = T.train(
                name, seed=0, loaders=loaders,
                model_kwargs=dict(n_vitals=len(mimic_data.VITALS), n_actions=len(mimic_data.ACTIONS)),
                wandb_config=dict(stage="12b", variant="act_now (default)", semisynthetic=True,
                                  injected_k_mmhg_per_dose=C.K_MMHG, injected_tau_h=SS.TAU_H,
                                  confounder_g_mmhg=float(g), confounder_kind="hidden (starts hour after)",
                                  confounder_tau_h=C.DECAY_TAU_H, nee_clip_mcg_kg_min=mimic_data.NEE_CLIP,
                                  fluid_clip_l_per_hour=mimic_data.FLUID_CLIP_L, **(S["clips"] or {})))
            tag = f"s12b_g{g}"
            S["models"][tag] = R.work / f"model_{name}.pt"
            ep = checkpoint_epoch(history)
            best = next(h for h in history if h["epoch"] == ep)
            R.save(f"train_12b_g{g}.json", dict(
                g=g, checkpoint_epoch=ep, epochs=len(history),
                seconds_per_epoch_median=float(np.median([h["seconds"] for h in history])),
                val_loss=best["val"]["loss"], val_recon=best["val"]["recon"], val_kl=best["val"]["kl"],
                val_kl_dims=best["val"]["kl_dims"], stopped_early=len(history) - ep >= T.PATIENCE))
            b = part_b(tag, key)
            R.save(f"partb_12b_g{g}.json", b)
            res = result_for(tag, g, S.get("u12b"))
            chk = checks_for(tag, key)
            chk["result"] = res
            R.save(f"result_12b_g{g}.json", chk)
            summary = dict(checkpoint_epoch=ep, epochs=len(history))
            if res.get("n_paired"):
                summary.update({f"effect/learned_{h}h": res["learned_mean"][h - 1] for h in REPORT_HOURS})
                print(f"12b G={g}: true " + " ".join(f"{res['true'][h - 1]:+.2f}" for h in REPORT_HOURS) +
                      " | learned " + " ".join(f"{res['learned_mean'][h - 1]:+.2f}" for h in REPORT_HOURS) +
                      (" | predicted " + " ".join(f"{res['predicted_learned'][h - 1]:+.2f}" for h in REPORT_HOURS)
                       if "predicted_learned" in res else ""))
            T.finish_wandb(summary)
            return f"checkpoint epoch {ep} of {len(history)}"
        return fn

    for g in GS_12B:
        R.step(f"train_h{g}", train_step(g), needs=["extract", "ref"])

    # ---------------------------------------------------------------- final
    def final():
        by_g = {}
        if "s12_g0" in S["per"]:
            by_g["0"] = result_for("s12_g0", 0, None)          # same data as G = 0 in Stage 12
        for g in GS_12B:
            if f"s12b_g{g}" in S["per"]:
                by_g[str(g)] = result_for(f"s12b_g{g}", g, S.get("u12b"))
        res = dict(horizons=H, report_hours=list(REPORT_HOURS), dose=dose, k=C.K_MMHG,
                   noise_floor=S.get("noise"), by_g=by_g, n_moments=len(S["moments"] or []),
                   stage12=({f"g{g}": S["checks12"][f"g{g}"]["result"] for g in GS_12} if "checks12" in S else None))
        R.save("stage12b_metrics.json", res)
        R12.plot(res, R.work / "figures" / "stage12b_confounding.png")

    R.step("final", final, needs=["ref"])
    print(f"\nall steps done in {R.elapsed_h():.2f} h")
    return R


def plot_event(out, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, kind, title in ((axes[0], "ramp", "Stage 12: ramp before the start"),
                            (axes[1], "hidden", "Stage 12b: u starts the hour after")):
        for name, r in out.items():
            if r["kind"] != kind:
                continue
            line, = ax.plot(r["d"], [np.nan if v is None else v for v in r["mean_diff"]], "o-",
                            label=f"G = {r['g']}: measured mean")
            ax.plot(r["d"], r["spec"], "--", color=line.get_color(), alpha=0.7, label=f"G = {r['g']}: specified")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(0, color="0.5", lw=0.8, ls=":")
        ax.set_title(title)
        ax.set_xlabel("hours from the vasopressor start (0 = start hour)")
        ax.set_ylabel("mean MAP' - MAP (mmHg)")
        ax.legend(fontsize=7)
    fig.suptitle("Check 1: what was injected around real vasopressor starts (K = 30)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
