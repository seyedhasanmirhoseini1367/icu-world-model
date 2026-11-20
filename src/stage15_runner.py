"""
stage15_runner.py - one unattended Kaggle GPU run for Stage 15 (pre-registered in LEARNING.md).

A beta sweep with three seeds on the Stage 12b data (K = 30 drug effect, hidden confounder G = 15),
default act_now model, everything else as in Stage 12b:
    beta in {1.0, 0.97, 0.95, 0.92, 0.9} x seed in {0, 1, 2}
beta 1.0 / seed 0 is the existing Stage 12b G = 15 model (same setup; not retrained). The other 14
are trained here; beta is set on train.BETA_KL for the duration of one training only (as Stage 13c).

Per run: validation KL per predicted hour and share of hours below free bits (kl_profile, as Stage
14b); share of the 12b post-start drop predicted (start-fit with the measured K = 30 change as the
reference, pre-registered); learned effect (Part B minus the act_now real-data seed-0 model) vs true
and observational; one-step MAP MAE on the 12b test split. Earlier 13 / 13b / 13c G = 15 models are
re-evaluated with the same definitions for the scatter plot (not part of the verdict).

Steps (each isolated; every run's result saved as soon as it finishes; summary.json after every step):
  extract, ref, run_b<beta>_s<seed> x 14 (H1-critical runs first), ref_earlier, final
New file; shared code is only imported.
"""
import csv
import json
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
import train_jepa_b as TJB        # noqa: E402
from overnight_stage11 import Runner, checkpoint_epoch, spread   # noqa: E402

G_HIDDEN = 15
BETAS = (1.0, 0.97, 0.95, 0.92, 0.9)
SEEDS = (0, 1, 2)
REUSED = (1.0, 0)                 # the Stage 12b G = 15 model
# H1-critical runs first (beta 0.9 and 1.0 for every seed), then the intermediate betas
ORDER = ((0.9, 0), (1.0, 1), (0.9, 1), (1.0, 2), (0.9, 2),
         (0.95, 0), (0.95, 1), (0.95, 2), (0.97, 0), (0.97, 1), (0.97, 2), (0.92, 0), (0.92, 1), (0.92, 2))
KL_RATIO, RHO_MAX, MIN_RUNS_H2 = 2.0, -0.6, 10
REPORT_HOURS = (1, 3, 6)
EARLIER = (("s13_act_delta", "13: act_delta"), ("s13b_weighted", "13b: start weight 10"),
           ("s13c_beta0p9", "13c: beta 0.9 (act_delta)"))


def run_tag(beta, seed):
    return f"b{beta:.2f}_s{seed}".replace(".", "p")


def run_all(icu_dir, weights, work, tmp, device="cpu", max_stays=0, max_epochs=None,
            b_vital="MAP", b_limit=imagine_mimic.MAP_LIMIT, total_budget_h=11.0, train_budget_h=1.0, order=ORDER):
    """weights: an_real (act_now real data, seed 0), reused (Stage 12b G = 15), and the earlier
    G = 15 models s13_act_delta, s13b_weighted, s13c_beta0p9."""
    R = Runner(work, tmp, total_budget_h, train_budget_h)
    T.OUT, T.FIG, T.DEVICE = R.work, R.work / "figures", device
    if max_epochs:
        T.MAX_EPOCHS = max_epochs
    S = dict(npz={}, sets={}, loaders={}, per={}, moments=None, clips=None, runs={}, earlier={}, meas=None)
    dose = imagine_mimic.DOSE
    H = list(range(1, imagine_mimic.HORIZON_B + 1))
    true = C.true_effect(dose, H)
    bias = [-G_HIDDEN * u for u in C.dose_arm_u(H, "hidden")]
    observational = [t + b for t, b in zip(true, bias)]
    i1 = B.D_FIT.index(1)
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
        return b

    # ---------------------------------------------------------------- extraction
    def extract():
        real = mimic_data.extract(icu_dir, max_stays)
        arrays = dict(real=real, k30=C.inject_confounded(real, k=C.K_MMHG, g=0.0, kind="ramp"),
                      h15=C.inject_confounded(real, k=C.K_MMHG, g=float(G_HIDDEN), kind="hidden"))
        for key, arr in arrays.items():
            S["npz"][key] = R.tmp / f"mimic_{key}.npz"
            np.savez_compressed(S["npz"][key], **arr)
        del arrays
        S["clips"] = mimic_data.clip_counts(S["npz"]["real"])
        ds, stats = sets("real")
        S["moments"] = imagine_mimic.find_moments(ds["test"], stats, b_vital, b_limit)
        R.save("extract.json", dict(n_stays=int(real["x"].shape[0]), n_moments=len(S["moments"]), clipped=S["clips"]))
        return f"{len(S['moments'])} moments"

    R.step("extract", extract)

    # ---------------------------------------------------------------- one evaluation, used for every model
    def evaluate(tag, model, model_path, extra):
        """All per-run metrics for one G = 15 model; saved at once as run_<tag>.json."""
        ds, stats = sets("h15")
        part_b(tag, model, "h15")
        learned = spread(S["per"][tag], S["per"]["an_real"])
        fit = B.start_fit_both(model, ds["test"], stats, device)
        meas = S["meas"]
        pred1 = fit["predicted_change"][i1]
        share = (pred1 - meas["k30"]) / (meas["h15"] - meas["k30"]) if meas["h15"] != meas["k30"] else None
        kl = TJB.kl_profile(model, loaders("h15")["val"], device)
        one = evaluate_mimic.evaluate(str(S["npz"]["h15"]), str(model_path))
        rec = dict(tag=tag, **extra, val_kl=kl.get("mean"), share_below_free_bits=kl.get("share_below_free_bits"),
                   kl_per_dim=kl.get("per_dim"), kl_by_hour=kl.get("by_hour"),
                   start_fit=dict(predicted_change_hour1=pred1,
                                  predicted_change_hour1_samples=fit["predicted_change_samples"][i1],
                                  measured_change_hour1_h15=meas["h15"], measured_change_hour1_k30=meas["k30"],
                                  share_of_drop_predicted=share, full=fit),
                   learned=learned, true=true, observational=observational,
                   onestep_map=one["metrics"]["MAP"]["prior"], onestep_map_posterior=one["metrics"]["MAP"]["posterior"])
        return rec

    def ref():
        an = SS.load_model(weights["an_real"], device)
        part_b("an_real", an, "real")
        ds_k, st_k = sets("k30")
        ds_h, st_h = sets("h15")
        S["meas"] = dict(k30=B.start_fit_both(an, ds_k["test"], st_k, device)["measured_change"][i1],
                         h15=B.start_fit_both(an, ds_h["test"], st_h, device)["measured_change"][i1])
        R.save("ref.json", dict(measured_change_hour1=S["meas"], true=true, observational=observational))
        beta, seed = REUSED
        tag = run_tag(beta, seed)
        rec = evaluate(tag, SS.load_model(weights["reused"], device), weights["reused"],
                       dict(beta=beta, seed=seed, reused="Stage 12b G = 15 model (model_mimic_s12b_g15.pt), not retrained"))
        S["runs"][tag] = rec
        R.save(f"run_{tag}.json", rec)
        write_table(R, S)
        return f"measured hour-1 change: K = 30 {S['meas']['k30']:+.2f}, 12b {S['meas']['h15']:+.2f} mmHg"

    R.step("ref", ref, needs=["extract"])

    # ---------------------------------------------------------------- the 14 trainings
    def train_one(beta, seed):
        tag = run_tag(beta, seed)
        left = R.total_s - (time.time() - R.t_start) - R.reserve_s
        if left < 0.3 * 3600:
            raise RuntimeError(f"not enough time left ({left / 3600:.2f} h) for {tag}")
        T.TIME_BUDGET_S = min(R.train_s, left)
        name = f"mimic_s15_beta{beta:.2f}".replace(".", "p")
        saved_beta = T.BETA_KL
        T.BETA_KL = beta
        try:
            _, history, _ = T.train(
                name, seed=seed, loaders=loaders("h15"), model_kwargs=kw,
                wandb_config=dict(stage="15", variant="act_now (default), beta sweep", data="h15", beta_sweep=beta,
                                  semisynthetic=True, injected_k_mmhg_per_dose=C.K_MMHG,
                                  confounder_g_mmhg=float(G_HIDDEN), confounder_kind="hidden",
                                  nee_clip_mcg_kg_min=mimic_data.NEE_CLIP, fluid_clip_l_per_hour=mimic_data.FLUID_CLIP_L,
                                  **(S["clips"] or {})))
        finally:
            T.BETA_KL = saved_beta
        run_name = name if seed == T.SEED else f"{name}_s{seed}"                # train.py's file naming
        ep = checkpoint_epoch(history)
        best = next(h for h in history if h["epoch"] == ep)
        path = R.work / f"model_{run_name}.pt"
        model = SS.load_model(path, device)
        rec = evaluate(tag, model, path, dict(
            beta=beta, seed=seed, reused=None, model_file=path.name, checkpoint_epoch=ep, epochs=len(history),
            seconds_per_epoch_median=float(np.median([h["seconds"] for h in history])),
            history_val_loss=best["val"]["loss"], history_val_recon=best["val"]["recon"],
            history_val_kl=best["val"]["kl"], stopped_early=len(history) - ep >= T.PATIENCE))
        S["runs"][tag] = rec
        R.save(f"run_{tag}.json", rec)
        write_table(R, S)
        T.finish_wandb({"kl_mean": rec["val_kl"], "share_below_free_bits": rec["share_below_free_bits"],
                        "share_of_drop_predicted": rec["start_fit"]["share_of_drop_predicted"],
                        **{f"effect/learned_{h}h": rec["learned"]["mean"][h - 1] for h in REPORT_HOURS},
                        "onestep_map_mae": rec["onestep_map"]["mae"]})
        f = lambda v, fmt: "n/a" if v is None else format(v, fmt)        # noqa: E731
        print(f"{tag}: KL {f(rec['val_kl'], '.2f')} nats/h ({f(100 * rec['share_below_free_bits'], '.0f')}% of hours "
              f"below free bits), drop share {f(rec['start_fit']['share_of_drop_predicted'], '.2f')}, learned 1h "
              f"{f(rec['learned']['mean'][0], '+.2f')}, MAP MAE {f(rec['onestep_map']['mae'], '.3f')}; "
              f"kept epoch {ep} of {len(history)}", flush=True)

    for beta, seed in order:
        R.step(f"run_{run_tag(beta, seed)}", (lambda b=beta, s=seed: train_one(b, s)), needs=["ref"])

    # ---------------------------------------------------------------- earlier designs, same definitions
    def ref_earlier():
        for key, label in EARLIER:
            if key not in weights:
                continue
            rec = evaluate(key, SS.load_model(weights[key], device), weights[key], dict(label=label))
            S["earlier"][key] = rec
        R.save("earlier.json", S["earlier"])

    R.step("ref_earlier", ref_earlier, needs=["ref"])

    # ---------------------------------------------------------------- final
    def final():
        res = dict(betas=list(BETAS), seeds=list(SEEDS), horizons=H, true=true, observational=observational,
                   n_moments=len(S["moments"] or []), measured_change_hour1=S["meas"],
                   runs={t: slim(r) for t, r in S["runs"].items()}, earlier={t: slim(r) for t, r in S["earlier"].items()})
        res["verdict"] = verdict(res["runs"])
        R.save("stage15_metrics.json", res)
        print("verdict:", json.dumps(res["verdict"]))
        fig = R.work / "figures"
        for fn, name in ((plot_kl, "kl_vs_beta"), (plot_scatter, "share_vs_effect"), (plot_effect_beta, "effect_vs_beta")):
            try:
                fn(res, fig / f"stage15_{name}.png")
            except Exception as e:                       # a figure must not lose the metrics
                print(f"figure {name} failed: {type(e).__name__}: {e}")

    R.step("final", final, needs=["ref"])
    print(f"\nall steps done in {R.elapsed_h():.2f} h")
    return R


def slim(r):
    """Per-run record without the bulky per-hour lists."""
    out = {k: v for k, v in r.items() if k not in ("kl_by_hour", "start_fit")}
    sf = dict(r["start_fit"])
    sf.pop("full", None)
    out["start_fit"] = sf
    return out


TABLE_COLS = ("tag", "beta", "seed", "reused", "val_kl", "share_below_free_bits", "share_of_drop_predicted",
              "learned_1h", "learned_3h", "learned_6h", "onestep_map_mae", "checkpoint_epoch", "epochs")


def write_table(R, S):
    rows = []
    for t, r in sorted(S["runs"].items(), key=lambda kv: (-kv[1]["beta"], kv[1]["seed"])):
        m = r["learned"].get("mean") or [None] * 6
        rows.append(dict(tag=t, beta=r["beta"], seed=r["seed"], reused=bool(r.get("reused")), val_kl=r["val_kl"],
                         share_below_free_bits=r["share_below_free_bits"],
                         share_of_drop_predicted=r["start_fit"]["share_of_drop_predicted"],
                         learned_1h=m[0], learned_3h=m[2], learned_6h=m[5], onestep_map_mae=r["onestep_map"]["mae"],
                         checkpoint_epoch=r.get("checkpoint_epoch"), epochs=r.get("epochs")))
    with open(R.work / "stage15_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_COLS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- verdict

def _ranks(v):
    v = np.asarray(v, dtype=float)
    order = v.argsort()
    ranks = np.empty(len(v))
    ranks[order] = np.arange(len(v))
    for x in np.unique(v):                               # average ranks for ties
        idx = v == x
        ranks[idx] = ranks[idx].mean()
    return ranks


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def verdict(runs):
    kl = {(r["beta"], r["seed"]): r["val_kl"] for r in runs.values() if r.get("val_kl") is not None}
    out = dict(rules=dict(kl_ratio=KL_RATIO, rho_max=RHO_MAX, min_runs_h2=MIN_RUNS_H2), n_runs=len(runs))
    # H1
    by_beta = {b: [kl[(b, s)] for s in SEEDS if (b, s) in kl] for b in BETAS}
    means = {b: float(np.mean(v)) for b, v in by_beta.items() if v}
    spreads = {b: float(max(v) - min(v)) for b, v in by_beta.items() if len(v) >= 2}
    present = [b for b in BETAS if b in means]
    steps = {f"{a:g}->{b:g}": means[b] - means[a] for a, b in zip(present, present[1:])}
    jump = max((abs(v) for v in steps.values()), default=None)
    per_seed_step = {}
    for s in SEEDS:
        bs = [b for b in BETAS if (b, s) in kl]
        st = {f"{a:g}->{b:g}": kl[(b, s)] - kl[(a, s)] for a, b in zip(bs, bs[1:])}
        per_seed_step[s] = max(st, key=lambda k: abs(st[k])) if st else None
    out["h1_numbers"] = dict(kl=by_beta, mean_kl=means, seed_spread=spreads, neighbour_steps=steps, jump=jump,
                             largest_step_per_seed=per_seed_step,
                             ratio_0p9_over_1p0={s: (kl[(0.9, s)] / kl[(1.0, s)] if (0.9, s) in kl and (1.0, s) in kl
                                                     else None) for s in SEEDS})
    if not all((b, s) in kl for b in (1.0, 0.9) for s in SEEDS):
        h1 = "inconclusive: a beta 1.0 or 0.9 run is missing"
    else:
        a = all(kl[(0.9, s)] >= KL_RATIO * kl[(1.0, s)] for s in SEEDS)
        b = jump is not None and all(sp < jump for sp in spreads.values())
        out["h1a_ratio_every_seed"], out["h1b_spread_below_jump"] = a, b
        h1 = "confirmed" if a and b else ("partly" if a else "rejected")
    # H2
    pts = [(r["start_fit"]["share_of_drop_predicted"], r["learned"]["mean"][0]) for r in runs.values()
           if r["start_fit"].get("share_of_drop_predicted") is not None and r["learned"].get("mean")]
    rho = spearman([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 3 else None
    out["h2_numbers"] = dict(n=len(pts), spearman=rho)
    if len(pts) < MIN_RUNS_H2 or rho is None:
        h2 = "inconclusive: fewer than 10 runs"
    else:
        h2 = "confirmed" if rho <= RHO_MAX else "rejected"
    out.update(h1=h1, h2=h2)
    if h1.startswith("inconclusive") or h2.startswith("inconclusive"):
        out["verdict"] = "inconclusive"
    elif h1 == "confirmed" and h2 == "confirmed":
        out["verdict"] = "confirmed"
    elif h1 == "rejected" and h2 == "rejected":
        out["verdict"] = "rejected"
    else:
        out["verdict"] = "partly"
    return out


# ---------------------------------------------------------------- figures

SEED_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}


def plot_kl(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for s in res["seeds"]:
        pts = sorted(((r["beta"], r["val_kl"], r["share_below_free_bits"]) for r in res["runs"].values()
                      if r["seed"] == s and r.get("val_kl") is not None), reverse=True)
        if pts:
            axes[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=SEED_COLORS[s], label=f"seed {s}")
            axes[1].plot([p[0] for p in pts], [100 * p[2] for p in pts], "o-", color=SEED_COLORS[s], label=f"seed {s}")
    axes[0].axhline(T.FREE_BITS, color="0.5", ls=":", label="free-bits floor (1 nat)")
    axes[0].set_ylabel("validation KL, nats per predicted hour")
    axes[0].set_title("KL vs beta (12b data, G = 15)")
    axes[1].set_ylabel("% of validation hours with KL below free bits")
    axes[1].set_title("Hours below the free-bits floor")
    for ax in axes:
        ax.set_xlabel("beta")
        ax.invert_xaxis()
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_scatter(res, path):
    fig, ax = plt.subplots(figsize=(9, 5.8))
    cmap = plt.get_cmap("viridis")
    lo, hi = min(res["betas"]), max(res["betas"])
    markers = {0: "o", 1: "s", 2: "^"}
    for r in res["runs"].values():
        x, y = r["start_fit"].get("share_of_drop_predicted"), (r["learned"].get("mean") or [None])[0]
        if x is None or y is None:
            continue
        ax.scatter(100 * x, y, color=cmap((r["beta"] - lo) / (hi - lo or 1)), marker=markers.get(r["seed"], "o"),
                   s=60, edgecolor="k", lw=0.5, zorder=3)
    for k, r in res["earlier"].items():
        x, y = r["start_fit"].get("share_of_drop_predicted"), (r["learned"].get("mean") or [None])[0]
        if x is not None and y is not None:
            ax.scatter(100 * x, y, marker="X", s=90, color="tab:red", edgecolor="k", lw=0.5, zorder=4)
            ax.annotate(r.get("label", k), (100 * x, y), fontsize=7, xytext=(-8, 6), textcoords="offset points", ha="right")
    ax.axhline(res["true"][0], color="k", ls="--", lw=1.5, label=f"true hour-1 effect ({res['true'][0]:+.2f})")
    ax.axhline(res["observational"][0], color="0.3", ls=":", lw=1.5,
               label=f"observational ({res['observational'][0]:+.2f})")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
    fig.colorbar(sm, ax=ax, label="beta")
    for s, m in markers.items():
        ax.scatter([], [], marker=m, color="0.6", edgecolor="k", label=f"seed {s}")
    ax.scatter([], [], marker="X", color="tab:red", edgecolor="k", label="earlier designs (13, 13b, 13c)")
    rho = res["verdict"].get("h2_numbers", {}).get("spearman")
    ax.set_title("Share of the post-start drop predicted vs learned hour-1 effect"
                 + (f"\nSpearman {rho:.2f} over the 15 grid runs" if rho is not None else ""), fontsize=10)
    ax.set_xlabel("share of the 12b post-start drop predicted (%)")
    ax.set_ylabel("learned hour-1 effect (mmHg)")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_effect_beta(res, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs, means, los, his = [], [], [], []
    for b in res["betas"]:
        v = [r["learned"]["mean"][0] for r in res["runs"].values() if r["beta"] == b and r["learned"].get("mean")]
        if v:
            xs.append(b)
            means.append(np.mean(v))
            los.append(np.mean(v) - min(v))
            his.append(max(v) - np.mean(v))
            ax.scatter([b] * len(v), v, color="0.5", s=15, zorder=2)
    ax.errorbar(xs, means, yerr=[los, his], fmt="o-", color="tab:purple", capsize=4, label="mean over seeds (min-max)")
    ax.axhline(res["true"][0], color="k", ls="--", label=f"true ({res['true'][0]:+.2f})")
    ax.axhline(res["observational"][0], color="0.3", ls=":", label=f"observational ({res['observational'][0]:+.2f})")
    ax.invert_xaxis()
    ax.set_xlabel("beta")
    ax.set_ylabel("learned hour-1 effect (mmHg)")
    ax.set_title("Learned hour-1 effect vs beta (12b data, G = 15)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
