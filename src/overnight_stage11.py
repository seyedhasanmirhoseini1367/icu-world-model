"""
overnight_stage11.py - one unattended Kaggle GPU run for the Stage 10b noise floor and Stage 11.

Order (each block is its own step; a failing step is logged and the run continues):
  extract     MIMIC extraction once -> real data + K = 10, 30, 100 injected copies (in tmp only)
  units       Step 4: vasopressor rows per rateuom, and rows in the "ELSE 1.0" branch
  old_ref     Part B of the existing models (real, K = 10, 30, 100), paired per moment
  old_k0_s1   Step 1: old architecture, real data, seed 1 -> noise floor (seed 1 minus seed 0)
  new_k0      Step 2: act_now variant on real data, seed 0, then Step 3 a), c), d) old vs new
  new_k100 / new_k10 / new_k30   variant on the injected data, then Step 3 b)
  new_k0_s1   extra: variant on real data, seed 1 -> noise floor of the new architecture
  final       combined tables and figures

Every step writes its JSON (and figure) to `work` as soon as it finishes, and summary.json is
rewritten after every step, so a run that is cut short still leaves everything done so far.
Only aggregate numbers, figures, histories and model weights are written; the hourly data stays
in `tmp`. The per-moment effects used for pairing are kept in memory only.
"""
import json
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402

import ablate_mimic               # noqa: E402
import evaluate_mimic             # noqa: E402
import imagine_mimic              # noqa: E402
import mimic_data                 # noqa: E402
import semisynthetic as SS        # noqa: E402
import train as T                 # noqa: E402
import vaso_units                 # noqa: E402

KS = (10, 30, 100)
REPORT_HOURS = (1, 3, 6)


class Runner:
    def __init__(self, work, tmp, total_budget_h=10.5, train_budget_h=1.25, eval_reserve_h=0.6):
        self.work, self.tmp = Path(work), Path(tmp)
        (self.work / "figures").mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.t_start = time.time()
        self.total_s, self.train_s, self.reserve_s = total_budget_h * 3600, train_budget_h * 3600, eval_reserve_h * 3600
        self.steps = []

    def elapsed_h(self):
        return (time.time() - self.t_start) / 3600

    def save(self, name, obj):
        (self.work / name).write_text(json.dumps(obj, indent=2, default=float))

    def write_summary(self):
        self.save("summary.json", dict(elapsed_h=self.elapsed_h(), steps=self.steps,
                                       succeeded=[s["step"] for s in self.steps if s["status"] == "ok"],
                                       failed=[s["step"] for s in self.steps if s["status"] == "failed"],
                                       skipped=[s["step"] for s in self.steps if s["status"] == "skipped"]))

    def step(self, name, fn, needs=()):
        missing = [n for n in needs if not self.ok(n)]
        t0 = time.time()
        print(f"\n===== step {name} (elapsed {self.elapsed_h():.2f} h) =====", flush=True)
        if missing:
            rec = dict(step=name, status="skipped", reason=f"needs {missing}")
        else:
            try:
                note = fn()
                rec = dict(step=name, status="ok", **({"note": note} if note else {}))
            except Exception as e:
                traceback.print_exc()
                rec = dict(step=name, status="failed", error=f"{type(e).__name__}: {str(e)[:500]}")
            finally:
                T.finish_wandb()                    # never leave a W&B run open after a failure
        rec["minutes"] = (time.time() - t0) / 60
        self.steps.append(rec)
        print(f"===== step {name}: {rec['status']} ({rec['minutes']:.1f} min)", flush=True)
        self.write_summary()

    def ok(self, name):
        return any(s["step"] == name and s["status"] == "ok" for s in self.steps)


def checkpoint_epoch(history):
    """The epoch whose weights train() kept: improvements smaller than 1e-4 do not count."""
    best, epoch = float("inf"), None
    for h in history:
        if h["val"]["loss"] < best - 1e-4:
            best, epoch = h["val"]["loss"], h["epoch"]
    return epoch


def spread(per_a, per_b):
    """Per moment a minus b, paired by key -> mean, 10th and 90th percentile per hour."""
    keys = [k for k in per_a if k in per_b]
    if not keys:
        return dict(n_paired=0)
    d = torch.stack([per_a[k] - per_b[k] for k in keys])                     # (M, H)
    return dict(n_paired=len(keys), mean=d.mean(0).tolist(), p10=d.quantile(0.1, dim=0).tolist(),
                p90=d.quantile(0.9, dim=0).tolist(), sd=d.std(0).tolist())


def against_noise(rec, noise):
    """Per hour: does a recovered effect clearly exceed the noise floor?
    strict: the recovered 10-90% range lies entirely above the noise floor's 90th percentile.
    mean:   the recovered mean lies above the noise floor's 90th percentile."""
    if not rec.get("n_paired") or not noise.get("n_paired"):
        return None
    return dict(strict=[lo > hi for lo, hi in zip(rec["recovered_p10"], noise["p90"])],
                mean=[m > hi for m, hi in zip(rec["recovered_mean"], noise["p90"])])


def run_all(icu_dir, weights, work, tmp, device="cpu", max_stays=0, max_epochs=None,
            b_vital="MAP", b_limit=imagine_mimic.MAP_LIMIT, total_budget_h=10.5, train_budget_h=1.25):
    R = Runner(work, tmp, total_budget_h, train_budget_h)
    T.OUT, T.FIG, T.DEVICE = R.work, R.work / "figures", device
    if max_epochs:
        T.MAX_EPOCHS = max_epochs
    S = dict(npz={}, sets={}, per={}, partb={}, models=dict(old_k0=weights["old_k0"]), clips=None)
    for k in KS:
        S["models"][f"old_k{k}"] = weights[f"old_k{k}"]

    def sets(k):
        if k not in S["sets"]:
            S["sets"][k] = mimic_data.load_splits_mimic(S["npz"][k])
        return S["sets"][k]

    def part_b(tag, k):
        """Part B for model `tag` on dataset K = k; keeps the per-moment effects in memory."""
        (ds, stats) = sets(k)
        b = imagine_mimic.part_b(SS.load_model(S["models"][tag], device), ds["test"], stats, device,
                                 vital=b_vital, limit=b_limit, keep_per_moment=True)
        S["per"][tag] = b.pop("_per_moment", {})
        S["partb"][tag] = b
        return b

    def hours():
        b = S["partb"].get("old_k0") or S["partb"].get("new_k0")
        return b["horizons"]

    # ---------------------------------------------------------------- extraction
    def extract():
        real = mimic_data.extract(icu_dir, max_stays)
        S["npz"][0] = R.tmp / "mimic_k0.npz"
        np.savez_compressed(S["npz"][0], **real)
        for k in KS:
            S["npz"][k] = R.tmp / f"mimic_k{k}.npz"
            np.savez_compressed(S["npz"][k], **SS.inject(real, k=float(k)))
        S["clips"] = mimic_data.clip_counts(S["npz"][0])
        R.save("extract.json", dict(n_stays=int(real["x"].shape[0]), clipped=S["clips"], ks=list(KS)))
        return f"{S['clips']}"

    R.step("extract", extract)

    # ---------------------------------------------------------------- step 4: units (report only)
    def units():
        r = vaso_units.count_units(icu_dir)
        R.save("step4_vaso_units.json", r)
        print(vaso_units.report(r))

    R.step("units", units)

    # ---------------------------------------------------------------- old models, Part B
    def old_ref():
        out = {}
        for k in (0,) + KS:
            b = part_b(f"old_k{k}", k)
            out[f"old_k{k}"] = b
        rec = {str(k): SS.recovered_stats(S["per"][f"old_k{k}"], S["per"]["old_k0"],
                                          SS.injected_effect(imagine_mimic.DOSE, hours(), k=float(k)))
               for k in KS}
        R.save("old_partb.json", dict(partb=out, recovered=rec))
        for k in KS:
            r = rec[str(k)]
            if r.get("n_paired"):
                print(f"old K={k}: recovered at 1/3/6 h " +
                      " ".join(f"{r['recovered_mean'][h - 1]:+.2f}" for h in REPORT_HOURS))

    R.step("old_ref", old_ref, needs=["extract"])

    # ---------------------------------------------------------------- training helper
    def train_step(tag, k, seed, act_now):
        def fn():
            left = R.total_s - (time.time() - R.t_start) - R.reserve_s
            if left < 0.3 * 3600:
                raise RuntimeError(f"not enough time left ({left / 3600:.2f} h) for training {tag}")
            T.TIME_BUDGET_S = min(R.train_s, left)
            loaders, _ = mimic_data.make_loaders_mimic(S["npz"][k], batch_size=T.BATCH_SIZE)
            name = "mimic_s11_" + (tag[:-3] if tag.endswith("_s1") else tag)   # train() appends _s1 itself
            _, history, _ = T.train(
                name, seed=seed, loaders=loaders,
                model_kwargs=dict(n_vitals=len(mimic_data.VITALS), n_actions=len(mimic_data.ACTIONS),
                                  act_now=act_now),
                wandb_config=dict(stage="11", variant="act_now" if act_now else "original", act_now=act_now,
                                  semisynthetic=k > 0, injected_k_mmhg_per_dose=float(k),
                                  injected_tau_h=SS.TAU_H, nee_clip_mcg_kg_min=mimic_data.NEE_CLIP,
                                  fluid_clip_l_per_hour=mimic_data.FLUID_CLIP_L, **(S["clips"] or {})))
            file_tag = name if seed == T.SEED else f"{name}_s{seed}"
            S["models"][tag] = R.work / f"model_{file_tag}.pt"
            best = next(h for h in history if h["epoch"] == checkpoint_epoch(history))
            info = dict(tag=tag, model_file=S["models"][tag].name, seed=seed, act_now=act_now, k=k,
                        checkpoint_epoch=best["epoch"], epochs=len(history),
                        seconds_per_epoch_median=float(np.median([h["seconds"] for h in history])),
                        val_loss=best["val"]["loss"], val_recon=best["val"]["recon"], val_kl=best["val"]["kl"],
                        val_kl_dims=best["val"]["kl_dims"],
                        stopped_early=len(history) - checkpoint_epoch(history) >= T.PATIENCE,
                        time_budget_h=T.TIME_BUDGET_S / 3600)
            R.save(f"train_{tag}.json", info)
            b = part_b(tag, k)
            summary = dict(checkpoint_epoch=info["checkpoint_epoch"], epochs=info["epochs"])
            if b.get("mean_diff"):
                summary.update({f"effect/imagined_{h}h": b["mean_diff"][h - 1] for h in REPORT_HOURS})
            T.finish_wandb(summary)
            R.save(f"partb_{tag}.json", b)
            return f"checkpoint epoch {info['checkpoint_epoch']} of {info['epochs']}"
        return fn

    # ---------------------------------------------------------------- step 1: noise floor, old architecture
    def noise_old():
        nf = spread(S["per"]["old_k0_s1"], S["per"]["old_k0"])
        R.save("step1_noise_floor_old.json", dict(seed_a=1, seed_b=0, **nf))
        if nf.get("n_paired"):
            print("noise floor (old, seed 1 minus seed 0) at 1/3/6 h: " + "  ".join(
                f"{nf['mean'][h - 1]:+.2f} [{nf['p10'][h - 1]:+.2f}..{nf['p90'][h - 1]:+.2f}]" for h in REPORT_HOURS))

    R.step("train_old_k0_s1", train_step("old_k0_s1", 0, 1, False), needs=["extract"])
    R.step("noise_floor_old", noise_old, needs=["train_old_k0_s1", "old_ref"])

    # ---------------------------------------------------------------- step 2 + 3 on real data
    R.step("train_new_k0", train_step("new_k0", 0, 0, True), needs=["extract"])

    def eval_onestep():
        out = {}
        for tag in ("old_k0", "new_k0"):
            out[tag] = evaluate_mimic.evaluate(str(S["npz"][0]), str(S["models"][tag]))
            print(f"--- one-step, {tag}\n" + evaluate_mimic.table(out[tag]))
        R.save("step3a_onestep.json", out)

    def eval_ablation():
        out = {}
        for tag in ("old_k0", "new_k0"):
            out[tag] = ablate_mimic.run(str(S["npz"][0]), str(S["models"][tag]), device)
            print(f"--- ablation, {tag}\n" + ablate_mimic.report(out[tag]))
        R.save("step3d_ablation.json", out)

    def eval_partb_real():
        out = {tag: S["partb"][tag] for tag in ("old_k0", "new_k0")}
        out["new_minus_old_paired"] = spread(S["per"]["new_k0"], S["per"]["old_k0"])
        R.save("step3c_partb_real.json", out)
        for tag in ("old_k0", "new_k0"):
            b = out[tag]
            if b.get("mean_diff"):
                print(f"{tag}: imagined NEE 0.1 minus 0 at 1/3/6 h " +
                      " ".join(f"{b['mean_diff'][h - 1]:+.2f}" for h in REPORT_HOURS))

    R.step("eval_onestep", eval_onestep, needs=["train_new_k0"])
    R.step("eval_partb_real", eval_partb_real, needs=["train_new_k0", "old_ref"])
    R.step("eval_ablation", eval_ablation, needs=["train_new_k0"])

    # ---------------------------------------------------------------- variant on K = 100, 10, 30
    for k in (100, 10, 30):
        R.step(f"train_new_k{k}", train_step(f"new_k{k}", k, 0, True), needs=["extract"])

    # ---------------------------------------------------------------- extra: noise floor, new architecture
    R.step("train_new_k0_s1", train_step("new_k0_s1", 0, 1, True), needs=["extract"])

    def noise_new():
        nf = spread(S["per"]["new_k0_s1"], S["per"]["new_k0"])
        R.save("noise_floor_new.json", dict(seed_a=1, seed_b=0, **nf))

    R.step("noise_floor_new", noise_new, needs=["train_new_k0_s1", "train_new_k0"])

    # ---------------------------------------------------------------- final tables and figures
    def final():
        nf_old = spread(S["per"]["old_k0_s1"], S["per"]["old_k0"]) if "old_k0_s1" in S["per"] else {}
        nf_new = spread(S["per"]["new_k0_s1"], S["per"]["new_k0"]) if "new_k0_s1" in S["per"] else {}
        table = {}
        for k in KS:
            inj = SS.injected_effect(imagine_mimic.DOSE, hours(), k=float(k))
            row = {}
            for arch in ("old", "new"):
                semi, real = f"{arch}_k{k}", f"{arch}_k0"
                if semi in S["per"] and real in S["per"]:
                    rec = SS.recovered_stats(S["per"][semi], S["per"][real], inj)
                    own = nf_old if arch == "old" else (nf_new or nf_old)
                    rec["vs_noise_floor_old"] = against_noise(rec, nf_old)
                    rec["vs_noise_floor_own"] = against_noise(rec, own)
                    row[arch] = rec
            table[str(k)] = row
        res = dict(horizons=hours(), report_hours=list(REPORT_HOURS), dose=imagine_mimic.DOSE,
                   noise_floor_old=nf_old, noise_floor_new=nf_new, recovery=table,
                   criterion=dict(strict="recovered 10-90% range entirely above the noise floor's 90th percentile",
                                  mean="recovered mean above the noise floor's 90th percentile"))
        R.save("stage11_recovery.json", res)
        plot_recovery(res, R.work / "figures" / "stage11_recovery.png")
        plot_real(S["partb"], nf_old, R.work / "figures" / "stage11_partb_real.png")

    R.step("final", final, needs=["old_ref"])
    print(f"\nall steps done in {R.elapsed_h():.2f} h")
    return R


# ---------------------------------------------------------------- figures

def plot_recovery(res, path):
    hours = [0] + res["horizons"]
    fig, axes = plt.subplots(1, len(KS), figsize=(5 * len(KS), 4.3), squeeze=False)
    nf = res["noise_floor_old"]
    for ax, k in zip(axes[0], KS):
        row = res["recovery"][str(k)]
        any_rec = next(iter(row.values()), None)
        if any_rec:
            ax.plot(hours, [0] + any_rec["injected"], "k--", lw=2, label="injected")
        if nf.get("n_paired"):
            ax.fill_between(hours, [0] + nf["p10"], [0] + nf["p90"], color="0.6", alpha=0.35,
                            label="noise floor (seed 1 - seed 0), 10-90%")
        for arch, color in (("old", "tab:blue"), ("new", "tab:orange")):
            r = row.get(arch)
            if not r or not r.get("n_paired"):
                continue
            ax.fill_between(hours, [0] + r["recovered_p10"], [0] + r["recovered_p90"], color=color, alpha=0.2)
            ax.plot(hours, [0] + r["recovered_mean"], "o-", color=color,
                    label=f"{arch} model, recovered mean (10-90%)")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(f"K = {k} mmHg per mcg/kg/min")
        ax.set_xlabel("hours after the decision hour")
        ax.set_ylabel("MAP difference, NEE 0.1 minus 0 (mmHg)")
        ax.legend(fontsize=7)
    fig.suptitle("Recovered vs injected effect: original model vs act_now variant")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_real(partb, nf, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    n = next((b["n_moments"] for b in partb.values() if b.get("n_moments")), 0)
    for tag, color, label in (("old_k0", "tab:blue", "original model"), ("new_k0", "tab:orange", "act_now variant")):
        b = partb.get(tag)
        if not b or not b.get("mean_diff"):
            continue
        hours = [0] + b["horizons"]
        ax.fill_between(hours, [0] + b["moment_p10"], [0] + b["moment_p90"], color=color, alpha=0.2)
        ax.plot(hours, [0] + b["mean_diff"], "o-", color=color, label=f"{label}, mean (10-90% across patients)")
    if nf.get("n_paired"):
        h = [0] + list(range(1, len(nf["p10"]) + 1))
        ax.fill_between(h, [0] + nf["p10"], [0] + nf["p90"], color="0.6", alpha=0.3, label="noise floor, 10-90%")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("hours after the decision hour")
    ax.set_ylabel("imagined MAP, NEE 0.1 minus 0 (mmHg)")
    ax.set_title(f"Real MIMIC data: imagined vasopressor effect (n = {n} moments)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
