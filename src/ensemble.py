"""
ensemble.py - Stage 6: epistemic uncertainty from an ensemble of 5 models per dataset.

Each member has the same architecture, data and settings; only the random seed
(initial weights, batch order) differs. Seed 0 is the Stage 4 model.

Two kinds of spread in the imagined MAP of one patient at one future hour:
    aleatoric = spread of the 50 samples WITHIN one model (noise the model expects
                even if it were perfect), averaged over the 5 members
    epistemic = spread of the 5 members' predictions (their medians) BETWEEN models
                (disagreement, i.e. what the data did not pin down)

Run from the project root:
    python src/ensemble.py train    # trains seeds 1-4 for both datasets
    python src/ensemble.py test     # causal test with all 5 members, plots, table
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout                   # noqa: E402
from train import OUT, train     # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
DATASETS = ["policy", "explore"]
FIG = OUT / "figures"
ARMS = {0: ("u=0", "tab:orange"), 1: ("u=1", "tab:green")}


def model_file(name, seed):
    return OUT / (f"model_{name}.pt" if seed == 0 else f"model_{name}_s{seed}.pt")


def run(name):
    """Causal test for every member on the same patients. Returns a dict with
    imagined (M, 2, N, S, H) mmHg, effect (M, N, H) and the true results."""
    members = [rollout.run_causal_test(name, seed=0, model_file=model_file(name, s)) for s in SEEDS]
    r = members[0]
    return dict(name=name, idx=r["idx"], true_effect=r["true_effect"], true_map=r["true_map"],
                true_hist=r["true_hist"], obs_hist=r["obs_hist"], obs_mask=r["obs_mask"],
                u_hist=r["u_hist"],
                imagined=np.stack([m["imagined"] for m in members]),          # (M, 2, N, S, H)
                effect=np.stack([m["pred_effect"] for m in members]))         # (M, N, H)


def spreads(r):
    """Aleatoric and epistemic SD (mmHg), averaged over patients -> each (2 arms, H)."""
    im = r["imagined"]                                                        # (M, 2, N, S, H)
    aleatoric = im.std(axis=3).mean(axis=0)                                   # (2, N, H)
    epistemic = np.median(im, axis=3).std(axis=0)                             # (2, N, H)
    return aleatoric.mean(axis=1), epistemic.mean(axis=1)                     # (2, H) each


def plot_spreads(results):
    hours = np.arange(1, rollout.HORIZON + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, r in enumerate(results):
        ax = axes[row, 0]
        for m, eff in enumerate(r["effect"]):                                 # eff (N, H)
            ax.plot(hours, eff.mean(0), color="tab:blue", alpha=0.6, lw=1.2,
                    label="each ensemble member (mean over patients)" if m == 0 else None)
        ax.plot(hours, r["true_effect"].mean(0), "k--", lw=1.5, label="true")
        ax.set_title(f"{r['name']} ensemble: effect MAP(u=1) - MAP(u=0), n={r['effect'].shape[1]}")
        ax.set_xlabel("hours after hour 24")
        ax.set_ylabel("mmHg")
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        alea, epi = spreads(r)
        for arm, (lab, c) in ARMS.items():
            ax.plot(hours, alea[arm], color=c, ls=":", lw=1.5, label=f"{lab} aleatoric (within model)")
            ax.plot(hours, epi[arm], color=c, lw=2, label=f"{lab} epistemic (between models)")
        ax.set_title(f"{r['name']}: spread of imagined MAP (SD, mean over patients)")
        ax.set_xlabel("hours after hour 24")
        ax.set_ylabel("SD (mmHg)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "ensemble_spread.png", dpi=110)
    plt.close(fig)


def plot_example(results):
    """Lowest-MAP patient per dataset: each member's median as a thin line, the
    sampling band of member 0 shaded, and the true trajectories dashed."""
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 4.3), squeeze=False)
    T0, H = rollout.T0, rollout.HORIZON
    for ax, r in zip(axes[0], results):
        j = int(np.argmin(r["obs_hist"][:, T0]))
        hist_h, fut_h = np.arange(T0 + 1), np.arange(T0 + 1, T0 + H + 1)
        ax.plot(hist_h, r["true_hist"][j], color="0.4", lw=1, label="true MAP (hidden)")
        obs = r["obs_mask"][j]
        ax.plot(hist_h[obs], r["obs_hist"][j][obs], ".", color="k", label="MAP_obs (history)")
        for arm, (lab, c) in ARMS.items():
            lo, hi = np.percentile(r["imagined"][0, arm, j], [10, 90], axis=0)
            ax.fill_between(fut_h, lo, hi, color=c, alpha=0.2, label=f"{lab}: 10-90% samples, one model")
            for m in range(len(SEEDS)):
                ax.plot(fut_h, np.median(r["imagined"][m, arm, j], axis=0), color=c, lw=1,
                        label=f"{lab}: median of each of 5 models" if m == 0 else None)
            ax.plot(np.r_[T0, fut_h], np.r_[r["true_hist"][j, T0], r["true_map"][arm, j]],
                    color=c, ls="--", lw=1.5, label=f"{lab}: true")
        ax.axvline(T0 + 0.5, color="0.5", lw=0.8)
        ax.set_title(f"{r['name']} ensemble, test patient {r['idx'][j]}")
        ax.set_xlabel("hour")
        ax.set_ylabel("MAP (mmHg)")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "ensemble_example.png", dpi=110)
    plt.close(fig)


def print_table(results):
    print("Imagined MAP spread, SD in mmHg (mean over patients): aleatoric | epistemic")
    print(f"{'dataset':<8} {'arm':<4} " + "  ".join(f"{'+' + str(k) + ' h':>15}" for k in (1, 6, 12)))
    for r in results:
        alea, epi = spreads(r)
        for arm, (lab, _) in ARMS.items():
            print(f"{r['name']:<8} {lab:<4} " + "  ".join(
                f"{alea[arm, k - 1]:6.2f} | {epi[arm, k - 1]:5.2f}" for k in (1, 6, 12)))
    print("\nEffect at +12 h (mmHg): true vs each member's mean over patients")
    for r in results:
        per_member = r["effect"][:, :, 11].mean(1)
        print(f"{r['name']:<8} true {r['true_effect'][:, 11].mean():5.1f} | members "
              + " ".join(f"{e:5.1f}" for e in per_member)
              + f" | ensemble mean {per_member.mean():5.1f}, range {np.ptp(per_member):4.1f}")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["train"]:
        for name in DATASETS:
            for seed in SEEDS[1:]:
                print(f"== {name}, seed {seed}")
                train(name, seed=seed)
    elif sys.argv[1:] == ["test"]:
        results = [run(name) for name in DATASETS]
        print_table(results)
        plot_spreads(results)
        plot_example(results)
        print(f"figures saved under {FIG}")
    else:
        sys.exit("usage: python src/ensemble.py train|test")
