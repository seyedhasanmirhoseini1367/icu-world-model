"""
rollout.py - Stage 5: imagination and the causal test.

Imagination: give the model a patient's real history (hours 0..24), then let it
dream the next 12 hours using only the PRIOR. Each imagined x_{t+1} is fed back
as the next token, as if it had been observed (mask = 1, delta = 0).

Causal test, per test patient whose observed MAP at hour 24 is below 70:
    predicted effect(k) = median imagined MAP(24+k | u=1)  - median imagined MAP(24+k | u=0)
    true effect(k)      = simulator MAP(24+k | u=1)          - simulator MAP(24+k | u=0)
Both arms keep the action fixed from hour 24 on (no fluids). The true arms start
from the patient's real hidden state at hour 24 and share the same noise.

Run from the project root:  python src/rollout.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_splits, norm_stats   # noqa: E402
from model import WorldModel                # noqa: E402
import simulator                            # noqa: E402

T0 = 24            # last real hour
HORIZON = 12       # imagined hours
N_SAMPLES = 50
MAP_THRESHOLD = 70.0
OUT = Path(__file__).resolve().parent.parent / "outputs"
FIG = OUT / "figures"


# ---------------------------------------------------------------- imagination

@torch.no_grad()
def imagine(model, x, mask, delta, a_hist, a_future):
    """Roll the model forward from a real history using the prior only.

    x, mask, delta (S, L0, V)  real hours 0..T0, repeated for S samples
    a_hist         (S, L0-1, A) real actions at hours 0..T0-1
    a_future       (S, H, A)    actions at hours T0..T0+H-1, one per imagined step
    returns        (S, H, V)    imagined vitals for hours T0+1..T0+H (z-units)
    """
    model.eval()
    S, L0, _ = x.shape
    a_prev = torch.cat([torch.zeros_like(a_hist[:, :1]), a_hist], dim=1)  # (S, L0, A): a_{t-1}
    imagined = []
    for k in range(a_future.shape[1]):
        L = x.shape[1]                                                    # current length, L0 + k
        tokens = model.embed(x, mask, delta, a_prev, torch.arange(L, device=x.device))  # (S, L, D)
        h_t = model.encode(tokens)[:, -1]                                 # (S, D) summary up to the last hour
        a_t = a_future[:, k]                                              # (S, A)
        mu, log_sigma = model.prior_from(h_t, a_t)                       # (S, Z) each
        z = model.sample(mu, log_sigma)                                   # (S, Z)
        dec_mu, dec_ls = model.decoder(torch.cat([z, h_t], -1))           # (S, V) each
        x_new = dec_mu + dec_ls.exp() * torch.randn_like(dec_mu)          # (S, V) one sampled observation
        imagined.append(x_new)
        # Feed the imagined hour back in as if it had been observed.
        x = torch.cat([x, x_new[:, None]], 1)                             # (S, L+1, V)
        mask = torch.cat([mask, torch.ones_like(x_new[:, None])], 1)      # (S, L+1, V)
        delta = torch.cat([delta, torch.zeros_like(x_new[:, None])], 1)   # (S, L+1, V)
        a_prev = torch.cat([a_prev, a_t[:, None]], 1)                     # (S, L+1, A)
    return torch.stack(imagined, 1)                                       # (S, H, V)


def imagine_both_arms(model, ds, i):
    """Patient i of dataset ds: N_SAMPLES rollouts with u=0 and N_SAMPLES with u=1.
    Returns imagined MAP in z-units, (2, N_SAMPLES, HORIZON): arm 0 is u=0, arm 1 is u=1."""
    S = 2 * N_SAMPLES
    rep = lambda t: t[i:i + 1].expand(S, *t.shape[1:])                    # noqa: E731  (1,...) -> (S,...)
    x, mask, delta = (rep(t[:, :T0 + 1]) for t in (ds.x, ds.mask, ds.delta))  # (S, T0+1, V)
    a_hist = rep(ds.a[:, :T0])                                            # (S, T0, A)
    a_future = torch.zeros(S, HORIZON, 2)                                 # (S, H, A), f = 0
    a_future[N_SAMPLES:, :, 0] = 1.0                                      # second half: u = 1
    out = imagine(model, x, mask, delta, a_hist, a_future)                # (S, H, V)
    return out[..., 0].reshape(2, N_SAMPLES, HORIZON)                     # MAP only


# ---------------------------------------------------------------- ground truth

def true_both_arms(sim, idx, seed=0):
    """Rerun the simulator from the true hidden state at hour T0 with u=0 and u=1,
    using the SAME noise for both arms. idx (N,) patient rows in the .npz.
    Returns true MAP (2, N, HORIZON) in mmHg and the true MAP history (N, T0+1)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((HORIZON, len(idx), 2))                   # shared by both arms
    m_base, s_star = sim["m_base"][idx], sim["s_star"][idx]               # (N,) each
    arms = []
    for u in (0.0, 1.0):
        state = {k: sim[f"hidden_{k}"][idx, T0].copy() for k in "mscv"}  # each (N,)
        traj = []
        for k in range(HORIZON):
            state = simulator.step(state, np.full(len(idx), u), np.zeros(len(idx)),
                                   m_base, s_star, noise[k])
            traj.append(state["m"])
        arms.append(np.stack(traj, 1))                                    # (N, H)
    return np.stack(arms), sim["hidden_m"][idx, :T0 + 1]


# ---------------------------------------------------------------- the experiment

def run_causal_test(name, seed=0, model_file=None):
    """model_file defaults to outputs/model_{name}.pt (the Stage 4 model)."""
    torch.manual_seed(seed)
    stats = norm_stats()
    to_mmhg = lambda z: z * stats["std"][0] + stats["mean"][0]           # noqa: E731
    model = WorldModel()
    model.load_state_dict(torch.load(model_file or OUT / f"model_{name}.pt"))
    test = load_splits(name)[0]["test"]
    sim = np.load(OUT / f"sim_{name}.npz")

    map24 = to_mmhg(test.x[:, T0, 0])                                     # (N_test,)
    picked = torch.where(test.mask[:, T0, 0].bool() & (map24 < MAP_THRESHOLD))[0]
    imagined = np.stack([to_mmhg(imagine_both_arms(model, test, int(i))).numpy()
                         for i in picked], 1)                             # (2, N, S, H) mmHg
    idx = test.idx[picked].numpy()
    true_map, true_hist = true_both_arms(sim, idx, seed)                  # (2, N, H), (N, T0+1)

    median = np.median(imagined, axis=2)                                  # (2, N, H)
    return dict(name=name, picked=picked.numpy(), idx=idx, imagined=imagined,
                pred_effect=median[1] - median[0],                        # (N, H)
                true_effect=true_map[1] - true_map[0],                    # (N, H)
                true_map=true_map, true_hist=true_hist,
                obs_hist=to_mmhg(test.x[picked, :T0 + 1, 0]).numpy(),     # (N, T0+1)
                obs_mask=test.mask[picked, :T0 + 1, 0].bool().numpy(),
                u_hist=test.a[picked, :T0, 0].numpy())                    # (N, T0)


# ---------------------------------------------------------------- plots

COLORS = dict(policy="tab:red", explore="tab:blue")


def plot_effects(results):
    hours = np.arange(1, HORIZON + 1)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for r in results:
        c = COLORS[r["name"]]
        pe, te = r["pred_effect"], r["true_effect"]                       # (N, H)
        ax.fill_between(hours, *np.percentile(pe, [10, 90], axis=0), color=c, alpha=0.15)
        ax.plot(hours, pe.mean(0), color=c, lw=2,
                label=f"{r['name']} model: predicted (n={len(pe)}, mean, 10-90% of patients)")
        ax.plot(hours, te.mean(0), color=c, ls="--", lw=1.5, label=f"{r['name']} patients: true")
        ax2.scatter(te[:, 5], pe[:, 5], s=14, color=c, alpha=0.6, label=f"{r['name']}")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("hours after hour 24")
    ax.set_ylabel("MAP(u=1) - MAP(u=0)  (mmHg)")
    ax.set_title("Effect of starting vasopressors at hour 24")
    ax.legend(fontsize=8)
    lim = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]), max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
    ax2.plot(lim, lim, color="0.5", lw=0.8, ls=":", label="predicted = true")
    ax2.set_xlabel("true effect at +6 h (mmHg)")
    ax2.set_ylabel("predicted effect at +6 h (mmHg)")
    ax2.set_title("Per patient, 6 hours after starting")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "causal_effect.png", dpi=110)
    plt.close(fig)


def plot_examples(results, n_examples=2):
    fig, axes = plt.subplots(len(results), n_examples, figsize=(7 * n_examples, 4.2 * len(results)),
                             squeeze=False)
    for row, r in enumerate(results):
        order = np.argsort(r["obs_hist"][:, T0])                          # lowest MAP at hour 24 first
        for col, j in enumerate([order[0], order[len(order) // 2]][:n_examples]):
            ax = axes[row, col]
            hist_h, fut_h = np.arange(T0 + 1), np.arange(T0 + 1, T0 + HORIZON + 1)
            for t in np.where(r["u_hist"][j] == 1)[0]:
                ax.axvspan(t + 0.5, t + 1.5, color="tab:purple", alpha=0.15, lw=0)
            ax.plot(hist_h, r["true_hist"][j], color="0.4", lw=1, label="true MAP (hidden)")
            obs = r["obs_mask"][j]
            ax.plot(hist_h[obs], r["obs_hist"][j][obs], ".", color="k", label="MAP_obs (history)")
            for arm, c, lab in [(0, "tab:orange", "u=0"), (1, "tab:green", "u=1")]:
                samples = r["imagined"][arm, j]                           # (S, H)
                lo, med, hi = np.percentile(samples, [10, 50, 90], axis=0)
                ax.fill_between(fut_h, lo, hi, color=c, alpha=0.25)
                ax.plot(fut_h, med, color=c, lw=2, label=f"imagined {lab} (median, 10-90%)")
                ax.plot(np.r_[T0, fut_h], np.r_[r["true_hist"][j, T0], r["true_map"][arm, j]],
                        color=c, ls="--", lw=1.5, label=f"true {lab}")
            ax.axvline(T0 + 0.5, color="0.5", lw=0.8)
            ax.set_title(f"{r['name']} model, test patient {r['idx'][j]} "
                         f"(purple = vasopressor in history)")
            ax.set_xlabel("hour")
            ax.set_ylabel("MAP (mmHg)")
            ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "rollout_examples.png", dpi=110)
    plt.close(fig)


def summary_table(results):
    lines = [f"{'model':<8} {'n':>4}  " + "  ".join(f"{'+' + str(k) + ' h':>21}" for k in (1, 3, 6, 12)),
             f"{'':<8} {'':>4}  " + "  ".join(f"{'pred (p10..p90) | true':>21}" for _ in range(4))]
    for r in results:
        cells = []
        for k in (1, 3, 6, 12):
            pe, te = r["pred_effect"][:, k - 1], r["true_effect"][:, k - 1]
            p10, p90 = np.percentile(pe, [10, 90])
            cells.append(f"{pe.mean():5.1f} ({p10:4.1f}..{p90:4.1f}) | {te.mean():4.1f}")
        lines.append(f"{r['name']:<8} {len(r['pred_effect']):>4}  " + "  ".join(f"{c:>21}" for c in cells))
    return "\n".join(lines)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    results = [run_causal_test(name) for name in ("policy", "explore")]
    print("Effect of u=1 vs u=0 from hour 24, MAP in mmHg (mean over patients):")
    print(summary_table(results))
    plot_effects(results)
    plot_examples(results)
    print(f"figures saved under {FIG}")
