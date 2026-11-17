"""
simulator.py - Stage 1: simulated ICU patients, one step per hour.

Hidden state per patient (the model never sees it):
    m  mean arterial pressure, MAP (mmHg)
    s  sepsis severity (0 = none, 1 = severe)
    c  vasopressor effect (follows the infusion rate with a lag)
    v  extra fluid volume (L); 20% washes out every hour
Actions per hour:
    u  vasopressor rate in [0, 1]
    f  fluid bolus (L)
Observations: MAP_obs and HR_obs, each missing with probability 0.1.

Order of events inside hour t:
    1. observe x_t from the state at hour t
    2. choose action a_t = (u_t, f_t) from what has been observed
    3. step to hour t+1: c, v and s update first, then m uses the NEW c, v, s.
       So an action taken at hour t already changes MAP at hour t+1.

Run from the project root:  python src/simulator.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_PATIENTS = 2000
T_HOURS = 48
P_MISSING = 0.1
TAU, KAPPA = 2.0, 0.05
ALPHA, BETA, GAMMA, DELTA = 0.1, 8.0, 6.0, 4.0

OUT = Path(__file__).resolve().parent.parent / "outputs"


# ---------------------------------------------------------------- dynamics

def init_patients(n, rng):
    """Draw per-patient constants and the hour-0 hidden state.
    Patients arrive already septic (s = s_star), untreated, at baseline MAP."""
    m_base = rng.normal(80, 5, n)                     # (n,)
    s_star = rng.uniform(0, 1, n)                     # (n,)
    state = dict(m=m_base.copy(), s=s_star.copy(), c=np.zeros(n), v=np.zeros(n))
    return state, m_base, s_star


def step(state, u, f, m_base, s_star, noise):
    """Advance every patient by one hour.
    noise: (n, 2) standard normals, column 0 for s and column 1 for m.
    Calling step twice with the SAME noise but different actions answers
    "what would have happened under the other action?" (used in Stage 5)."""
    c = state["c"] + (u - state["c"]) / TAU
    v = 0.8 * state["v"] + f
    s = state["s"] + KAPPA * (s_star - state["s"]) + 0.02 * noise[:, 0]
    m = (state["m"] + ALPHA * (m_base - state["m"]) - BETA * s
         + GAMMA * c + DELTA * v + 1.5 * noise[:, 1])
    return dict(m=m, s=s, c=c, v=v)


def observe(state, rng):
    """Noisy vitals with random missingness. Returns x (n, 2) with NaN where
    missing, and mask (n, 2) with True where observed. Columns: MAP, HR."""
    n = len(state["m"])
    map_obs = state["m"] + 2 * rng.standard_normal(n)
    hr_obs = 80 + 0.6 * (75 - state["m"]) + 15 * state["s"] + 3 * rng.standard_normal(n)
    x = np.stack([map_obs, hr_obs], axis=1)           # (n, 2)
    mask = rng.random((n, 2)) > P_MISSING             # (n, 2)
    return np.where(mask, x, np.nan), mask


# ---------------------------------------------------------------- policies

def clinician_policy(map_seen, hours_since_fluid, rng):
    """Confounded on purpose: treatment depends on the patient's MAP.
    map_seen is the last observed MAP (NaN if none yet -> no treatment)."""
    n = len(map_seen)
    u = ((map_seen < 65) & (rng.random(n) < 0.8)).astype(float)
    f = 0.5 * ((map_seen < 60) & (hours_since_fluid > 3))
    return u, f


def random_policy(map_seen, hours_since_fluid, rng):
    """Ignores the patient: u is 0 or 1 with prob 0.5; a 0.5 L bolus with prob 0.25."""
    n = len(map_seen)
    u = (rng.random(n) < 0.5).astype(float)
    f = 0.5 * (rng.random(n) < 0.25)
    return u, f


# ---------------------------------------------------------------- simulate

def simulate(policy, n=N_PATIENTS, T=T_HOURS, seed=0):
    rng = np.random.default_rng(seed)
    state, m_base, s_star = init_patients(n, rng)
    keys = ["m", "s", "c", "v"]
    hidden = {k: np.zeros((n, T)) for k in keys}      # each (n, T)
    x = np.zeros((n, T, 2))                           # observed vitals, NaN = missing
    mask = np.zeros((n, T, 2), dtype=bool)
    actions = np.zeros((n, T, 2))                     # (u, f) chosen at hour t
    map_seen = np.full(n, np.nan)                     # last observed MAP
    hours_since_fluid = np.full(n, np.inf)

    for t in range(T):
        for k in keys:
            hidden[k][:, t] = state[k]
        x[:, t], mask[:, t] = observe(state, rng)
        map_seen = np.where(mask[:, t, 0], x[:, t, 0], map_seen)
        u, f = policy(map_seen, hours_since_fluid, rng)
        actions[:, t] = np.stack([u, f], axis=1)
        hours_since_fluid = np.where(f > 0, 1, hours_since_fluid + 1)
        state = step(state, u, f, m_base, s_star, rng.standard_normal((n, 2)))

    return dict(x=x, mask=mask, actions=actions, m_base=m_base, s_star=s_star,
                **{f"hidden_{k}": v for k, v in hidden.items()})


def summarise(name, d):
    m, u = d["hidden_m"], d["actions"][:, :, 0]
    f = d["actions"][:, :, 1]
    # Naive "effect" of vasopressors: MAP one hour after u=1 vs after u=0.
    m_next, u_now = m[:, 1:], u[:, :-1]
    naive = m_next[u_now == 1].mean() - m_next[u_now == 0].mean()
    print(f"{name:>7}: MAP true mean {m.mean():5.1f}  range [{m.min():6.1f}, {m.max():6.1f}]"
          f" | hours with u=1 {u.mean():.0%}, with bolus {(f > 0).mean():.0%}"
          f" | observed {d['mask'].mean():.1%}"
          f" | naive MAP(t+1 | u=1) - MAP(t+1 | u=0) = {naive:+.1f} mmHg")


# ---------------------------------------------------------------- plotting

def plot_patients(d, title, path, n_show=4):
    """Patients picked across the range of sepsis severity s_star."""
    order = np.argsort(d["s_star"])
    idx = order[np.linspace(0, len(order) - 1, n_show).astype(int)]
    hours = np.arange(d["x"].shape[1])
    fig, axes = plt.subplots(3, n_show, figsize=(4 * n_show, 7.5), sharex=True)
    for j, i in enumerate(idx):
        ax = axes[0, j]
        ax.plot(hours, d["hidden_m"][i], color="0.3", label="true MAP (hidden)")
        ax.plot(hours, d["x"][i, :, 0], ".", color="tab:red", label="MAP_obs")
        ax.axhline(65, ls="--", lw=0.8, color="tab:red")
        ax.set_title(f"patient {i}  s*={d['s_star'][i]:.2f}  m_base={d['m_base'][i]:.0f}")
        axes[1, j].plot(hours, d["x"][i, :, 1], ".", color="tab:blue", label="HR_obs")
        ax2 = axes[2, j]
        ax2.step(hours, d["actions"][i, :, 0], where="post", color="tab:purple", label="u vasopressor")
        ax2.bar(hours, d["actions"][i, :, 1], width=0.8, color="tab:green", alpha=0.6, label="f bolus (L)")
        ax2.set_ylim(-0.05, 1.1)
        ax2.set_xlabel("hour")
    axes[0, 0].set_ylabel("MAP (mmHg)")
    axes[1, 0].set_ylabel("HR (bpm)")
    axes[2, 0].set_ylabel("action")
    for row in axes[:, 0]:
        row.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    for name, policy, seed in [("policy", clinician_policy, 0), ("random", random_policy, 1)]:
        d = simulate(policy, seed=seed)
        np.savez_compressed(OUT / f"sim_{name}.npz", **d)
        summarise(name, d)
        plot_patients(d, f"{name} actions", OUT / "figures" / f"patients_{name}.png")
    print(f"saved datasets and figures under {OUT}")
