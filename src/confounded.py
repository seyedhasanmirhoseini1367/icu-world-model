"""
confounded.py - Stage 12: the semi-synthetic benchmark WITH a hidden confounder.

Real histories, real actions and real missingness are kept. On top of the measured MAP:

    MAP'_t = MAP_t + K * e_t - G * u_t

e_t   the dose seen through a 1-hour lag (Stage 10):  e_{t+1} = e_t + (1 - exp(-1)) (NEE_t - e_t), e_0 = 0
u_t   a hidden deterioration, never given to the model. A vasopressor START is an hour s with
      NEE_s > 0 and (s = 0 or NEE_{s-1} = 0). Each start s contributes the kernel
          k(d) = (d + 3) / 3     for -3 <= d <= 0     (linear ramp over the 3 h before, 1 at the start)
          k(d) = exp(-d / 4)     for d > 0            (decay with time constant 4 h after)
          k(d) = 0               otherwise
      with d = t - s, and u_t = max over all starts s of k(t - s)   (so 0 <= u_t <= 1).
So sicker patients get the drug, and they keep worsening for a while after it starts.

True causal effect of giving a constant dose D from the decision hour t (instead of none):
the drug does not change u, so it is only the drug term,
    true_h = K * D * (1 - exp(-h))                                     h = hours after t

What a model that learns the observational relationship E[MAP' | history, actions] should
imagine instead. Part B compares, for the same history, the plan "constant dose from t"
against "no dose". For the hour t + h the model has seen the actions of hours t .. t+h-1:
  - dose arm: a start at t, so u_{t+h} = k(h) = exp(-h / 4)  (moments have no NEE before t,
    so there is no earlier start);
  - no-dose arm: no start in t .. t+h-1, so u_{t+h} can only be the ramp of a start still
    to come, at s in t+h .. t+h+3. Its expected value is estimated from the TRAINING split,
    at moments chosen by the same rule as Part B, where NEE really stayed 0 in t .. t+h-1:
        U0_h = mean of u_{t+h} over those moments.
  So the confounding bias the model should show is
        bias_obs_h = -G * (exp(-h / 4) - U0_h)
  and the observational learned effect is true_h + bias_obs_h. This ignores what the
  history itself reveals about u (a ramp already under way lowers MAP' before t); it is
  the population-level prediction for these moments.
"""
import math

import numpy as np

import imagine_mimic
import mimic_data
import semisynthetic as SS

K_MMHG = 30.0
RAMP_H, DECAY_TAU_H = 3, 4.0
MAP = mimic_data.VITALS.index("MAP")
NEE = mimic_data.ACTIONS.index("NEE mcg/kg/min")


def vaso_starts(nee):
    """nee (N, T) -> bool (N, T): True where a vasopressor starts."""
    prev = np.concatenate([np.zeros_like(nee[:, :1]), nee[:, :-1]], 1)
    return (nee > 0) & (prev <= 0)


def kernel(d):
    """d = t - s (array of ints) -> contribution of a start at s to u_t."""
    d = np.asarray(d, dtype=float)
    ramp = np.where((d >= -RAMP_H) & (d <= 0), (d + RAMP_H) / RAMP_H, 0.0)
    decay = np.where(d > 0, np.exp(-np.clip(d, 0, None) / DECAY_TAU_H), 0.0)
    return ramp + decay


def hidden_u(nee):
    """nee (N, T) -> u (N, T) = max over starts s of kernel(t - s)."""
    starts = vaso_starts(nee)
    N, T = nee.shape
    u = np.zeros((N, T))
    t = np.arange(T)
    for i, s in zip(*np.nonzero(starts)):
        u[i] = np.maximum(u[i], kernel(t - s))
    return u


def inject_confounded(data, k=K_MMHG, g=0.0):
    """Copy of an extracted dataset with MAP' = MAP + k*e - g*u on measured MAP only."""
    out = SS.inject(data, k=k, tau=SS.TAU_H)                     # adds k * e (copies the arrays)
    u = hidden_u(out["a"][..., NEE])
    out["x"][..., MAP] = out["x"][..., MAP] - g * u               # NaN (missing) stays NaN
    out["confounder_g"], out["confounder_ramp_h"], out["confounder_tau_h"] = g, RAMP_H, DECAY_TAU_H
    return out


def true_effect(dose, hours, k=K_MMHG):
    return SS.injected_effect(dose, hours, k=k, tau=SS.TAU_H)


def no_dose_u(ds, stats, hours):
    """U0_h on one split: mean u_{t+h} at Part B moments where NEE stayed 0 in t .. t+h-1.
    Also returns the dose-arm counterpart from the data (moments where NEE started at t)."""
    moments = imagine_mimic.find_moments(ds, stats)
    nee = ds.a[..., NEE].numpy()
    u = hidden_u(nee)
    out = dict(n_moments=len(moments), U0=[], n_U0=[], U1_data=[], n_U1=[])
    for h in hours:
        no = [u[i, t + h] for i, t in moments if nee[i, t:t + h].sum() == 0]
        yes = [u[i, t + h] for i, t in moments if nee[i, t] > 0]
        out["U0"].append(float(np.mean(no)) if no else None)
        out["n_U0"].append(len(no))
        out["U1_data"].append(float(np.mean(yes)) if yes else None)
        out["n_U1"].append(len(yes))
    out["U1_formula"] = [math.exp(-h / DECAY_TAU_H) for h in hours]
    return out


def predicted_bias(g, u_stats):
    """bias_obs_h = -G * (exp(-h/4) - U0_h)."""
    return [-g * (u1 - (u0 or 0.0)) for u1, u0 in zip(u_stats["U1_formula"], u_stats["U0"])]
