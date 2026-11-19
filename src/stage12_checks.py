"""
stage12_checks.py - checks that explain what the Stage 12 models learned (aggregates only).

event_study     around every real vasopressor start (all stays): mean of (MAP' - MAP) at
                d = -6 .. +6 hours, over hours where MAP was measured, and the specified curve
                -G * mean k(d) + K * mean e at the same hours.
raw_observed    on the fixed Part B moments: measured MAP' at t + h for moments where NEE
                started at t vs did not, and the mean hidden u in each group at those same
                measured hours; the difference-in-differences against G = 0 should equal
                -G * (u_started - u_not).
groups          per moment: does the history already show the confounder?
                  ramp_in_history  u_t > 0 at the decision hour (a start in t .. t+3 under the
                                   Stage 12 ramp), known only from the injected u
                  observed_fall    measured MAP'_t at least FALL_MMHG below the mean of the
                                   measured MAP' in t-3 .. t-1, what the model can actually see
group_effects   learned effect (semi model minus real-data model, paired per moment) per group.
map_gap         read-only: hours of the extracted stays with no MAP (220052 / 220181) but both
                cuff systolic (220179) and diastolic (220180) present.
"""
import duckdb
import numpy as np
import torch

import confounded as C
import mimic_data
from mimic_data import MIN_HOURS, T_HOURS, find_files, src

D_RANGE = list(range(-6, 7))
FALL_MMHG = 5.0
OBS_HOURS = (0, 1, 3, 6)


def event_study(real, g, kind, k=C.K_MMHG):
    """real: extracted dataset (original units, NaN = missing). Returns mean (MAP' - MAP) per d."""
    inj = C.inject_confounded(real, k=k, g=float(g), kind=kind)
    diff = inj["x"][..., C.MAP] - real["x"][..., C.MAP]                       # (N, T), NaN where missing
    nee = real["a"][..., C.NEE]
    e = C.SS.lagged_dose(nee, C.SS.TAU_H)
    rows, cols = np.nonzero(C.vaso_starts(nee))
    T = diff.shape[1]
    out = dict(d=D_RANGE, g=g, kind=kind, n_starts=int(len(rows)), mean_diff=[], n=[], spec=[], mean_ke=[])
    for d in D_RANGE:
        c = cols + d
        ok = (c >= 0) & (c < T)
        r, c = rows[ok], c[ok]
        vals = diff[r, c]
        m = ~np.isnan(vals)
        out["n"].append(int(m.sum()))
        out["mean_diff"].append(float(vals[m].mean()) if m.any() else None)
        ke = float((k * e[r, c][m]).mean()) if m.any() else 0.0
        out["mean_ke"].append(ke)
        out["spec"].append(ke - g * float(C.kernel(d, kind)))
    del inj
    return out


def _values(ds, stats):
    j = C.MAP
    value = ds.x[..., j].numpy() * stats["std"][j] + stats["mean"][j]
    return value, ds.mask[..., j].numpy() > 0


def raw_observed(ds, stats, moments, kind):
    """Measured MAP' at t + h, started at t vs not, with the mean u at the same hours."""
    value, measured = _values(ds, stats)
    nee = ds.a[..., C.NEE].numpy()
    u = C.hidden_u(nee, kind)
    out = {}
    for h in OBS_HOURS:
        res = {}
        for name, cond in (("started", lambda i, t: nee[i, t] > 0), ("not_started", lambda i, t: nee[i, t] <= 0)):
            sel = [(i, t + h) for i, t in moments if cond(i, t) and measured[i, t + h]]
            res[name] = dict(n=len(sel), map=float(np.mean([value[a, b] for a, b in sel])) if sel else None,
                             u=float(np.mean([u[a, b] for a, b in sel])) if sel else None)
        out[str(h)] = res
    return out


def groups(ds, stats, moments):
    """Per moment: ramp_in_history (true, from u) and observed_fall (from the measured MAP')."""
    value, measured = _values(ds, stats)
    u = C.hidden_u(ds.a[..., C.NEE].numpy(), "ramp")
    g = {}
    for i, t in moments:
        prev = [value[i, s] for s in range(max(0, t - 3), t) if measured[i, s]]
        fall = bool(measured[i, t] and prev and value[i, t] - np.mean(prev) <= -FALL_MMHG)
        g[(i, t)] = dict(ramp_in_history=bool(u[i, t] > 0), observed_fall=fall)
    return g


def group_effects(per_semi, per_real, grp, true):
    """Learned effect per group (paired per moment) for both groupings."""
    out = {}
    for key in ("ramp_in_history", "observed_fall"):
        for flag in (True, False):
            keys = [m for m in per_semi if m in per_real and m in grp and grp[m][key] == flag]
            name = f"{key}={'yes' if flag else 'no'}"
            if not keys:
                out[name] = dict(n=0)
                continue
            d = torch.stack([per_semi[m] - per_real[m] for m in keys])
            mean = d.mean(0).tolist()
            out[name] = dict(n=len(keys), learned_mean=mean, learned_p10=d.quantile(0.1, dim=0).tolist(),
                             learned_p90=d.quantile(0.9, dim=0).tolist(),
                             bias_mean=[m - t for m, t in zip(mean, true)])
    return out


def group_overlap(grp):
    both = sum(v["ramp_in_history"] and v["observed_fall"] for v in grp.values())
    return dict(n=len(grp), ramp_in_history=sum(v["ramp_in_history"] for v in grp.values()),
                observed_fall=sum(v["observed_fall"] for v in grp.values()), both=both)


def map_gap(root):
    """Hours of the extracted stays (>= MIN_HOURS, first T_HOURS hours) with no MAP reading
    (220052 / 220181) but both 220179 and 220180. Counted twice: any non-null reading, and
    only readings inside the extraction's plausible range (what the extraction keeps)."""
    con = duckdb.connect()
    stay_cols = dict(stay_id="BIGINT", intime="TIMESTAMP", outtime="TIMESTAMP")
    chart_cols = dict(stay_id="BIGINT", charttime="TIMESTAMP", itemid="BIGINT", valuenum="DOUBLE")
    lo, hi = mimic_data.VALID_RANGE["MAP"]
    q = f"""
        WITH stays AS (
            SELECT stay_id, intime,
                   LEAST({T_HOURS}, CEIL(date_diff('second', intime, outtime) / 3600))::INT AS n_hours
            FROM {src(find_files(root, 'icustays'), stay_cols)}
            WHERE date_diff('second', intime, outtime) >= {MIN_HOURS} * 3600),
        ev AS (
            SELECT c.stay_id, FLOOR(date_diff('second', s.intime, c.charttime) / 3600)::INT AS hour,
                   c.itemid, c.valuenum AS v, s.n_hours
            FROM {src(find_files(root, 'chartevents'), chart_cols)} c JOIN stays s USING (stay_id)
            WHERE c.itemid IN (220052, 220181, 220179, 220180) AND c.valuenum IS NOT NULL),
        h AS (
            SELECT stay_id, hour,
                   max(itemid IN (220052, 220181))::INT AS map_any,
                   max(itemid IN (220052, 220181) AND v BETWEEN {lo} AND {hi})::INT AS map_plausible,
                   max(itemid = 220179)::INT AS sys, max(itemid = 220180)::INT AS dia
            FROM ev WHERE hour >= 0 AND hour < n_hours GROUP BY stay_id, hour)
        SELECT (SELECT sum(n_hours) FROM stays) AS valid_hours,
               sum(map_any) AS hours_map_any,
               sum(map_plausible) AS hours_map_plausible,
               sum(CASE WHEN map_any = 0 AND sys = 1 AND dia = 1 THEN 1 ELSE 0 END) AS gap_any,
               sum(CASE WHEN map_plausible = 0 AND sys = 1 AND dia = 1 THEN 1 ELSE 0 END) AS gap_plausible
        FROM h"""
    valid, m_any, m_pl, gap_any, gap_pl = con.execute(q).fetchone()
    valid = int(valid)
    return dict(valid_hours=valid, hours_with_map_any=int(m_any or 0), hours_with_map_plausible=int(m_pl or 0),
                gap_hours_any=int(gap_any or 0), gap_share_any=(gap_any or 0) / valid,
                gap_hours_plausible=int(gap_pl or 0), gap_share_plausible=(gap_pl or 0) / valid)


D_FIT = list(range(-3, 4))


@torch.no_grad()
def start_fit(model, ds, stats, device="cpu", batch_size=256):
    """One-step prior fit around real vasopressor starts in `ds` (one split).
    For d = -3 .. +3, hour j = s + d with MAP measured at j and j-1:
        measured change  = MAP'_j - MAP'_{j-1}
        predicted change = (prior one-step prediction of MAP'_j) - MAP'_{j-1}
    The prediction for hour j uses the history up to j-1 and the action a_{j-1}; so d = +1 is
    the hour after the start, predicted with the start itself as the current action."""
    j_map = C.MAP
    std = float(stats["std"][j_map])
    nee = ds.a[..., C.NEE].numpy()
    starts = C.vaso_starts(nee)
    sums = {d: dict(n=0, measured=0.0, predicted=0.0, abs_err=0.0) for d in D_FIT}
    model.eval()
    for lo in range(0, len(ds), batch_size):
        sl = slice(lo, lo + batch_size)
        batch = {k: getattr(ds, k)[sl].to(device) for k in ("x", "mask", "delta", "a", "valid", "idx")}
        out = model(batch)
        point, _ = model.decoder(torch.cat([out["prior_mu"], out["h"][:, :-1]], -1))   # (B, T-1, V)
        x = batch["x"][..., j_map].cpu().numpy()
        m = (batch["mask"][..., j_map] > 0).cpu().numpy() & (batch["valid"] > 0).cpu().numpy()
        p = point[..., j_map].cpu().numpy()                                               # p[:, j-1] predicts hour j
        T = x.shape[1]
        rows, cols = np.nonzero(starts[sl])
        for d in D_FIT:
            j = cols + d
            ok = (j >= 1) & (j < T)
            r, jj = rows[ok], j[ok]
            ok2 = m[r, jj] & m[r, jj - 1]
            r, jj = r[ok2], jj[ok2]
            meas = (x[r, jj] - x[r, jj - 1]) * std
            pred = (p[r, jj - 1] - x[r, jj - 1]) * std
            s = sums[d]
            s["n"] += len(r)
            s["measured"] += float(meas.sum())
            s["predicted"] += float(pred.sum())
            s["abs_err"] += float(np.abs(pred - meas).sum())
    return dict(d=D_FIT, n=[sums[d]["n"] for d in D_FIT],
                measured_change=[sums[d]["measured"] / max(sums[d]["n"], 1) for d in D_FIT],
                predicted_change=[sums[d]["predicted"] / max(sums[d]["n"], 1) for d in D_FIT],
                mae=[sums[d]["abs_err"] / max(sums[d]["n"], 1) for d in D_FIT])
