"""
mimic_data.py - Stage 7: hourly tensors from real MIMIC-IV ICU tables, in the
same format as data.py, so the Stage 3 model and Stage 4 training loop run unchanged.

Tables used (from the MIMIC-IV icu/ module, .csv or .csv.gz; several files
matching e.g. chartevents*.csv are read together):
    icustays, chartevents, inputevents, procedureevents

Per ICU stay, hours 0..47 after ICU admission (hour k = [intime+k h, intime+(k+1) h)):
  Vitals x (5), hourly mean of valid values, NaN when not measured that hour:
    0 HR    220045                  bpm
    1 RR    220210                  breaths/min
    2 SpO2  220277                  %
    3 MAP   220052 arterial, else 220181 non-invasive cuff  (mmHg)
    4 Temp  223762 Celsius, else 223761 Fahrenheit converted to Celsius
  Actions a (3):
    0 vasopressor, norepinephrine-equivalent dose (mcg/kg/min), time-weighted
      average over the hour from rate, starttime, endtime. Equivalence factors:
      norepinephrine 1, epinephrine 1, phenylephrine 0.1, dopamine 0.01,
      vasopressin 2.5 per unit/min
    1 fluid bolus (L) started in that hour: NaCl 0.9% 225158 and Lactated
      Ringer's 225828, only rows in the "03-IV Fluid Bolus" order category
      (the same items are also used as continuous carriers for drugs)
    2 invasive ventilation 225792 (procedureevents): fraction of the hour on it
  Implausible doses are capped (NEE at 2 mcg/kg/min, bolus at 3 L per hour) and the
  number of capped hours is saved with the data.
  valid (T,): True for hours inside the ICU stay; stays shorter than 48 h are padded.

Output: one .npz with raw values; load_splits_mimic() z-scores with train
statistics, adds mask/delta, and splits by PATIENT (subject_id).

Run:  python src/mimic_data.py --root <folder with the icu tables> --out outputs/mimic_hourly.npz
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ICUSequences, hours_since_observed, split_patients   # noqa: E402

T_HOURS = 48
MIN_HOURS = 6                 # drop stays shorter than this
DELTA_CAP = 24.0              # hours-since-observed is capped here
VITALS = ["HR", "RR", "SpO2", "MAP", "Temp"]
ACTIONS = ["NEE mcg/kg/min", "fluid bolus L", "ventilation fraction"]
VALID_RANGE = {"HR": (20, 250), "RR": (1, 80), "SpO2": (50, 100), "MAP": (20, 200), "Temp": (30, 43)}
VITAL_ITEMS = {220045: "HR", 220210: "RR", 220277: "SpO2", 220181: "MAP", 220052: "MAP",
               223761: "Temp", 223762: "Temp"}
PREFERRED = {220052, 223762}  # arterial MAP and Celsius temperature win over the alternative
NEE_FACTOR = {221906: 1.0, 221289: 1.0, 229617: 1.0, 221749: 0.1, 221662: 0.01, 222315: 2.5}
FLUID_ITEMS = (225158, 225828)
VENT_ITEM = 225792
# Hours above these are capped: higher values are almost always unit or charting errors.
NEE_CLIP = 2.0                # mcg/kg/min norepinephrine equivalent
FLUID_CLIP_L = 3.0            # litres of bolus started in one hour


def find_files(root, table):
    """All files under root named like table*.csv or table*.csv.gz."""
    files = sorted(glob.glob(os.path.join(root, "**", f"{table}*.csv*"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no {table}*.csv(.gz) under {root}")
    return [f.replace("\\", "/") for f in files]


def src(files, columns):
    """DuckDB table expression reading only the needed columns, with fixed types."""
    types = {c: t for c, t in columns.items()}
    return (f"(SELECT {', '.join(columns)} FROM read_csv({files}, union_by_name=true, "
            f"types={types}, header=true))")


def extract(root, max_stays=0):
    con = duckdb.connect()
    stay_cols = dict(subject_id="BIGINT", stay_id="BIGINT", intime="TIMESTAMP", outtime="TIMESTAMP")
    limit = f"LIMIT {max_stays}" if max_stays else ""
    con.execute(f"""
        CREATE TABLE stays AS
        SELECT *, row_number() OVER (ORDER BY stay_id) - 1 AS row FROM (
            SELECT subject_id, stay_id, intime,
                   LEAST({T_HOURS}, CEIL(date_diff('second', intime, outtime) / 3600))::INT AS n_hours
            FROM {src(find_files(root, 'icustays'), stay_cols)}
            WHERE date_diff('second', intime, outtime) >= {MIN_HOURS} * 3600
            ORDER BY stay_id {limit})""")
    stays = con.execute("SELECT subject_id, stay_id, n_hours FROM stays ORDER BY row").fetchnumpy()
    N = len(stays["stay_id"])
    print(f"stays with >= {MIN_HOURS} h: {N}")

    # ---- vitals: hourly mean per (stay, hour, itemid)
    chart_cols = dict(stay_id="BIGINT", charttime="TIMESTAMP", itemid="BIGINT", valuenum="DOUBLE")
    # Implausible single readings are dropped BEFORE averaging, so one bad value
    # does not spoil the whole hour.
    in_range = " OR ".join(
        f"(itemid IN ({', '.join(str(i) for i, n in VITAL_ITEMS.items() if n == name)}) "
        f"AND value BETWEEN {lo} AND {hi})" for name, (lo, hi) in VALID_RANGE.items())
    v = con.execute(f"""
        WITH ev AS (
            SELECT s.row, FLOOR(date_diff('second', s.intime, c.charttime) / 3600)::INT AS hour,
                   s.n_hours, c.itemid,
                   CASE WHEN c.itemid = 223761 THEN (c.valuenum - 32) * 5 / 9 ELSE c.valuenum END AS value
            FROM {src(find_files(root, 'chartevents'), chart_cols)} c JOIN stays s USING (stay_id)
            WHERE c.itemid IN ({', '.join(map(str, VITAL_ITEMS))}) AND c.valuenum IS NOT NULL)
        SELECT row, hour, itemid, avg(value) AS value FROM ev
        WHERE hour >= 0 AND hour < n_hours AND ({in_range}) GROUP BY ALL""").fetchnumpy()

    x = np.full((N, T_HOURS, len(VITALS)), np.nan)
    col = np.array([VITALS.index(VITAL_ITEMS[int(i)]) for i in v["itemid"]], dtype=int)
    preferred = np.isin(v["itemid"], list(PREFERRED))
    for keep in (~preferred, preferred):                  # preferred items are written last and win
        x[v["row"][keep], v["hour"][keep], col[keep]] = v["value"][keep]

    # ---- actions
    a = np.zeros((N, T_HOURS, len(ACTIONS)))
    in_cols = dict(stay_id="BIGINT", starttime="TIMESTAMP", endtime="TIMESTAMP", itemid="BIGINT",
                   amount="DOUBLE", amountuom="VARCHAR", rate="DOUBLE", rateuom="VARCHAR",
                   patientweight="DOUBLE", ordercategoryname="VARCHAR", statusdescription="VARCHAR")
    inputs = src(find_files(root, "inputevents"), in_cols)
    nee_case = " ".join(f"WHEN {k} THEN {f}" for k, f in NEE_FACTOR.items())
    vaso = con.execute(f"""
        WITH inf AS (
            SELECT s.row, s.n_hours,
                   date_diff('second', s.intime, i.starttime) / 3600.0 AS t0,
                   date_diff('second', s.intime, i.endtime) / 3600.0 AS t1,
                   i.rate * (CASE i.itemid {nee_case} END)
                          * (CASE i.rateuom WHEN 'mcg/min' THEN 1.0 / NULLIF(i.patientweight, 0)
                                            WHEN 'units/hour' THEN 1.0 / 60 ELSE 1.0 END) AS nee
            FROM {inputs} i JOIN stays s USING (stay_id)
            WHERE i.itemid IN ({', '.join(map(str, NEE_FACTOR))}) AND i.rate > 0
              AND coalesce(i.statusdescription, '') <> 'Rewritten'),
        hours AS (
            SELECT row, nee, t0, t1,
                   unnest(range(GREATEST(0, FLOOR(t0))::INT, LEAST(n_hours, CEIL(t1))::INT)) AS hour
            FROM inf WHERE nee IS NOT NULL AND t1 > t0)
        SELECT row, hour, sum(nee * (LEAST(t1, hour + 1) - GREATEST(t0, hour))) AS dose
        FROM hours GROUP BY ALL""").fetchnumpy()
    a[vaso["row"], vaso["hour"], 0] = vaso["dose"]

    fluid = con.execute(f"""
        SELECT row, hour, sum(amount) / 1000 AS liters FROM (
            SELECT s.row, s.n_hours, i.amount,
                   FLOOR(date_diff('second', s.intime, i.starttime) / 3600)::INT AS hour
            FROM {inputs} i JOIN stays s USING (stay_id)
            WHERE i.itemid IN ({', '.join(map(str, FLUID_ITEMS))}) AND i.amountuom = 'mL'
              AND i.ordercategoryname = '03-IV Fluid Bolus'
              AND coalesce(i.statusdescription, '') <> 'Rewritten')
        WHERE hour >= 0 AND hour < n_hours GROUP BY ALL""").fetchnumpy()
    a[fluid["row"], fluid["hour"], 1] = fluid["liters"]

    proc_cols = dict(stay_id="BIGINT", starttime="TIMESTAMP", endtime="TIMESTAMP", itemid="BIGINT")
    vent = con.execute(f"""
        WITH ep AS (
            SELECT s.row, s.n_hours,
                   date_diff('second', s.intime, p.starttime) / 3600.0 AS t0,
                   date_diff('second', s.intime, p.endtime) / 3600.0 AS t1
            FROM {src(find_files(root, 'procedureevents'), proc_cols)} p JOIN stays s USING (stay_id)
            WHERE p.itemid = {VENT_ITEM}),
        hours AS (
            SELECT row, t0, t1,
                   unnest(range(GREATEST(0, FLOOR(t0))::INT, LEAST(n_hours, CEIL(t1))::INT)) AS hour
            FROM ep WHERE t1 > t0)
        SELECT row, hour, LEAST(1.0, sum(LEAST(t1, hour + 1) - GREATEST(t0, hour))) AS frac
        FROM hours GROUP BY ALL""").fetchnumpy()
    a[vent["row"], vent["hour"], 2] = vent["frac"]

    # Cap implausible doses and count how many hours were affected.
    clipped_nee = int((a[..., 0] > NEE_CLIP).sum())
    clipped_fluid = int((a[..., 1] > FLUID_CLIP_L).sum())
    a[..., 0] = np.minimum(a[..., 0], NEE_CLIP)
    a[..., 1] = np.minimum(a[..., 1], FLUID_CLIP_L)
    print(f"clipped hours: NEE > {NEE_CLIP} mcg/kg/min: {clipped_nee}, "
          f"fluid bolus > {FLUID_CLIP_L} L: {clipped_fluid}")

    valid = np.arange(T_HOURS)[None, :] < stays["n_hours"][:, None]          # (N, T)
    return dict(x=x, a=a, valid=valid, subject_id=stays["subject_id"], stay_id=stays["stay_id"],
                clipped_nee_hours=clipped_nee, clipped_fluid_hours=clipped_fluid)


def load_splits_mimic(path):
    """Datasets (train/val/test, split by patient) and stats, same format as data.load_splits."""
    raw = np.load(path)
    x_raw, a, valid = raw["x"], raw["a"], raw["valid"]                      # (N,T,V), (N,T,A), (N,T)
    mask = ~np.isnan(x_raw) & valid[..., None]                              # (N, T, V)
    subjects = np.unique(raw["subject_id"])
    sub_split = split_patients(len(subjects))                               # indices into subjects
    row_split = {k: np.where(np.isin(raw["subject_id"], subjects[v]))[0] for k, v in sub_split.items()}

    train_x = np.where(mask, x_raw, np.nan)[row_split["train"]]
    stats = dict(mean=np.nanmean(train_x, axis=(0, 1)), std=np.nanstd(train_x, axis=(0, 1)))
    x = np.where(mask, (x_raw - stats["mean"]) / stats["std"], 0.0)
    delta = np.minimum(hours_since_observed(mask), DELTA_CAP)
    datasets = {k: ICUSequences(x[i], mask[i], delta[i], a[i], i, valid=valid[i])
                for k, i in row_split.items()}
    return datasets, stats


def make_loaders_mimic(path, batch_size=64):
    datasets, stats = load_splits_mimic(path)
    return {k: DataLoader(ds, batch_size=batch_size, shuffle=(k == "train"))
            for k, ds in datasets.items()}, stats


def clip_counts(path):
    """Hours capped during extraction; empty for files made before clipping existed."""
    raw = np.load(path)
    return {k: int(raw[k]) for k in ("clipped_nee_hours", "clipped_fluid_hours") if k in raw}


def describe(path):
    print("clipped hours:", clip_counts(path) or "none recorded")
    datasets, stats = load_splits_mimic(path)
    print("splits (stays): " + ", ".join(f"{k} {len(v)}" for k, v in datasets.items()))
    tr = datasets["train"]
    valid = tr.valid.bool()                                                 # (N, T)
    print(f"valid hours per stay: median {tr.valid.sum(1).median():.0f} of {T_HOURS}")
    for j, name in enumerate(VITALS):
        print(f"  {name:<5} train mean {stats['mean'][j]:7.1f} std {stats['std'][j]:6.1f} | "
              f"observed in {tr.mask[..., j][valid].mean():.1%} of valid hours")
    for j, name in enumerate(ACTIONS):
        act = tr.a[..., j][valid]
        print(f"  {name:<22} nonzero in {(act > 0).float().mean():.1%} of valid hours, "
              f"max {act.max():.2f}")
    batch = next(iter(DataLoader(tr, batch_size=16)))
    print("one batch: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in batch.items()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="folder that contains the MIMIC-IV icu tables")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "outputs" / "mimic_hourly.npz"))
    p.add_argument("--max-stays", type=int, default=0, help="0 = all stays")
    args = p.parse_args()
    data = extract(args.root, args.max_stays)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f"saved {args.out}: " + ", ".join(f"{k} {np.shape(v)}" for k, v in data.items()))
    describe(args.out)
