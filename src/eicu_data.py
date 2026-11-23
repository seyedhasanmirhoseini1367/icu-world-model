"""
eicu_data.py - hourly tensors from eICU-CRD (full or demo), in the same format as mimic_data.py,
so a model trained on MIMIC-IV can be run on eICU without retraining. New file; mimic_data.py is
only imported.

Times in eICU are minute offsets from ICU unit admission. Hour k = [60k, 60k + 60) minutes, k = 0..47.
Stays shorter than MIN_HOURS are dropped; shorter than 48 h are padded (valid = False).

Vitals x (5), hourly mean of plausible readings (same ranges as MIMIC), NaN when absent:
    HR    vitalPeriodic.heartrate, nurseCharting "Heart Rate"
    RR    vitalPeriodic.respiration, nurseCharting "Respiratory Rate"
    SpO2  vitalPeriodic.sao2, nurseCharting "O2 Saturation"
    MAP   arterial (vitalPeriodic.systemicmean, nurseCharting "Invasive BP Mean") wins over
          cuff (vitalAperiodic.noninvasivemean, nurseCharting "Non-Invasive BP Mean")
    Temp  vitalPeriodic.temperature, nurseCharting "Temperature (C)"
Actions a (3):
    0 vasopressor, norepinephrine-equivalent mcg/kg/min, same factors as MIMIC. eICU charts a rate at
      a point in time, so each rate is held until the next charting of the same drug (at most 4 h;
      the last one for 1 h). Units come from the drug name: mcg/kg/min, mcg/min (divided by weight),
      units/min or units/hr (vasopressin). At an offset with only an ml/hr row, a standard
      concentration is assumed (STD_CONC). Rows with no usable unit are dropped and counted.
    1 fluid bolus (L): intakeOutput rows whose label names a saline or Ringer's bolus.
    2 invasive ventilation, fraction of the hour: respiratoryCharting rows with ventilator
      settings (VENT_LABELS); consecutive chartings up to 8 h apart are joined into one episode,
      a lone charting counts as 1 h.
Same dose caps as MIMIC (NEE 2 mcg/kg/min, bolus 3 L per hour).

Run:  python src/eicu_data.py --root <folder with the eICU csv.gz files> --out eicu_hourly.npz
"""
import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mimic_data import (ACTIONS, FLUID_CLIP_L, MIN_HOURS, NEE_CLIP, T_HOURS, VALID_RANGE,  # noqa: E402
                        VITALS)

NEE = dict(norepinephrine=1.0, epinephrine=1.0, phenylephrine=0.1, dopamine=0.01, vasopressin=2.5)
STD_CONC = dict(norepinephrine=16.0, epinephrine=16.0, phenylephrine=40.0, dopamine=1600.0,
                vasopressin=0.2)  # mcg/ml (vasopressin: units/ml), common premixed bags
VENT_LABELS = ("Tidal Volume (set)", "Vent Rate", "TV/kg IBW", "Mean Airway Pressure",
               "Plateau Pressure", "Peak Insp. Pressure")
HOLD_MAX_MIN, LAST_HOLD_MIN, VENT_GAP_MIN = 240, 60, 480


def extract(root):
    root = Path(root)

    def t(name):
        return f"read_csv('{(root / f'{name}.csv.gz').as_posix()}', quote='\"', escape='\"', header=true)"

    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE stays AS
        SELECT *, row_number() OVER (ORDER BY sid) - 1 AS row FROM (
            SELECT patientunitstayid AS sid, uniquepid,
                   LEAST({T_HOURS}, CEIL(unitdischargeoffset / 60.0))::INT AS n_hours,
                   coalesce(try_cast(admissionweight AS DOUBLE), try_cast(dischargeweight AS DOUBLE)) AS w
            FROM {t('patient')} WHERE unitdischargeoffset >= {MIN_HOURS} * 60)""")
    stays = con.execute("SELECT sid, uniquepid, n_hours FROM stays ORDER BY row").fetchnumpy()
    N = len(stays["sid"])
    print(f"stays with >= {MIN_HOURS} h: {N}")

    # ---- vitals: long table (sid, minute, vital, arterial, value)
    nurse = {"Heart Rate": ("HR", 0), "Respiratory Rate": ("RR", 0), "O2 Saturation": ("SpO2", 0),
             "Invasive BP Mean": ("MAP", 1), "Non-Invasive BP Mean": ("MAP", 0), "Temperature (C)": ("Temp", 0)}
    nurse_case = " ".join(f"WHEN '{k}' THEN '{v}'" for k, (v, _) in nurse.items())
    con.execute(f"""
        CREATE TABLE vit AS
        SELECT patientunitstayid AS sid, observationoffset AS m, v.name AS vital, v.art, v.value FROM {t('vitalPeriodic')},
            LATERAL (VALUES ('HR', 0, heartrate), ('RR', 0, respiration), ('SpO2', 0, sao2),
                            ('MAP', 1, systemicmean), ('Temp', 0, temperature)) v(name, art, value)
        UNION ALL
        SELECT patientunitstayid, observationoffset, 'MAP', 0, noninvasivemean FROM {t('vitalAperiodic')}
        UNION ALL
        SELECT patientunitstayid, nursingchartoffset,
               CASE nursingchartcelltypevalname {nurse_case} END,
               (nursingchartcelltypevalname = 'Invasive BP Mean')::INT,
               try_cast(nursingchartvalue AS DOUBLE)
        FROM {t('nurseCharting')} WHERE nursingchartcelltypevalname IN ({', '.join(f"'{k}'" for k in nurse)})""")
    in_range = " OR ".join(f"(vital = '{n}' AND value BETWEEN {lo} AND {hi})" for n, (lo, hi) in VALID_RANGE.items())
    v = con.execute(f"""
        SELECT s.row, FLOOR(m / 60)::INT AS hour, vital, art, avg(value) AS value
        FROM vit JOIN stays s USING (sid)
        WHERE value IS NOT NULL AND m >= 0 AND FLOOR(m / 60) < s.n_hours AND ({in_range})
        GROUP BY ALL""").fetchnumpy()
    x = np.full((N, T_HOURS, len(VITALS)), np.nan)
    col = np.array([VITALS.index(n) for n in v["vital"]], dtype=int)
    art = v["art"].astype(bool)
    for keep in (~art, art):                              # arterial MAP is written last and wins
        x[v["row"][keep], v["hour"][keep], col[keep]] = v["value"][keep]
    source = dict(MAP_arterial_hours=int(len(np.unique(np.stack([v["row"][art], v["hour"][art]]), axis=1).T))
                  if art.any() else 0)

    # ---- vasopressors
    kind_case = ("CASE WHEN regexp_matches(lower(drugname), 'norepi|levophed') THEN 'norepinephrine' "
                 "WHEN regexp_matches(lower(drugname), 'epinep|adrenalin') THEN 'epinephrine' "
                 "WHEN regexp_matches(lower(drugname), 'phenyleph|neo-syn') THEN 'phenylephrine' "
                 "WHEN regexp_matches(lower(drugname), 'dopamin') THEN 'dopamine' "
                 "WHEN regexp_matches(lower(drugname), 'vasopres') THEN 'vasopressin' END")
    f_case = " ".join(f"WHEN '{k}' THEN {f}" for k, f in NEE.items())
    c_case = " ".join(f"WHEN '{k}' THEN {c}" for k, c in STD_CONC.items())
    con.execute(f"""
        CREATE TABLE inf AS
        SELECT i.patientunitstayid AS sid, i.infusionoffset AS m, {kind_case} AS kind,
               lower(regexp_extract(drugname, '\\(([^)]*)\\)\\s*$', 1)) AS unit,
               try_cast(i.drugrate AS DOUBLE) AS rate,
               coalesce(s.w, try_cast(i.patientweight AS DOUBLE), 80.0) AS w
        FROM {t('infusiondrug')} i JOIN stays s ON s.sid = i.patientunitstayid""")
    con.execute(f"""
        CREATE TABLE inf2 AS
        SELECT *, (CASE kind {f_case} END) * CASE
                  WHEN kind <> 'vasopressin' AND unit = 'mcg/kg/min' THEN rate
                  WHEN kind <> 'vasopressin' AND unit = 'mcg/min' THEN rate / w
                  WHEN kind = 'vasopressin' AND unit = 'units/min' THEN rate
                  WHEN kind = 'vasopressin' AND unit IN ('units/hr', 'units/hour') THEN rate / 60
                  WHEN unit = 'ml/hr' AND kind = 'vasopressin' THEN rate * (CASE kind {c_case} END) / 60
                  WHEN unit = 'ml/hr' THEN rate * (CASE kind {c_case} END) / 60 / w END AS nee,
               unit <> 'ml/hr' AS dose_unit
        FROM inf WHERE kind IS NOT NULL""")
    dropped = con.execute("SELECT count(*), count(*) FILTER (WHERE rate IS NOT NULL) FROM inf2 WHERE nee IS NULL").fetchone()
    by_unit = dict(con.execute("SELECT CASE WHEN dose_unit THEN 'dose unit' ELSE 'ml/hr (assumed conc.)' END, "
                               "count(*) FROM inf2 WHERE nee IS NOT NULL GROUP BY 1").fetchall())
    vaso = con.execute(f"""
        WITH pick AS (      -- one rate per stay, drug and offset; a dose-unit row wins over ml/hr
            SELECT sid, kind, m, arg_max(nee, dose_unit::INT * 1e9 + nee) AS nee
            FROM inf2 WHERE nee IS NOT NULL AND nee >= 0 GROUP BY ALL),
        seg AS (
            SELECT sid, nee, m / 60.0 AS t0,
                   coalesce(LEAST(lead(m) OVER w, m + {HOLD_MAX_MIN}), m + {LAST_HOLD_MIN}) / 60.0 AS t1
            FROM pick WINDOW w AS (PARTITION BY sid, kind ORDER BY m)),
        hours AS (
            SELECT s.row, nee, t0, t1,
                   unnest(range(GREATEST(0, FLOOR(t0))::INT, LEAST(s.n_hours, CEIL(t1))::INT)) AS hour
            FROM seg JOIN stays s USING (sid) WHERE nee > 0 AND t1 > t0)
        SELECT row, hour, sum(nee * (LEAST(t1, hour + 1) - GREATEST(t0, hour))) AS dose FROM hours GROUP BY ALL
        """).fetchnumpy()
    a = np.zeros((N, T_HOURS, len(ACTIONS)))
    a[vaso["row"], vaso["hour"], 0] = vaso["dose"]

    # ---- fluid boluses
    fluid = con.execute(f"""
        SELECT s.row, FLOOR(intakeoutputoffset / 60)::INT AS hour, sum(cellvaluenumeric) / 1000 AS liters
        FROM {t('intakeOutput')} io JOIN stays s ON s.sid = io.patientunitstayid
        WHERE regexp_matches(lower(celllabel), 'bolus')
          AND regexp_matches(lower(celllabel), 'saline|sodium chloride|\\bns\\b|lactated|ringer|\\blr\\b')
          AND NOT regexp_matches(lower(celllabel), 'enteral|feeding|flush')
          AND cellvaluenumeric > 0 AND intakeoutputoffset >= 0
          AND FLOOR(intakeoutputoffset / 60) < s.n_hours
        GROUP BY ALL""").fetchnumpy()
    a[fluid["row"], fluid["hour"], 1] = fluid["liters"]

    # ---- invasive ventilation
    vent = con.execute(f"""
        WITH c AS (
            SELECT DISTINCT patientunitstayid AS sid, respchartoffset AS m FROM {t('respiratoryCharting')}
            WHERE respchartvaluelabel IN ({', '.join(f"'{l}'" for l in VENT_LABELS)})),
        seg AS (
            SELECT sid, m / 60.0 AS t0,
                   (CASE WHEN lead(m) OVER w - m <= {VENT_GAP_MIN} THEN lead(m) OVER w ELSE m + 60 END) / 60.0 AS t1
            FROM c WINDOW w AS (PARTITION BY sid ORDER BY m)),
        hours AS (
            SELECT s.row, t0, t1,
                   unnest(range(GREATEST(0, FLOOR(t0))::INT, LEAST(s.n_hours, CEIL(t1))::INT)) AS hour
            FROM seg JOIN stays s USING (sid) WHERE t1 > t0)
        SELECT row, hour, LEAST(1.0, sum(LEAST(t1, hour + 1) - GREATEST(t0, hour))) AS frac FROM hours GROUP BY ALL
        """).fetchnumpy()
    a[vent["row"], vent["hour"], 2] = vent["frac"]

    clipped_nee = int((a[..., 0] > NEE_CLIP).sum())
    clipped_fluid = int((a[..., 1] > FLUID_CLIP_L).sum())
    a[..., 0] = np.minimum(a[..., 0], NEE_CLIP)
    a[..., 1] = np.minimum(a[..., 1], FLUID_CLIP_L)
    valid = np.arange(T_HOURS)[None, :] < stays["n_hours"][:, None]
    subject = np.unique(stays["uniquepid"], return_inverse=True)[1]        # integer patient code
    info = dict(n_stays=N, n_patients=int(subject.max() + 1), clipped_nee_hours=clipped_nee,
                clipped_fluid_hours=clipped_fluid, vaso_rows_used=by_unit,
                vaso_rows_dropped_no_unit=int(dropped[0]), **source)
    print(info)
    return dict(x=x, a=a, valid=valid, subject_id=subject, stay_id=stays["sid"],
                clipped_nee_hours=clipped_nee, clipped_fluid_hours=clipped_fluid), info


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    data, _ = extract(args.root)
    np.savez_compressed(args.out, **data)
    print(f"saved {args.out}")
