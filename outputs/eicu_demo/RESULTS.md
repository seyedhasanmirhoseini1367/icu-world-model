# eICU demo check: results

Pre-registration: `PREREGISTRATION.md` (committed before running). Numbers: `eicu_demo_metrics.json`.

**Data:** the open eICU-CRD demo v2.0.1 (PhysioNet). 2,315 stays of 6 h or more, from 1,771 patients.
Extracted by `src/eicu_data.py` and scored by `src/eval_eicu.py`.
The frozen MIMIC model `real_actnow_s0.pt` was used with no retraining; seed 1 (`real_actnow_s1.pt`) is shown as a check.

| MAP, one hour ahead | model MAE | carry-forward MAE | ratio | 10-90% band covers |
|---|---|---|---|---|
| MIMIC-IV test (Stage 7) | 7.01 | 8.33 | 0.84 | 79% |
| eICU demo, seed 0 | 6.81 | 7.62 | 0.89 | 78.0% |
| eICU demo, seed 1 | 6.83 | 7.62 | 0.90 | 79.1% |

- **P1 holds:** the model beats carry-forward on MAP (6.81 vs 7.62 mmHg).
- **P2 holds:** the band covers 78% of observed values, inside the pre-registered 70-90%.
- **Verdict: transfers.**
- **P3 (descriptive):** the advantage over carry-forward is a bit smaller than on MIMIC (ratio 0.89 vs 0.84).
- **Sensitivity:** z-scoring with eICU's own mean/std instead of MIMIC's changes MAP MAE by 0.01.

**Other vitals:** the model is about equal to carry-forward on RR and SpO2 and slightly better on Temp,
but slightly **worse on HR** (3.64 vs 3.57 bpm). Its bands cover 77-86% of values.

**Why eICU errors are lower than MIMIC's for both methods:**
eICU vitals are mostly 5-minute monitor medians, averaged here over each hour. That makes hours
smoother than MIMIC's nurse-charted values, so "same as last hour" is already a strong forecast.

**Caveats:**

- **Vasopressor doses are approximate.** eICU charts a rate at a point in time, which is held until
  the next charting (at most 4 h). Rows charted only in ml/hr (3,112 of 6,568 used) were converted
  with an assumed standard bag concentration. 847 rows had no usable unit and were dropped.
- **Fluid boluses are nearly invisible in eICU:** only 42 stays have one.
- **The MIMIC normalisation values** come from the training log, which prints them to 0.1.
- **Treatment effects were not tested:** too few vasopressor stays (188), and the same hidden-confounder problem as MIMIC.
