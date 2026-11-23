# eICU demo check: pre-registration

Written and committed before any model was run on eICU data.

**What:** the frozen Stage 11 MIMIC model `real_actnow_s0.pt` (release v1-vae-final) is run, with no
retraining, on every usable stay of the open eICU-CRD demo (about 2,500 ICU stays from US hospitals
outside Boston). This is an external check: different hospitals, different charting
(5-minute monitor medians instead of hourly nurse charting, cuff MAP far more common than arterial).

**Inputs:** the same 5 vitals, 3 actions, 48 hours, dose caps and minimum stay (6 h) as Stage 7.
Vitals are z-scored with the MIMIC training mean and std (the model only knows that scale);
a second run with eICU's own mean and std is reported as a sensitivity check.

**Scored:** one-step forecasts of every observed vital, exactly as in `src/evaluate_mimic.py`.

**Predictions (MAP, the main vital):**

- P1: the model's MAP MAE is lower than carry-forward ("same as last hour").
- P2: the 10-90% band covers between 70% and 90% of observed MAP values.
- P3 (descriptive, no verdict): the ratio model MAE / carry-forward MAE, compared with MIMIC's 7.01 / 8.33 = 0.84.

**Verdict:** "transfers" if P1 and P2 both hold, "partly" if exactly one does, "does not transfer" if neither does.

**Not tested:** treatment effects. The demo is too small (a few hundred vasopressor stays) and has
the same hidden-confounder problem as MIMIC.
