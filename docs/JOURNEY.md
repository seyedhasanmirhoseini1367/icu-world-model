# The journey: building an ICU world model, stage by stage

This is the short map of the project. [LEARNING.md](../LEARNING.md) has the full explanations,
numbers and figures for every stage; this file shows how the stages connect, what each one
asked, and what we concluded. Numbers are in mmHg of mean arterial pressure (MAP) unless
stated otherwise.

## Overview

```mermaid
flowchart TD
    subgraph SIM["Simulated ICU (known truth)"]
        S1["1 Simulator<br/>hidden state, clinician policy"] --> S2["2 Tensors<br/>x, mask, delta, a"]
        S2 --> S3["3 Model<br/>transformer + prior/posterior/decoder"]
        S3 --> S4["4 Training<br/>NLL + KL with free bits"]
        S4 --> S5["5 Causal test<br/>imagine u=0 vs u=1"]
        S5 --> S6["6 Ensemble<br/>epistemic uncertainty"]
    end
    subgraph REAL["Real ICU data (MIMIC-IV, no known truth)"]
        S7["7 MIMIC-IV hourly data<br/>+ training, test evaluation"] --> S8["8 Imagination on real stays<br/>vasopressor check: ~0 effect"]
        S8 --> S9["9 Ablation<br/>actions barely used"]
    end
    subgraph SEMI["Semi-synthetic (real data + planted truth)"]
        S10["10 Plant K = 100<br/>85% recovered"] --> S10b["10b Threshold + noise floor<br/>K = 30 clear at 1, 3, 6 h"]
        S10b --> S11["11 act_now path<br/>fixes hour 1 for strong effects"]
        S11 --> S12["12 Confounder with a visible ramp"]
        S12 --> S12b["12b Confounder hidden until after the start<br/>unbiased for the wrong reason"]
    end
    S6 --> S7
    S9 --> S10
```

## The model, with shapes

B = patients in a batch, T = 48 hours, V = vitals (2 simulated, 5 in MIMIC: HR, RR, SpO2,
MAP, Temp), A = actions (2 simulated, 3 in MIMIC: vasopressor dose, fluid bolus,
ventilation), D = 64 (model width), Z = 8 (latent size).

```mermaid
flowchart LR
    X["x, mask, delta<br/>(B, T, 3V)"] --> OBS["obs_proj<br/>Linear 3V -> D"]
    AP["a_(t-1)<br/>(B, T, A)"] --> ACT["act_proj<br/>Linear A -> D"]
    HR["hour index<br/>(T,)"] --> TIME["time_emb<br/>(T, D)"]
    OBS --> SUM(("+"))
    ACT --> SUM
    TIME --> SUM
    SUM --> TOK["tokens (B, T, D)"]
    TOK --> TR["causal transformer<br/>2 layers, 4 heads"]
    TR --> H["h_t (B, T, D)"]
    AN["a_t (B, T-1, A)"] --> AE["act_now (Stage 11, default)<br/>Linear A -> D, GELU, Linear D -> D"]
    H --> PRI["prior p(z_t+1 | h_t, a_t)<br/>-> mean, log sigma (B, T-1, Z)"]
    AE --> PRI
    H --> POST["posterior q(z_t+1 | h_t, a_t, x_t+1, mask_t+1)<br/>training only"]
    AE --> POST
    XN["x_t+1, mask_t+1<br/>(B, T-1, 2V)"] --> POST
    PRI -.->|imagination| Z["z_t+1 (B, T-1, Z)"]
    POST -.->|training| Z
    Z --> DEC["decoder p(x_t+1 | z_t+1, h_t)<br/>-> mean, log sigma (B, T-1, V)<br/>never sees the action"]
    H --> DEC
```

Loss per predicted hour: Gaussian negative log-likelihood of the observed vitals (missing
ones count 0) + 1.0 x max(KL(posterior || prior), 1 nat). Imagination feeds sampled vitals
back in as if observed and uses the prior only.

## Stage by stage

Each stage: **Question**, **Hypothesis** (what we expected before running it), **Method**,
**Baseline** (what we compared against), **Result**, **Hypothesis right?**, **Lesson**, **Next**.

### Stage 1 - A simulated ICU
- **Question:** Can we build a small ICU where the true effect of every treatment is known?
- **Hypothesis:** A 4-number hidden state (MAP, sepsis severity, vasopressor effect, fluid
  volume) and a clinician who treats low MAP give realistic confounding.
- **Method:** 2,000 patients x 48 h. A clinician policy (vasopressor when MAP is low) and an
  "explore" policy (random actions 60% of the time). 10% of observations missing.
- **Baseline:** none; this stage creates the ground truth.
- **Result:** Treated patients look *worse* than untreated ones in the raw data, although
  the drug raises MAP: textbook confounding by indication. The policy had to be tuned
  (beta = 5) to give enough treated hours.
- **Hypothesis right?** Yes.
- **Lesson:** The raw association between a treatment and an outcome can have the wrong sign.
- **Next:** Turn patients into model inputs.

### Stage 2 - Turning patients into tensors
- **Question:** What exactly does the model see each hour?
- **Hypothesis:** Values, a missingness mask and "hours since last seen" are enough to
  handle missing data honestly.
- **Method:** x (vitals, 0 where missing), mask, delta, a (actions); split by patient
  70/15/15; one normalisation from the policy training split.
- **Baseline:** none.
- **Result:** Tensors of shape (patients, 48, vitals); missing values cannot leak.
- **Hypothesis right?** Yes.
- **Lesson:** Split by patient, never by hour; tell the model what is missing.
- **Next:** The model.

### Stage 3 - The model
- **Question:** How do we build a model that can be asked "what if I did a?"
- **Hypothesis:** A causal transformer summary h_t plus a latent z_{t+1} drawn from a prior
  that depends on the action, with a decoder that never sees the action, forces any
  action effect to go through z.
- **Method:** The diagram above; a forward pass printing every shape; causality checks.
- **Baseline:** none.
- **Result:** Changing hour 30 does not change h before hour 30; changing the action at hour
  30 changes only the prior for hour 31.
- **Hypothesis right?** Yes (structurally).
- **Lesson:** Test causality of the architecture directly; it is easy to leak the future.
- **Next:** Train it.

### Stage 4 - Training
- **Question:** Does the model learn to forecast, and does the latent carry information?
- **Hypothesis:** It will beat a naive forecast; free bits will prevent posterior collapse.
- **Method:** One model on policy data, one on explore data, same settings, early stopping.
- **Baseline:** Naive "last value carried forward".
- **Result:** One-step MAP error 2.35 / 2.38 mmHg; the 10-90% band contains 77-78% of
  outcomes; KL sits at the free-bits floor with 1 of 8 latent dimensions used.
- **Hypothesis right?** Partly: forecasts are good, but the latent is barely used.
- **Lesson:** Good one-step forecasts say nothing yet about treatment effects.
- **Next:** Ask it a causal question.

### Stage 5 - Imagination and the causal test
- **Question:** Does the model know what the vasopressor does?
- **Hypothesis:** The explore model will recover the true effect better than the policy model.
- **Method:** From hour 24, imagine 12 h with u = 0 and with u = 1 (paired random numbers).
- **Baseline:** The simulator's true effect (3.0 / 11.7 / 24.2 / 40.9 at +1 / +3 / +6 / +12 h).
- **Result:** Policy model 2.6 / 9.1 / 15.8 / 22.7; explore model 2.6 / 9.8 / 18.6 / 31.4.
  Both get the sign and the first hour right and fall short later.
- **Hypothesis right?** Yes: exploration helps, mostly in the long, rarely seen plans.
- **Lesson:** Conditioning on the full history removes measured confounding; long action
  sequences the data never contains (no positivity) are where it fails.
- **Next:** Can the model tell us when it is unsure?

### Stage 6 - Epistemic uncertainty with an ensemble
- **Question:** Do 5 models trained with different seeds disagree where data is missing?
- **Hypothesis:** Yes, so the ensemble spread flags unreliable counterfactuals.
- **Method:** 5 seeds per dataset; spread of their medians vs spread within a model.
- **Baseline:** The true effect from the simulator.
- **Result:** The ensemble is confidently wrong in the untreated arm: error 7-9 mmHg with a
  spread of only 0.6-0.8 mmHg; the truth is inside the 5 models' range for 1-9% of patients.
- **Hypothesis right?** No.
- **Lesson:** Models trained on the same data extrapolate the same way; seed ensembles do not
  detect missing positivity.
- **Next:** Real data.

### Stage 7 - Real ICU data (MIMIC-IV)
- **Question:** Does the same model train on real, messy ICU data?
- **Hypothesis:** It will forecast better than the naive baseline.
- **Method:** DuckDB extraction of 93,224 stays: 5 hourly vitals, norepinephrine-equivalent
  dose, fluid boluses, ventilation; outliers filtered before averaging; doses capped
  (363 + 108 hours); trained on a Kaggle GPU with W&B logging.
- **Baseline:** Naive carry-forward.
- **Result:** The prior beats naive by 7-17% MAE on every vital (MAP 7.02 vs 8.33);
  coverage 77-80%; 3 of 8 latent dimensions used.
- **Hypothesis right?** Yes.
- **Lesson:** Real data needs cleaning before averaging; one implausible reading can spoil an hour.
- **Next:** Ask the real model a causal question.

### Stage 8 - Imagination on real stays, and a vasopressor check
- **Question:** Does imagined MAP rise when the model "gives" a vasopressor to hypotensive patients?
- **Hypothesis:** A small positive effect, in the clinically expected direction.
- **Method:** 6,413 test moments with MAP < 65 and no earlier vasopressor; imagine NEE 0.1
  vs 0 for 6 h with paired random numbers.
- **Baseline:** Naive "started vs not started" comparison in the data.
- **Result:** +0.12 / +0.22 / +0.06 at 1 / 3 / 6 h: essentially zero. Multi-hour forecasts
  still beat naive.
- **Hypothesis right?** No.
- **Lesson:** A good forecaster can know almost nothing about the effect of an action.
- **Next:** Is the model ignoring the actions?

### Stage 9 - Does the model use the actions at all?
- **Question:** Does removing or shuffling the actions hurt the forecast?
- **Hypothesis:** If the model used them, it would.
- **Method:** Replace actions with zeros, shuffled actions, or zero vasopressor; measure the
  prior's sensitivity to the dose.
- **Baseline:** The real actions.
- **Result:** At most +0.5% MAP error; a dose change moves z by 0.04 against a spread of 1.65.
- **Hypothesis right?** The model barely uses the actions.
- **Lesson:** Ablation shows *whether* inputs matter; it cannot say whether the cause is the
  model or the data.
- **Next:** Plant a known effect to tell the two apart.

### Stage 10 - Semi-synthetic benchmark, K = 100
- **Question:** Is the near-zero effect an architecture problem or a data problem?
- **Hypothesis:** If the architecture can learn effects, it will recover a planted one.
- **Method:** Real MIMIC histories, MAP' = MAP + K * lagged dose (K = 100 mmHg per
  mcg/kg/min, +10 mmHg at steady state for dose 0.1); same model and training.
- **Baseline:** The planted effect, and the real-data model.
- **Result:** 54% recovered at hour 1, 85% from hour 2; ablation now matters (+27.6%).
- **Hypothesis right?** Yes, mostly: mainly a data problem, with a weak first hour.
- **Lesson:** Semi-synthetic data keeps real-data difficulty but plants a known answer.
- **Next:** How small an effect can it find?

### Stage 10b - Detection threshold and noise floor
- **Question:** What is the smallest planted effect that stands out from noise?
- **Hypothesis:** Somewhere between K = 10 and K = 100.
- **Method:** K = 10, 30, 100; then a second real-data model with another seed to measure how
  much two training runs disagree (the noise floor).
- **Baseline:** The real-data model; the seed-to-seed noise floor (up to 0.35 mmHg at 6 h).
- **Result:** Steady-state recovery 72 / 80 / 85%. K = 10 clears the noise floor at 3 and
  6 h but not at hour 1; K = 30 clears it at 1, 3 and 6 h. The real-data effect changes
  sign between seeds.
- **Hypothesis right?** Yes, with the threshold at K = 30 for all three hours.
- **Lesson:** "Distinguishable" needs a noise floor from repeated training, not only the
  spread across patients.
- **Next:** Fix the weak first hour.

### Stage 11 - The act_now path
- **Question:** Does giving the current action its own learned embedding fix hour 1?
- **Hypothesis:** Yes, for all K.
- **Method:** act_now(a_t) = Linear-GELU-Linear into prior and posterior; same training.
- **Baseline:** The original model, and each architecture's own noise floor.
- **Result:** K = 100: hour 1 54% -> 112%, steady state 85% -> 96%. K = 10: no change.
  Forecast accuracy and the real-data effect unchanged. Unit conversion checked: only 2
  wrong rows.
- **Hypothesis right?** Partly: fixed for strong effects, not for small ones.
- **Lesson:** An architecture change can fix a capacity problem without creating an
  effect the data does not contain.
- **Next:** What happens when there is a confounder the model cannot see? (act_now is now
  the default.)

### Stage 12 - Hidden confounding with a visible ramp
- **Question:** Does an unmeasured deterioration that drives treatment bias the learned effect?
- **Hypothesis:** The model learns the observational relationship, so the effect is biased by
  about -G * exp(-h/4) (G = 15: -11.3 / -6.9 / -3.2 mmHg at 1 / 3 / 6 h).
- **Method:** K = 30 plus -G * u, u ramping up over the 3 h before each vasopressor start
  and decaying (tau 4 h) after; G = 0, 5, 15.
- **Baseline:** True effect +1.9 / +2.9 / +3.0; the G = 0 model; the noise floor.
- **Result:** G = 15 learned -0.08 / +2.49 / +3.15; the part due to confounding (vs G = 0)
  is only -0.41 / +0.30 / +0.62. Checks: the confounder is injected as specified, and the raw
  data shows exactly the predicted bias (-11.27 / -6.76 / -3.39). Splitting moments by whether
  their history shows the ramp changes nothing.
- **Hypothesis right?** No: the model is not biased as predicted, although the data is.
- **Lesson:** Check the data first: the bias is really there, so the model is missing it.
  "The confounder is visible in the history" was not confirmed by the split.
- **Next:** Remove every trace of the confounder before and at the decision hour (Stage 12b).

### Stage 12b - A truly hidden confounder
- **Question:** If the deterioration leaves no trace before treatment, is the effect biased?
- **Hypothesis:** Yes, by -G * exp(-(h-1)/4).
- **Method:** u = 0 up to and including the start hour, 1 the hour after, decaying with tau
  4 h; G = 5, 15.
- **Baseline:** As in Stage 12.
- **Result:** Raw data bias -15.00 / -8.77 / -4.19, as predicted; the model's
  confounding-driven change is only -0.34 / -0.21 / +0.06 (G = 15). Its one-step prediction
  after a start moves by about 2% of the 15 mmHg drop in the data.
- **Hypothesis right?** No. "Visible at the decision hour" is ruled out as the explanation.
  The measured explanation: the one-step prior barely responds to a start (the same hour-1
  weakness as for the drug), so imagination never produces the drop. Later hours are learned,
  by teacher forcing, as recovery from an observed low MAP, a property of the history.
  Why the prior misses starts (rare: 0.93% of hours; needs current AND previous dose) is untested.
- **Lesson:** A counterfactual can be right for the wrong reason. Here, under-fitting hides the
  confounding; a better-fitted model would be biased by up to -15 mmHg. Benchmarks must check
  the mechanism (the one-step response to the action), not only the final number.
- **Next:** Test why the prior misses starts (one-step residuals at real starts; a start
  indicator or more capacity on the current-action path), knowing that fixing it should
  *expose* the bias, which is what a confounding benchmark is for.

## How we got from "no effect" to an answer (Stages 8-12b)

```mermaid
flowchart TD
    A["Stage 8: imagined vasopressor effect on real data ~ 0"] --> B{"Stage 9: does the model<br/>use the actions?"}
    B -- "barely (+0.5% error without them)" --> C{"Stage 10: can it learn a<br/>planted effect (K = 100)?"}
    C -- "yes, 85% from hour 2" --> D["Mainly a DATA problem<br/>(confounding, titration, positivity)"]
    C -- "only 54% at hour 1" --> E["Weak path from the current action"]
    D --> F{"Stage 10b: how small an effect,<br/>above which noise?"}
    F -- "K = 30 clear at 1, 3, 6 h;<br/>real-data effect within seed noise" --> G["Real-data effect is zero within noise"]
    E --> H{"Stage 11: act_now embedding"}
    H -- "K = 100 hour 1: 54% -> 112%;<br/>small K unchanged" --> I["Architecture weak spot fixed for strong effects"]
    G --> J{"Stage 12: confounder with<br/>a visible ramp"}
    I --> J
    J -- "learned ~ true, not biased as predicted" --> K{"Checks 1-3"}
    K -- "1: injected as specified<br/>2: raw data shows the full bias" --> K2["The data is confounded;<br/>the model does not reproduce it"]
    K -- "3: ramp in history or not:<br/>no difference" --> K3["'Visible in the history' not confirmed"]
    K2 --> L{"Stage 12b: no trace before<br/>or at the decision hour"}
    K3 --> L
    L -- "raw bias -15 at 1 h;<br/>model still ~unbiased" --> M["'Visible at decision hour' ruled out"]
    M --> N["Measured: one-step prior barely responds to a start (2% of the drop)<br/>=> unbiased for the wrong reason (under-fitting)"]
    N --> O["Next: why the prior misses starts;<br/>a better fit should expose the bias"]
```

## Glossary

- **World model:** a model that predicts how a system evolves, and how it would evolve under
  actions we choose, so it can be used to *imagine* outcomes.
- **Latent (z):** an unobserved summary the model invents; here 8 numbers per hour.
- **Prior / posterior:** the latent distribution without / with sight of the next observation.
- **KL, free bits:** how far posterior and prior differ; free bits let the model keep 1 nat
  of information without penalty, to avoid posterior collapse (latent ignored).
- **Imagination (rollout):** predicting forward hour by hour from the prior, feeding samples back.
- **Counterfactual, paired random numbers:** the same patient and the same randomness under two
  different plans, so the difference is due to the plan only.
- **Confounding:** a factor that drives both treatment and outcome (sicker patients get the
  drug), so the raw association is not the effect. **Unmeasured (hidden) confounding:** that
  factor is not in the data.
- **Positivity:** every plan we ask about must actually occur in the data for similar patients.
- **Aleatoric / epistemic uncertainty:** randomness of the patient / the model's own ignorance.
- **Ablation:** removing an input to see whether the model relies on it.
- **Semi-synthetic benchmark:** real data plus a planted effect of known size.
- **Noise floor:** how much two training runs on identical data disagree; an effect must beat it.
- **NEE:** norepinephrine-equivalent vasopressor dose, mcg/kg/min.
- **MAP:** mean arterial pressure, mmHg.

## Open questions

1. **Hidden confounding in practice.** How large is unmeasured confounding in real ICU data,
   and can proxies (notes, lab trends, measurement frequency) make it visible to the model?
2. **Positivity for action sequences.** How can a world model detect, for a given history and
   plan, that the data does not support the question, and refuse to answer?
3. **Uncertainty that catches shared extrapolation.** Seed ensembles failed (Stage 6); do
   diverse ensembles, Bayesian or conformal methods do better?
4. **First-hour response.** Why does the prior barely respond in the first hour, both to small
   drug effects and to the drop after a start (rarity? the start being an interaction of the
   current and previous dose? the KL budget?), and how big does the confounding bias become
   once it does respond?
5. **Validation without ground truth.** Semi-synthetic benchmarks, natural experiments and
   randomised trials (e.g. MAP-target trials) as external checks.
6. **Hybrid models.** Does built-in physiology make untreated arms extrapolate correctly?
7. **Transfer and resolution.** Does a MIMIC model hold in another hospital (eICU)? Would
   continuous time avoid hourly-binning artefacts?
