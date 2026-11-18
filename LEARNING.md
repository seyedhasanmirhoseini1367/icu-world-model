# Learning world models on simulated ICU patients

This is a learning log for someone meeting world models for the first time. Each
stage explains what we did, why, what the plots show, and what it teaches. All numbers
are from our own runs. Terms are defined in **bold** the first time they appear.

A **world model** is a model that learns how an environment changes over time, so that
you can ask it "what would happen next?" without touching the real environment. An
**action-conditioned** world model also takes the actions you could take (here: give a
vasopressor or not) and predicts what happens *under each action*. That is the point of
the whole project: in medicine we care less about "what will this patient's blood
pressure be?" and more about "what will it be *if I start this drug*?"

We build everything on a simulator first, because there we know the true answer and can
check whether the model got it right. On real ICU data you never can.

---

## Stage 1 - A simulated ICU

**What we did.** [`src/simulator.py`](src/simulator.py) creates 2000 patients, each
observed for 48 hours, one step per hour. Every patient has a **hidden state**: four
numbers the model will never see directly.

- `m`: mean arterial pressure (MAP), the blood pressure that matters for organ perfusion
- `s`: sepsis severity, from 0 (none) to 1 (severe); it pulls MAP down
- `c`: the effect of the vasopressor, which follows the infusion with a lag of about 2 hours
- `v`: extra fluid volume from boluses, 20% of which washes out each hour

Each hour a clinician picks an **action**: a vasopressor rate `u` (0 or 1) and a fluid
bolus `f` (0 or 0.5 L). What we get to see are noisy **observations**: MAP (±2 mmHg
noise) and heart rate, each missing 10% of the time, as in a real chart.

**The two datasets.** In the *policy* dataset a simulated clinician treats when MAP is
low: vasopressor with probability 0.8 if observed MAP < 65, a bolus if MAP < 60 and no
bolus in the last 3 hours. In the *explore* dataset the same clinician is overruled 60%
of the time by a random action.

**Why it was built this way: confounding.** **Confounding** means something drives
both the treatment and the outcome, so their raw association is misleading. Here,
patients with low MAP get the vasopressor, and patients with low MAP *also* tend to have
low MAP an hour later. So in the policy data, MAP one hour after `u=1` is **4.5 mmHg
lower** than after `u=0`, even though the drug *raises* MAP by **+3.0 mmHg** in that
hour. In the explore data the same raw comparison gives **+1.2 mmHg**: closer, because
most treatment decisions there are random and unrelated to MAP.

| | Policy | Explore |
|---|---|---|
| True MAP mean, range (mmHg) | 70.0, 50.7 to 104.7 | 86.2, 44.1 to 153.6 |
| Hours with vasopressor | 18.1% | 31.9% |
| Hours with a fluid bolus | 2.2% | 15.3% |
| Raw MAP(t+1 \| u=1) - MAP(t+1 \| u=0) | -4.5 mmHg | +1.2 mmHg |

**What the plots show.** [Policy patients](outputs/figures/patients_policy.png) and
[explore patients](outputs/figures/patients_explore.png), from no sepsis (left) to the
most severe (right). In the policy data the sickest patients hover around 60-65 mmHg:
the clinician keeps pulling them back up. In the explore data random fluids and
vasopressors push some patients far above normal.

**What it teaches.**
- Getting the *data* right took several rounds. With the first sepsis strength
  (`beta = 8`), untreated MAP fell towards 0; with `beta = 2` almost nobody needed
  treatment (4% of hours), so there would have been little treatment to learn from.
  We settled on `beta = 5` (untreated MAP settles at `m_base - 50*s`).
- A model can only learn the effect of an action it has seen, in situations where it
  has seen it. This requirement is called **positivity**; it returns in Stage 5.
- The simulator's dynamics are linear with additive noise. One consequence, which
  matters in Stage 5: the true effect of the vasopressor is the same for every patient.

---

## Stage 2 - Turning patients into tensors

**What we did.** [`src/data.py`](src/data.py) turns each patient into four arrays of
shape (48 hours, 2):

- `x`: MAP and HR as **z-scores**: (value - mean) / standard deviation, so both vitals
  are on a comparable scale. Missing values are set to 0.
- `mask`: 1 where the vital was measured. Without it, a missing value (0) would look like
  "exactly average".
- `delta`: hours since that vital was last measured (here never more than 4).
- `a`: the action chosen at that hour.

Patients are split 1400 / 300 / 300 into *train* (to fit the model), *validation* (to
decide when to stop training) and *test* (touched only in Stage 5). The split is by
patient, so no patient's hours leak between sets.

**One shared scale.** Both datasets are z-scored with the policy training set's
statistics (MAP 70.0 ± 6.5 mmHg, HR 90.4 ± 7.9 bpm). One z-unit then means the same
number of mmHg for both models. On this scale explore MAP averages +2.5 z with 2.35
times the spread: the explore model trains on a much wider range of patients.

**What it teaches.** Normalising with training statistics only is not pedantry: using
the test set's statistics leaks information about it. And the choice of shared scale is
part of making the two models comparable later.

---

## Stage 3 - The model

**What we did.** [`src/model.py`](src/model.py), 120,676 parameters. For each hour `t`:

1. **Token.** The hour's observation `[x, mask, delta]`, the *previous* action
   `a_{t-1}`, and a learned embedding of the hour number are each projected to 64
   numbers and added together.
2. **h, the history summary.** A small **causal transformer** (2 layers, 4 heads) reads
   the tokens. "Causal" means hour `t` can only look at hours `0..t`, never the future.
   Its output `h_t` is a 64-number summary of everything known up to hour `t`.
3. **Latent z.** A **latent** variable is one the model invents to explain the data; it
   is not measured. Here `z_{t+1}` is 8 numbers describing "what happens next".
4. **Prior** `p(z_{t+1} | h_t, a_t)`: the model's guess about `z_{t+1}` from the history
   and the action, *before* seeing hour `t+1`. It outputs a mean and a spread (a
   Gaussian distribution).
5. **Posterior** `q(z_{t+1} | h_t, a_t, x_{t+1})`: the same, but allowed to peek at the
   real next observation. It exists only during training.
6. **Decoder** `p(x_{t+1} | z_{t+1}, h_t)`: turns `z` back into a predicted MAP and HR,
   again as a mean and a spread.

This design (transformer history + stochastic latent per step) is called a **TSSM**,
transformer state-space model.

**The key design choice.** The decoder never sees the action. The only way an action can
change the predicted vitals is by changing the prior's distribution over `z`. So "what
does the model think the drug does?" is a question about the prior.

**Checks we ran.** Changing MAP at hour 30 leaves `h` for hours 0-29 unchanged (no
peeking at the future). Flipping the vasopressor at hour 30 leaves `h` up to hour 30
unchanged but changes the prior for hour 31 (the action reaches exactly the next step).

**What it teaches.** Reading the tensor shapes through one forward pass is the fastest
way to understand a model: 16 patients x 48 hours become tokens `(16, 48, 64)`, then the
model drops to 47 positions because the last hour has nothing left to predict.

---

## Stage 4 - Training

**What we did.** [`src/train.py`](src/train.py) trains one model per dataset with the
same settings, using the loss

`loss = reconstruction + beta x max(KL, free bits)`

- **Reconstruction**: how surprised the decoder is by the real next observation
  (Gaussian negative log-likelihood, counted only for measured vitals).
- **KL** (Kullback-Leibler divergence): a measure, in **nats** (the natural-log unit
  of information), of how different two distributions are. Here it measures how far
  the posterior (which saw the answer) moved away from the prior (which did not). It
  pushes the prior to predict what the posterior learns, and charges the posterior for
  every bit of information it uses.
- **Free bits**: the first 1 nat of KL per hour is free. Without it, the KL penalty can
  push the posterior to copy the prior exactly, so `z` carries no information at all.
  That failure is called **posterior collapse**.

Training stops when validation loss has not improved for 15 epochs (**early stopping**);
the best checkpoint is kept.

| | Policy | Explore |
|---|---|---|
| Best epoch (stopped at) | 67 (82) | 92 (107) |
| Validation reconstruction (nats/hour) | -0.03 | 0.06 |
| Validation KL (nats/hour) | 1.11 | 1.05 |
| Latent dimensions used (KL > 0.1 nats) | 1 of 8 | 1 of 8 |
| One-step MAP error, validation | 2.35 mmHg | 2.38 mmHg |
| Observed MAP inside the 10-90% band | 77% | 78% |

**What the plots show.**
- [Loss curves](outputs/figures/train_curves.png): train and validation stay close (no
  overfitting). Reconstruction falls below 0: that is fine, a narrow density can exceed
  1, so its negative log is negative. KL climbs to just above the free-bits line and
  stays there.
- [KL per latent dimension](outputs/figures/kl_per_dim.png): one dimension carries all
  of it; seven are unused.
- [One validation patient](outputs/figures/one_step_val_patient.png): the orange band is
  the honest one-hour-ahead prediction from the prior. It follows the hidden true MAP,
  and about 80% of measurements fall inside the 10-90% band, as they should.

**What it teaches.**
- **Not collapsed, but propped up.** The KL sits right on the free-bits floor. The
  posterior uses the 1 free nat and almost nothing more. The dotted green line in the
  patient plot shows what it spends it on: it follows each measurement exactly, noise
  included. So `z` mostly encodes the measurement noise of the next hour; the
  predictable part already lives in `h`.
- **A low KL does not mean actions are ignored.** The action moves the prior itself; the
  KL only counts what the posterior adds beyond the prior.
- **One-step accuracy is not the test.** Both models predict the next hour equally well
  (2.4 mmHg; the best possible is about 2 because of the measurement and process noise).
  That says nothing about whether they learned what the *drug* does.
- The band in the patient plot is **aleatoric uncertainty**: randomness the model
  expects even if it knew the dynamics perfectly (measurement noise, the patient's
  random drift). It cannot be reduced by more data.

---

## Stage 5 - Imagination and the causal test

**What we did.** [`src/rollout.py`](src/rollout.py) and
[`notebooks/causal_test.ipynb`](notebooks/causal_test.ipynb). A **rollout** (or
"imagination") means letting the model run forward on its own: give it the real history
up to hour 24, then for 12 hours sample `z` from the prior, decode a predicted
observation, and feed that prediction back in as if it had been measured. We do this
50 times per patient, once with the vasopressor on (`u=1`) and once off (`u=0`), no
fluids. The **predicted effect** is the difference of the two medians.

The **true effect** comes from the simulator: start from the patient's real hidden state
at hour 24, run 12 hours with `u=1` and with `u=0`, using the *same* random noise in both
arms so that only the action differs. This is a **counterfactual**: what would have
happened to the same patient under a different action. Only a simulator gives it to you.

Test patients with observed MAP below 70 at hour 24: 166 (policy), 33 (explore).

| Hours after hour 24 | +1 | +3 | +6 | +12 |
|---|---|---|---|---|
| True effect (mmHg) | 3.0 | 11.7 | 24.2 | 40.9 |
| Policy model | 2.6 | 9.1 | 15.8 | 22.7 |
| Explore model | 2.6 | 9.8 | 18.6 | 31.4 |

Where the error comes from, MAP at +12 h:

| | Imagined u=1 | True u=1 | Imagined u=0 | True u=0 |
|---|---|---|---|---|
| Policy model | 84.0 | 95.4 | 61.3 | 54.5 |
| Explore model | 89.0 | 89.2 | 57.6 | 48.3 |

**What the plots show.**
- [Predicted vs true effect](outputs/figures/causal_effect.png): both models start on
  the true curve and fall below it; the explore model (blue) stays closer. In the
  scatter, every patient's true effect at +6 h is the same 24.2 mmHg (a property of our
  linear simulator), so the vertical spread is pure model error.
- [Example patients](outputs/figures/rollout_examples.png): green is `u=1`, orange
  `u=0`; solid lines are imagined, dashed are true. The models' `u=0` lines level off
  around 57-61 mmHg while the true untreated patients keep falling.

**What it teaches.**
- **Confounding did not fool either model in the simple sense.** Both got the sign and
  the one-hour size right (2.6 vs 3.0 mmHg), even the policy model whose raw data says
  the drug *lowers* MAP. The reason: the clinician's decision depends only on observed
  MAP, and the model conditions on the observed history `h`. Conditioning on everything
  that drove the decision removes the confounding.
- **Positivity is what hurt.** Positivity (also called overlap) means every action you
  want to ask about must have been taken, at least sometimes, in comparable situations. A
  hypotensive patient left untreated for 12 hours essentially never occurs in either
  dataset (probability about 0.2^12 under the clinician, 0.38^12 in the explore data),
  so both models extrapolate the `u=0` arm and fall back on "MAP drifts back to the
  middle". The policy model also never saw sustained vasopressors in a patient whose
  MAP was already back to normal (the clinician stops at 65), so it gets the `u=1` arm
  wrong too.
- **Exploration buys coverage.** The explore model gets the `u=1` arm right to within
  0.2 mmHg; its whole error is in the `u=0` arm that even random exploration rarely
  produces for 12 hours in a row.
- **Imagination compounds.** Each imagined hour is built on the previous imagined hour,
  so small one-step errors grow over the horizon. One-step accuracy (Stage 4) was equal
  for both models; 12-hour counterfactuals are not.

---

## Stage 6 - Epistemic uncertainty with an ensemble

**What we did.** [`src/ensemble.py`](src/ensemble.py) trains 5 models per dataset that
differ only in their random seed (initial weights and the order of training batches);
seed 0 is the Stage 4 model. Each member reruns the Stage 5 causal test on the same
patients. For every patient and future hour we measure two spreads of imagined MAP:

- **Aleatoric** (within one model): the spread of its 50 samples, averaged over members.
  Randomness the model believes is in the world itself.
- **Epistemic** (between models): the spread of the 5 members' medians.
  **Epistemic uncertainty** is uncertainty about the model itself: how much the
  data failed to pin down the dynamics. More (or more varied) data can reduce it;
  aleatoric uncertainty it cannot.

The hope behind ensembles: where data is missing, independently trained models should
fill the gap in different ways and so *disagree*, flagging "I don't know".

**Results.** Spread of imagined MAP, standard deviation in mmHg, mean over patients:

| Dataset, arm | +1 h aleatoric / epistemic | +6 h | +12 h |
|---|---|---|---|
| Policy, u=0 | 2.75 / 0.54 | 3.44 / 0.66 | 3.45 / 0.76 |
| Policy, u=1 | 2.69 / 0.56 | 3.47 / 1.20 | 3.68 / 1.78 |
| Explore, u=0 | 2.83 / 0.57 | 3.83 / 0.70 | 4.00 / 0.64 |
| Explore, u=1 | 2.80 / 0.56 | 4.20 / 1.02 | 4.96 / 1.97 |

Compared with the truth at +12 h:

| Dataset, arm | Ensemble error (mean of 5) | Epistemic SD | Truth inside the 5 models' range |
|---|---|---|---|
| Policy, u=0 | +7.0 mmHg (too high) | 0.76 | 9% of patients |
| Policy, u=1 | -13.6 mmHg (too low) | 1.78 | 1% |
| Explore, u=0 | +9.1 mmHg (too high) | 0.64 | 3% |
| Explore, u=1 | -0.5 mmHg | 1.97 | 39% |

The predicted effect at +12 h ranges over 18.4-22.7 mmHg across the policy members and
29.6-34.1 across the explore members; the truth is 40.9 for both.

**What the plots show.**
- [Ensemble spread](outputs/figures/ensemble_spread.png), left: the 5 members' effect
  curves lie close together, all well below the true curve. Right: the dotted
  (aleatoric) lines are 3-5 mmHg; the solid (epistemic) lines are much smaller, and the
  `u=0` one (orange) stays flat at about 0.7 mmHg for all 12 hours.
- [Example patients](outputs/figures/ensemble_example.png): the five orange medians sit
  on top of each other, around 57 mmHg, while the true untreated MAP falls to 45-47.

**The answer to the question: no.** Disagreement does *not* grow in the `u=0` arm,
where data is missing. It grows in the `u=1` arm instead. The ensemble is most
confident exactly where it is most wrong: in the explore data, `u=0` has the lowest
epistemic spread (0.64) and an error of 9.1 mmHg, while `u=1` has the highest spread
(1.97) and almost no error.

**What it teaches.**
- **An ensemble only measures how much the members disagree, not how wrong they are.**
  All five share the same architecture, data and training recipe, so they share the
  same way of filling a gap. Here that is "without treatment, MAP drifts back towards
  the average patient", a sensible guess from data in which low MAP almost always got
  treated. Five models making the same mistake look certain.
- **Epistemic uncertainty from seeds is a lower bound.** It catches gaps that can be
  filled in several different ways; it misses gaps that every model fills the same way.
  Missing positivity (Stage 5) produces exactly the second kind.
- **Where the spread did grow, it meant something.** In the `u=1` arm the members drift
  apart with the horizon: each imagined hour compounds slightly different learned
  dynamics. That is real model uncertainty, but it is not where the big errors are.
- On real data, where there is no simulator to check against, a narrow ensemble band
  must never be read as "the model knows". The data's coverage of the action you ask
  about has to be checked separately.

---

## Stage 7 - Preparing real ICU data (MIMIC-IV)

**What we did.** [`src/mimic_data.py`](src/mimic_data.py) turns the real MIMIC-IV ICU
tables into exactly the tensors the simulator gave us (`x`, `mask`, `delta`, `a`), so
the same model and training loop run unchanged. **MIMIC-IV** is a public, de-identified
database of patients from one Boston hospital; we use four of its ICU tables:
`icustays` (admission and discharge times), `chartevents` (bedside measurements),
`inputevents` (drugs and fluids) and `procedureevents` (procedures such as ventilation).

Each ICU stay becomes 48 hourly steps from ICU admission:

- **5 vitals**: heart rate, respiratory rate, SpO2 (oxygen saturation), MAP (from the
  arterial line if there is one, otherwise the blood-pressure cuff) and temperature
  (Celsius, or Fahrenheit converted). Each hour gets the mean of its readings;
  impossible single readings (e.g. heart rate 900) are dropped first.
- **3 actions**:
  - *Vasopressor dose* as a **norepinephrine equivalent**: different vasopressors
    converted to the dose of norepinephrine with a similar effect (epinephrine x1,
    phenylephrine x0.1, dopamine x0.01, vasopressin 2.5 per unit/min). Infusions are
    recorded as "rate R from start to end", so each hour gets the **time-weighted**
    average: an infusion running for the second half of an hour counts half.
  - *Fluid bolus* in litres: saline or Ringer's lactate, but only orders in the
    "IV fluid bolus" category. In the local sample, only 79 of 1,325 saline rows are
    boluses; 848 are small volumes used to carry drugs.
  - *Invasive ventilation*: the fraction of the hour spent on a ventilator.
- **valid**: real stays are not all 48 hours long. Shorter stays are padded at the end,
  and the loss ignores padded hours. We checked that a stay padded after 30 hours gives
  exactly the same loss as the same stay cut to 30 hours.

**How it was tested.** The local `sample/` folder is incomplete (its event tables are
random rows from all of MIMIC, not complete stays), so it can only show that the code
runs: 1,386 stays, tensors of shape `(16, 48, 5)` and `(16, 48, 3)` per batch of 16. To
check the logic itself, a hand-made stay with known answers was run through it: a
norepinephrine infusion from 0:30 to 2:30 at 0.1 gives 0.05, 0.1, 0.05 in hours 0-2;
ventilation from 6:15 to 8:00 gives 0.75 then 1.0; 98.6 F becomes 37.0 C. That test
found one bug (an impossible reading spoiled its whole hour), now fixed.

[`notebooks/kaggle_mimic.ipynb`](notebooks/kaggle_mimic.ipynb) runs the extraction and
the Stage 4 training on the full tables on Kaggle. It ran end to end locally on the
sample for 2 epochs; it has not been run on the full data yet.

**What it teaches.**
- **Most of the work in real data is deciding what an "action" is.** The simulator gave
  us `u` and `f` for free. In MIMIC we had to choose units, convert between drugs,
  spread infusions over hours and tell a bolus from a drug carrier. Each choice changes
  what the model can learn, and none of them is in the model code.
- **Real missingness is not random.** In the simulator every vital was missing 10% of
  the time by coin flip. In real ICU data, temperature is measured about every 4 hours
  and an arterial line exists only for sicker patients, so *whether* something is
  measured says something about the patient. The `mask` and `delta` inputs let the model
  use that, but they also let it learn hospital habits rather than physiology.
- **The real clinician policy is the policy dataset, not the explore dataset.** There
  is no random exploration in MIMIC. Everything Stages 5 and 6 showed about missing
  positivity applies, and there is no simulator to tell us the true effect.

---

## Stage 8 - What we learned

### The main lessons

1. **A world model is a learned simulator, and the action enters through the prior.**
   Everything the model believes about a treatment is in how `p(z_{t+1} | h_t, a_t)`
   moves when `a_t` changes. Asking "what does the model think the drug does?" means
   imagining with different actions and comparing.

2. **Predicting the next hour well is not the same as knowing what treatment does.**
   Both models predicted the next hour's MAP equally well (2.35 and 2.38 mmHg, about 80%
   band coverage). Twelve hours after starting a vasopressor they predicted effects of
   22.7 and 31.4 mmHg, against a true 40.9. Standard validation metrics cannot see this.

3. **Confounding by things the model observes can be handled; missing data on actions
   cannot.** Our clinician decided from observed MAP only, so conditioning on the history
   removed the confounding: the raw data said the drug *lowers* MAP by 4.5 mmHg, yet
   both models got the right sign and the right one-hour size. What they could not fix
   was positivity: treatment sequences that (almost) never happened in the data, such as
   12 hours without treatment at low MAP.

4. **Exploration in the data helps, but only for what it explores.** The explore model
   got the vasopressor arm right to within 0.2 mmHg. Even with 60% random actions, a
   12-hour run of no treatment at low MAP was too rare to learn, so its untreated arm
   was 9 mmHg too high.

5. **Errors compound in imagination.** Small one-step errors grow with every imagined
   hour fed back in as if it were real.

6. **Ensembles measure disagreement, not error.** Five models differing only in seed
   agreed closely (spread 0.6-0.8 mmHg) exactly where they were most wrong (7-9 mmHg),
   because they all filled the gap in the same way. The truth was inside the ensemble's
   range for only 3-9% of patients in the untreated arm.

7. **The latent `z` was barely used.** KL sat at the free-bits floor with 1 of 8
   dimensions active, encoding mostly measurement noise. In this simple world the
   history `h` already carried the predictable part. A richer world may need more.

### What a world model on real ICU data can and cannot tell us

**It can:**
- forecast vitals under the care patients actually receive, and flag deterioration;
- simulate "what usually happens next" for patients like this one under usual care;
- compare *small* departures from usual care, where similar patients really were
  treated differently (for example, starting a vasopressor an hour earlier or later);
- give a compact summary of patient state (`h`) for other tasks.

**It cannot, without strong extra assumptions:**
- give the effect of strategies clinicians rarely or never use (no positivity). It will
  still produce a confident-looking answer;
- remove confounding by information that is not in the tables: what the clinician saw
  at the bedside, read in the notes or heard from the patient. Our simulated clinician
  used only charted MAP; real ones do not;
- separate physiology from hospital habits. When and how often something is measured
  depends on how sick the patient is, and the model learns both;
- be checked against the truth. There is no simulator for real patients, so neither
  validation loss nor ensemble agreement can confirm a counterfactual.

### Open questions for a PhD proposal

1. **Detecting missing positivity for action sequences.** How can a world model tell,
   for a given history and a planned sequence of actions, whether the data supports the
   question, and refuse to answer when it does not? (Sequential propensity models,
   density estimation over `(h, a)` pairs.)
2. **Uncertainty that catches shared extrapolation.** Seed ensembles failed here. Do
   ensembles with different architectures or training data, Bayesian approaches, or
   conformal prediction for rollouts do better, measured on simulators where the truth
   is known?
3. **Validation without ground truth.** Semi-synthetic benchmarks (real patient
   histories with simulated treatment effects), natural experiments such as changes in
   hospital protocols, and agreement with randomised trials of MAP targets
   (e.g. SEPSISPAM, the 65 trial) as external checks.
4. **Unmeasured confounding.** How sensitive are a world model's counterfactuals to
   confounders it cannot see, and can clinical notes or other proxies reduce it?
5. **Hybrid models.** Our simulator has simple physiological structure. Can building
   known physiology into the model (grey-box or mechanistic components) make the
   untreated arm extrapolate correctly where a purely learned model did not?
6. **Informative measurement.** Modelling *when* things are measured together with
   *what* is measured, so that the measurement process is not mistaken for physiology.
7. **From prediction to decisions.** When is evaluating a treatment strategy with a
   world model (model-based off-policy evaluation) more reliable than reweighting the
   observed data, and how should the two be combined?
8. **Transfer and resolution.** Does a model trained on one hospital (MIMIC) hold in
   another (e.g. eICU)? Do continuous-time models avoid problems that hourly binning
   creates?

---

## Glossary

| Term | Meaning | First defined |
|---|---|---|
| World model | A learned model of how an environment changes, used to predict "what happens next" | Introduction |
| Action-conditioned | Predicts what happens under each possible action | Introduction |
| Hidden state | The true variables of the system that we never observe directly | Stage 1 |
| Confounding | Something drives both the treatment and the outcome, so their raw association misleads | Stage 1 |
| Positivity (overlap) | Every action you ask about was actually taken in comparable situations | Stage 1, Stage 5 |
| z-score | (value - mean) / standard deviation | Stage 2 |
| h | The transformer's summary of everything up to hour t | Stage 3 |
| Latent z | Variables the model invents to describe what happens next | Stage 3 |
| Prior | Distribution over z before seeing the next observation | Stage 3 |
| Posterior | Distribution over z after seeing the next observation (training only) | Stage 3 |
| KL divergence | How far one distribution is from another, in nats | Stage 4 |
| Free bits | KL below this amount is not penalised | Stage 4 |
| Posterior collapse | The posterior copies the prior and z carries nothing | Stage 4 |
| Aleatoric uncertainty | Randomness in the world itself; more data does not remove it | Stage 4 |
| Rollout (imagination) | Letting the model run forward, feeding its own predictions back in | Stage 5 |
| Counterfactual | What would have happened to the same patient under a different action | Stage 5 |
| Epistemic uncertainty | Uncertainty about the model, from limited data; more data can reduce it | Stage 6 |
| Norepinephrine equivalent | Vasopressor doses converted to the norepinephrine dose with similar effect | Stage 7 |
