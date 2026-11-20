# ICU world model (learning project)

An action-conditioned transformer world model (TSSM-style), trained first on simulated ICU
patients and then on real MIMIC-IV data (93,224 stays). The goal is to understand how world
models work, especially whether they learn the *effect* of treatments, not just their
correlation with outcomes. Two designs are compared: a VAE that predicts the vitals
(Stages 1-13c) and a JEPA-style model that predicts embeddings (Stages 14-14c).

## Start here

| If you want... | Read |
|---|---|
| The results in one page | [docs/JOURNEY.md, "Conclusions"](docs/JOURNEY.md#conclusions-stages-1-14c) |
| How the stages connect (question, hypothesis, result, lesson) | [docs/JOURNEY.md](docs/JOURNEY.md) |
| Full explanations, numbers, figures and pre-registrations | [LEARNING.md](LEARNING.md) |
| Trained checkpoints | [Releases](https://github.com/seyedhasanmirhoseini1367/icu-world-model/releases) (table below) |

## Releases

Each release has a `README.md` and `manifest.json` describing every file and how to load it.

| Release | Contents |
|---|---|
| [v1-vae-final](https://github.com/seyedhasanmirhoseini1367/icu-world-model/releases/tag/v1-vae-final) | VAE world model, Stages 1-13 (simulated and MIMIC-IV checkpoints) |
| [v1.1-vae-stage13b](https://github.com/seyedhasanmirhoseini1367/icu-world-model/releases/tag/v1.1-vae-stage13b) | Stage 13b: start-weighted VAE models |
| [v1.2-vae-stage13c](https://github.com/seyedhasanmirhoseini1367/icu-world-model/releases/tag/v1.2-vae-stage13c) | Stage 13c: lowered-beta VAE controls |
| [v2-jepa](https://github.com/seyedhasanmirhoseini1367/icu-world-model/releases/tag/v2-jepa) | Stage 14: JEPA-style models with vitals probes |
| [v2.1-jepa](https://github.com/seyedhasanmirhoseini1367/icu-world-model/releases/tag/v2.1-jepa) | Stage 14b: JEPA with the action leak removed from the target |
| [v2.2-jepa](https://github.com/seyedhasanmirhoseini1367/icu-world-model/releases/tag/v2.2-jepa) | Stage 14c: leak-free JEPA with beta tuned (0.03) so `z` is used |

## Layout

```
icu-world-model/
  src/
    simulator.py   Stage 1  simulated patients: hidden state, actions, observations
    data.py        Stage 2  hourly tensors: x, mask, delta_t, action; patient split
    model.py       Stage 3  token embedding, causal transformer, prior/posterior/decoder
    train.py       Stage 4  masked Gaussian NLL + beta * KL (free bits), loss plots
    rollout.py     Stage 5  imagination from the prior; u=0 vs u=1 causal test
    ensemble.py    Stage 6  5 seeds per dataset; epistemic vs aleatoric spread
    mimic_data.py  Stage 7  same tensors from real MIMIC-IV icu tables
    *_mimic.py, semisynthetic.py, confounded.py, stage1x_runner.py
                   Stages 8-13c: imagination, ablation, planted effects, confounding, Kaggle runners
    model_jepa*.py, train_jepa*.py, stage14*_runner.py
                   Stages 14-14c: JEPA-style model, training, Kaggle runners
  notebooks/
    causal_test.ipynb   Stage 5 results
    kaggle_mimic.ipynb  Stage 7: extraction + training on the full MIMIC-IV tables
  outputs/         generated data and checkpoints (local only); figures/ and jepa/ (result
                   JSON and figures for Stages 14-14c) are versioned
  LEARNING.md      what each stage does and teaches, for a first-time reader
  docs/JOURNEY.md  the map: stage-by-stage question, hypothesis, result, lesson (Stages 1-14c)
  requirements.txt pinned package versions
```

Reproduce the simulated results in order:
`simulator.py`, `train.py`, `rollout.py`, `ensemble.py train`, `ensemble.py test`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib jupyter
```

Run scripts from the project root, e.g. `python src/simulator.py`.
