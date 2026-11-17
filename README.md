# ICU world model (learning project)

An action-conditioned transformer world model (TSSM-style), trained on
simulated ICU patients. The goal is to understand how world models work,
especially whether they learn the *effect* of treatments, not just their
correlation with outcomes.

## Layout

```
icu-world-model/
  src/
    simulator.py   Stage 1  simulated patients: hidden state, actions, observations
    data.py        Stage 2  hourly tensors: x, mask, delta_t, action; patient split
    model.py       Stage 3  token embedding, causal transformer, prior/posterior/decoder
    train.py       Stage 4  masked Gaussian NLL + beta * KL (free bits), loss plots
    rollout.py     Stage 5  imagination from the prior; u=0 vs u=1 causal test
  notebooks/       exploration and the causal-test notebook
  outputs/         generated datasets, checkpoints, figures (not versioned)
  requirements.txt pinned package versions
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib jupyter
```

Run scripts from the project root, e.g. `python src/simulator.py`.
