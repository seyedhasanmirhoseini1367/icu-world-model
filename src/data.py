"""
data.py - Stage 2: turn a simulated dataset into hourly tensors for the model.

Per patient and hour t (T = 48 hours, 2 vitals: MAP, HR):
    x      (T, 2)  vitals, z-scored with the mean/std of the POLICY dataset's
                   train split (the same numbers for both datasets); 0 where missing
    mask   (T, 2)  1 where the vital was observed at hour t, else 0
    delta  (T, 2)  hours since the vital was last observed (0 if observed now;
                   before the first observation it counts from hour -1)
    a      (T, 2)  action chosen at hour t: (u vasopressor 0/1, f bolus in L)
    idx    ()      patient's row in the .npz file (to look up the hidden state later)

The action a_t is taken AFTER x_t is seen and affects x_{t+1}. The model
(Stage 3) shifts it itself: the token at hour t gets a_{t-1}.

Patients are split 70/15/15 into train/val/test with a fixed seed, so no
patient appears in two splits. Run from the project root:  python src/data.py
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

OUT = Path(__file__).resolve().parent.parent / "outputs"
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SPLIT_SEED = 0
VITALS = ("MAP", "HR")


def split_patients(n, seed=SPLIT_SEED):
    """Random patient-level split. Returns dict name -> sorted patient indices."""
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(SPLIT_FRACTIONS[0] * n)
    n_val = int(SPLIT_FRACTIONS[1] * n)
    parts = dict(train=perm[:n_train], val=perm[n_train:n_train + n_val],
                 test=perm[n_train + n_val:])
    return {k: np.sort(v) for k, v in parts.items()}


def hours_since_observed(mask):
    """mask (N, T, V) bool -> delta (N, T, V) float: hours since last observation."""
    N, T, V = mask.shape
    delta = np.zeros((N, T, V))
    last = np.full((N, V), -1.0)                      # pretend last seen at hour -1
    for t in range(T):
        last = np.where(mask[:, t], t, last)
        delta[:, t] = t - last
    return delta


class ICUSequences(Dataset):
    """One item = one patient's whole 48-hour sequence.
    valid (N, T) marks real hours; simulated stays are always 48 h long (all True),
    real ICU stays can be shorter and are padded at the end (Stage 7)."""

    def __init__(self, x, mask, delta, a, idx, valid=None):
        self.x = torch.tensor(x, dtype=torch.float32)          # (N, T, V)
        self.mask = torch.tensor(mask, dtype=torch.float32)    # (N, T, V)
        self.delta = torch.tensor(delta, dtype=torch.float32)  # (N, T, V)
        self.a = torch.tensor(a, dtype=torch.float32)          # (N, T, A)
        self.idx = torch.tensor(idx)                           # (N,)
        self.valid = (torch.ones(self.x.shape[:2]) if valid is None
                      else torch.tensor(valid, dtype=torch.float32))   # (N, T)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return dict(x=self.x[i], mask=self.mask[i], delta=self.delta[i],
                    a=self.a[i], idx=self.idx[i], valid=self.valid[i])


def norm_stats():
    """{'mean': (2,), 'std': (2,)} from observed values of the POLICY train split.
    Shared by both datasets, so one z-unit means the same mmHg / bpm in both."""
    x_raw = np.load(OUT / "sim_policy.npz")["x"]              # (N, T, 2), NaN where missing
    train_x = x_raw[split_patients(len(x_raw))["train"]]
    return dict(mean=np.nanmean(train_x, axis=(0, 1)),        # (2,)
                std=np.nanstd(train_x, axis=(0, 1)))          # (2,)


def load_splits(name):
    """name: 'policy' or 'explore'. Returns (datasets dict, stats dict);
    use stats to turn model outputs back into mmHg / bpm."""
    raw = np.load(OUT / f"sim_{name}.npz")
    x_raw, mask, a = raw["x"], raw["mask"], raw["actions"]   # (N, T, 2) each
    splits = split_patients(len(x_raw))
    stats = norm_stats()

    x = np.where(mask, (x_raw - stats["mean"]) / stats["std"], 0.0)
    delta = hours_since_observed(mask)
    datasets = {k: ICUSequences(x[i], mask[i], delta[i], a[i], i) for k, i in splits.items()}
    return datasets, stats


def make_loaders(name, batch_size=64):
    datasets, stats = load_splits(name)
    loaders = {k: DataLoader(ds, batch_size=batch_size, shuffle=(k == "train"))
               for k, ds in datasets.items()}
    return loaders, stats


if __name__ == "__main__":
    for name in ["policy", "explore"]:
        datasets, stats = load_splits(name)
        print(f"== {name}: patients " + ", ".join(f"{k} {len(v)}" for k, v in datasets.items()))
        print("   norm stats (policy train)  " + "  ".join(
            f"{v}: mean {m:.1f} std {s:.1f}" for v, m, s in zip(VITALS, stats["mean"], stats["std"])))
        for k, ds in datasets.items():
            obs = ds.mask.bool()
            z = [ds.x[..., j][obs[..., j]] for j in range(2)]
            print(f"   {k:<5} normalised observed x: " + "  ".join(
                f"{v} mean {zj.mean():+.2f} std {zj.std():.2f}" for v, zj in zip(VITALS, z))
                + f" | observed {ds.mask.mean():.1%} | max delta {ds.delta.max():.0f} h")
        batch = next(iter(make_loaders(name)[0]["train"]))
        print("   one train batch: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in batch.items()))
    p, e = split_patients(2000), split_patients(2000)
    assert all((p[k] == e[k]).all() for k in p), "splits differ between datasets"
    assert not set(p["train"]) & set(p["val"]) and not set(p["train"]) & set(p["test"])
    print("splits are disjoint and identical for both datasets")
