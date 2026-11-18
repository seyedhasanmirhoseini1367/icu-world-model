"""
train.py - Stage 4: train one world model on the policy data and one on the
explore data, with identical settings, then plot what they learned.

Loss per predicted hour (averaged over patients and hours):
    recon = Gaussian NLL of the observed vitals x_{t+1} under the decoder,
            summed over the 2 vitals; a missing vital contributes 0
    kl    = KL( posterior || prior ), summed over the 8 latent dims
    loss  = recon + BETA_KL * max(kl, FREE_BITS)
Free bits: once the KL is below FREE_BITS nats it gets no gradient, so the
model may keep that much information in z without being pushed to discard it.

Early stopping on validation loss; the best checkpoint is kept.
Run from the project root:  python src/train.py
"""
import copy
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_splits, make_loaders   # noqa: E402
from model import WorldModel                  # noqa: E402

BETA_KL = 1.0
FREE_BITS = 1.0            # nats per predicted hour, for the KL summed over latent dims
LR, WEIGHT_DECAY = 1e-3, 1e-2
BATCH_SIZE = 64
MAX_EPOCHS, PATIENCE = 300, 15
USED_DIM_NATS = 0.1        # a latent dim counts as "used" above this KL
SEED = 0
DEVICE = "cpu"             # set to "cuda" for GPU runs (Stage 7 on Kaggle)
TIME_BUDGET_S = None       # stop before an epoch would end past this many seconds
WANDB_RUN = None           # active Weights & Biases run, only when WANDB_API_KEY is set

OUT = Path(__file__).resolve().parent.parent / "outputs"
FIG = OUT / "figures"


# ---------------------------------------------------------------- loss

def gaussian_nll(x, mu, log_sigma):
    """Per-element negative log-likelihood. All inputs (B, T-1, V) -> (B, T-1, V)."""
    return 0.5 * math.log(2 * math.pi) + log_sigma + 0.5 * ((x - mu) / log_sigma.exp()) ** 2


def gaussian_kl(mu_q, ls_q, mu_p, ls_p):
    """KL(q || p) for diagonal Gaussians, per dimension. All (B, T-1, Z) -> (B, T-1, Z)."""
    return ls_p - ls_q + (torch.exp(2 * ls_q) + (mu_q - mu_p) ** 2) / (2 * torch.exp(2 * ls_p)) - 0.5


def compute_loss(out):
    valid = out["valid_target"]                                                # (B, T-1), all 1 for simulated data
    n = valid.sum()                                                            # number of real predicted hours
    nll = gaussian_nll(out["x_target"], out["dec_mu"], out["dec_log_sigma"])   # (B, T-1, V)
    recon = ((nll * out["mask_target"]).sum(-1) * valid).sum() / n             # scalar, nats/hour
    kl_dims = gaussian_kl(out["post_mu"], out["post_log_sigma"],
                          out["prior_mu"], out["prior_log_sigma"])             # (B, T-1, Z)
    kl_dims = kl_dims * valid[..., None]                                       # padding contributes 0
    kl = kl_dims.sum(-1).sum() / n                                             # scalar, nats/hour
    loss = recon + BETA_KL * torch.clamp(kl, min=FREE_BITS)                    # scalar
    return loss, recon, kl, kl_dims.sum(dim=(0, 1)) / n                        # last: (Z,)


# ---------------------------------------------------------------- training

def run_epoch(model, loader, optimizer=None):
    """One pass over loader. Trains if an optimizer is given, otherwise evaluates
    with a fixed random seed so validation numbers are comparable across epochs."""
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    sums, n = dict(loss=0.0, recon=0.0, kl=0.0), 0
    kl_dims_sum = 0.0
    with torch.random.fork_rng(enabled=not training), torch.set_grad_enabled(training):
        if not training:
            torch.manual_seed(123)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, recon, kl, kl_dims = compute_loss(model(batch))
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            b = len(batch["idx"])
            for k, v in zip(sums, (loss, recon, kl)):
                sums[k] += v.item() * b
            kl_dims_sum = kl_dims_sum + kl_dims.detach() * b
            n += b
    result = {k: v / n for k, v in sums.items()}
    result["kl_dims"] = (kl_dims_sum / n).tolist()
    return result


def _wandb_start(tag, seed, loaders, model_kwargs, extra_config):
    """Start a Weights & Biases run only if WANDB_API_KEY is set; otherwise None.
    The key is read by wandb from the environment and never printed or stored here."""
    global WANDB_RUN
    if not os.environ.get("WANDB_API_KEY"):
        return None
    import wandb
    model_defaults = {k: p.default for k, p in inspect.signature(WorldModel.__init__).parameters.items()
                      if p.default is not inspect.Parameter.empty}
    config = dict(run=tag, seed=seed, beta_kl=BETA_KL, free_bits=FREE_BITS, lr=LR,
                  weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
                  patience=PATIENCE, time_budget_s=TIME_BUDGET_S, device=DEVICE,
                  **{**model_defaults, **(model_kwargs or {})},
                  **{f"n_{k}_stays": len(dl.dataset) for k, dl in loaders.items()},
                  **(extra_config or {}))
    WANDB_RUN = wandb.init(project="icu-world-model", name=tag, config=config)
    return WANDB_RUN


def finish_wandb(summary=None):
    """Write final numbers into the W&B run summary and close it (no-op without W&B)."""
    global WANDB_RUN
    if WANDB_RUN is None:
        return
    if summary:
        WANDB_RUN.summary.update(summary)
    WANDB_RUN.finish()
    WANDB_RUN = None


def train(name, seed=SEED, loaders=None, model_kwargs=None, wandb_config=None):
    """Train on dataset `name`. seed changes the initial weights and batch order;
    the default seed saves to model_{name}.pt, other seeds to model_{name}_s{seed}.pt.
    loaders / model_kwargs let other data (Stage 7) reuse the same loop.
    The best checkpoint is written every time validation improves, so a run stopped by
    TIME_BUDGET_S (or killed) still leaves the best model on disk."""
    tag = name if seed == SEED else f"{name}_s{seed}"
    torch.manual_seed(seed)
    if loaders is None:
        loaders, _ = make_loaders(name, batch_size=BATCH_SIZE)
    model = WorldModel(**(model_kwargs or {})).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history, best_val, best_state, bad_epochs = [], float("inf"), None, 0
    run = _wandb_start(tag, seed, loaders, model_kwargs, wandb_config)
    start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr = run_epoch(model, loaders["train"], optimizer)
        va = run_epoch(model, loaders["val"])
        seconds = time.time() - t0
        is_best = va["loss"] < best_val - 1e-4
        history.append(dict(epoch=epoch, train=tr, val=va, seconds=seconds))
        if is_best:
            best_val, best_state, bad_epochs = va["loss"], copy.deepcopy(model.state_dict()), 0
            torch.save(best_state, OUT / f"model_{tag}.pt")
        else:
            bad_epochs += 1
        (OUT / f"history_{tag}.json").write_text(json.dumps(history))
        print(f"  [{tag}] epoch {epoch:3d}  train loss {tr['loss']:.3f}  "
              f"val loss {va['loss']:.3f} (recon {va['recon']:.3f}, KL {va['kl']:.3f})  "
              f"{seconds:.0f}s" + ("  *best" if is_best else ""), flush=True)
        if run is not None:
            run.log(dict(epoch=epoch, train_loss=tr["loss"], val_loss=va["loss"],
                         val_recon=va["recon"], val_kl=va["kl"], seconds_per_epoch=seconds,
                         lr=optimizer.param_groups[0]["lr"], is_best=int(is_best),
                         **{f"val_kl_dim_{i}": k for i, k in enumerate(va["kl_dims"])}), step=epoch)
        if bad_epochs >= PATIENCE:
            print(f"  [{tag}] early stop at epoch {epoch}; best epoch {epoch - PATIENCE}")
            break
        if TIME_BUDGET_S and time.time() - start + seconds > TIME_BUDGET_S:
            print(f"  [{tag}] time budget of {TIME_BUDGET_S / 3600:.1f} h reached after epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, history, loaders


# ---------------------------------------------------------------- one-step-ahead prediction

@torch.no_grad()
def predict_one_step(model, batch, n_samples=200):
    """Honest one-step-ahead prediction: z_{t+1} from the PRIOR (no peeking at x_{t+1}),
    then vitals sampled from the decoder. Returns quantiles (3, B, T-1, V) in z-units
    (10%, 50%, 90%) and the posterior-mean reconstruction (B, T-1, V) for comparison."""
    model.eval()
    out = model(batch)
    h_now = out["h"][:, :-1]                                                    # (B, T-1, D)
    mu, ls = out["prior_mu"], out["prior_log_sigma"]                            # (B, T-1, Z)
    z = mu + ls.exp() * torch.randn(n_samples, *mu.shape)                       # (S, B, T-1, Z)
    h_rep = h_now.expand(n_samples, *h_now.shape)                               # (S, B, T-1, D)
    dec_mu, dec_ls = model.decoder(torch.cat([z, h_rep], -1))                   # (S, B, T-1, V)
    x_samples = dec_mu + dec_ls.exp() * torch.randn_like(dec_mu)                # (S, B, T-1, V)
    q = torch.quantile(x_samples, torch.tensor([0.1, 0.5, 0.9]), dim=0)         # (3, B, T-1, V)
    recon_mu, _ = model.decoder(torch.cat([out["post_mu"], h_now], -1))         # (B, T-1, V)
    return q, recon_mu


def one_step_map_error(model, loader, stats):
    """Mean absolute error (mmHg) of the prior's median MAP vs observed MAP, and
    how often the observed MAP falls inside the 10-90% band, on a whole split."""
    errs, inside = [], []
    for batch in loader:
        q, _ = predict_one_step(model, batch, n_samples=100)
        obs = batch["mask"][:, 1:, 0].bool()                                    # (B, T-1)
        x = batch["x"][:, 1:, 0]                                                # (B, T-1)
        errs.append((q[1, ..., 0] - x)[obs].abs() * stats["std"][0])
        inside.append(((x >= q[0, ..., 0]) & (x <= q[2, ..., 0]))[obs].float())
    return torch.cat(errs).mean().item(), torch.cat(inside).mean().item()


# ---------------------------------------------------------------- plots

def plot_curves(histories):
    fig, axes = plt.subplots(len(histories), 3, figsize=(14, 3.5 * len(histories)),
                             sharex="row", squeeze=False)
    for row, (name, hist) in enumerate(histories.items()):
        ep = [h["epoch"] for h in hist]
        for col, key in enumerate(["loss", "recon", "kl"]):
            ax = axes[row, col]
            ax.plot(ep, [h["train"][key] for h in hist], label="train")
            ax.plot(ep, [h["val"][key] for h in hist], label="val")
            if key == "kl":
                ax.axhline(FREE_BITS, ls="--", color="0.5", lw=0.8, label="free bits")
            ax.set_title(f"{name}: {dict(loss='total loss', recon='reconstruction NLL', kl='KL')[key]}")
            ax.set_xlabel("epoch")
            ax.set_ylabel("nats per hour")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "train_curves.png", dpi=110)
    plt.close(fig)


def plot_kl_dims(kl_dims):
    fig, axes = plt.subplots(1, len(kl_dims), figsize=(5 * len(kl_dims), 3.5), sharey=True,
                             squeeze=False)
    axes = axes[0]
    for ax, (name, kd) in zip(axes, kl_dims.items()):
        ax.bar(range(len(kd)), kd, color=["tab:blue" if k > USED_DIM_NATS else "0.7" for k in kd])
        ax.axhline(USED_DIM_NATS, ls="--", color="tab:red", lw=0.8, label=f"{USED_DIM_NATS} nats")
        ax.set_title(f"{name}: val KL per latent dim ({sum(k > USED_DIM_NATS for k in kd)}/8 used)")
        ax.set_xlabel("latent dim")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("nats per hour")
    fig.tight_layout()
    fig.savefig(FIG / "kl_per_dim.png", dpi=110)
    plt.close(fig)


def plot_one_patient(models, stats):
    """One validation patient per dataset: observed MAP vs one-step-ahead prediction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax, (name, model) in zip(axes, models.items()):
        val = load_splits(name)[0]["val"]
        u_hours = val.a[:, :, 0].sum(1)                                         # (N_val,)
        i = int((u_hours - 12).abs().argmin())                                  # a patient with ~12 h of vasopressors
        batch = dict(x=val.x[i:i + 1], mask=val.mask[i:i + 1], delta=val.delta[i:i + 1],
                     a=val.a[i:i + 1], idx=val.idx[i:i + 1])                    # B = 1
        q, recon = predict_one_step(model, batch)
        to_mmhg = lambda z: z * stats["std"][0] + stats["mean"][0]              # noqa: E731
        hours = np.arange(1, 48)
        true_m = np.load(OUT / f"sim_{name}.npz")["hidden_m"][int(val.idx[i])]
        obs = val.mask[i, :, 0].bool().numpy()
        for t in np.where(val.a[i, :, 0].numpy() == 1)[0]:
            ax.axvspan(t + 0.5, t + 1.5, color="tab:purple", alpha=0.18, lw=0)
        ax.fill_between(hours, to_mmhg(q[0, 0, :, 0]), to_mmhg(q[2, 0, :, 0]),
                        color="tab:orange", alpha=0.3, label="prior one-step 10-90%")
        ax.plot(hours, to_mmhg(q[1, 0, :, 0]), color="tab:orange", label="prior one-step median")
        ax.plot(hours, to_mmhg(recon[0, :, 0]), ls=":", color="tab:green",
                label="posterior reconstruction (sees x)")
        ax.plot(np.arange(48), true_m, color="0.4", lw=1, label="true MAP (hidden)")
        ax.plot(np.arange(48)[obs], to_mmhg(val.x[i, obs, 0]), ".", color="tab:red", label="MAP_obs")
        ax.set_title(f"{name} model, val patient {int(val.idx[i])} "
                     f"(purple = vasopressor given the hour before)")
        ax.set_xlabel("hour")
        ax.set_ylabel("MAP (mmHg)")
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "one_step_val_patient.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    models, histories, kl_dims, stats = {}, {}, {}, None
    for name in ["policy", "explore"]:
        print(f"== training on {name} data")
        model, history, loaders = train(name)
        models[name], histories[name] = model, history
        best = min(history, key=lambda h: h["val"]["loss"])
        kl_dims[name] = best["val"]["kl_dims"]
        stats = load_splits(name)[1]
        mae, cover = one_step_map_error(model, loaders["val"], stats)
        used = sum(k > USED_DIM_NATS for k in kl_dims[name])
        print(f"  [{name}] best epoch {best['epoch']}: val recon {best['val']['recon']:.3f}, "
              f"KL {best['val']['kl']:.3f} nats/hour, latent dims used {used}/8")
        print(f"  [{name}] val one-step MAP: MAE {mae:.2f} mmHg, observed inside 10-90% band {cover:.0%}")
        print("  KL per dim: " + " ".join(f"{k:.3f}" for k in kl_dims[name]))
        if best["val"]["kl"] < 0.1 or used == 0:
            print(f"  WARNING [{name}]: KL is near zero -> posterior collapse; z carries no information")
    plot_curves(histories)
    plot_kl_dims(kl_dims)
    plot_one_patient(models, stats)
    print(f"figures saved under {FIG}")
