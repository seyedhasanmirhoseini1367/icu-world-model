"""
model_jepa.py - Stage 14: a JEPA-style world model (new file; model.py is only imported).

Same data, tokens, causal transformer, act_now path, prior, posterior and KL as the current
default (WorldModel with act_now=True). What changes is what the model predicts:

    VAE (model.py)   decoder p(x_{t+1} | z, h_t)            -> Gaussian NLL of the vitals
    JEPA (this file) predictor [h_t, act_now(a_t), z] -> e_hat_{t+1}
                     target e_{t+1} = target encoder at hour t+1 (stop-gradient)
                     loss = 0.5 * ||e_hat - e||^2 + VICReg variance/covariance + beta * KL

The target encoder is an EMA copy of the token layers (obs_proj, act_proj, time_emb) and the
transformer. It reads the real hours 0..t+1 causally, so its output at hour t+1 embeds that hour
in the context of the stay. It gets no gradient; after every optimizer step its weights move a
little towards the online ones (exponential moving average, momentum EMA_MOMENTUM).

The posterior still sees the raw x_{t+1} and mask_{t+1}, exactly as in the VAE, so z has the same
meaning: what the next hour adds beyond h_t and the action.

0.5 * squared error summed over the D embedding dims is the Gaussian NLL with unit variance
(up to a constant), so beta = 1 keeps its VAE meaning: one nat in z must buy one nat of
prediction loss.

Vitals are read out by a probe trained afterwards on the frozen model (ProbeHead):
predicted embedding -> mean and log sigma of the 5 vitals. JEPAForEval wraps model + probe in
the interface the shared evaluation code expects (embed, encode, prior_from, sample, decoder,
forward), so Stages 7-12b are measured by the same code as the VAE.
"""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GaussianHead, WorldModel   # noqa: E402

EMA_MOMENTUM = 0.996
VAR_WEIGHT, COV_WEIGHT = 25.0, 1.0          # VICReg-style weights (VICReg paper defaults: 25, 1)
VAR_TARGET_STD = 1.0
FIXED_LOG_SIGMA = -30.0                     # "no noise" for the action part of the eval latent


class JEPAWorldModel(WorldModel):
    """WorldModel without a decoder, plus an EMA target encoder and a predictor."""

    def __init__(self, n_vitals=5, n_actions=3, d=64, n_layers=2, n_heads=4, z_dim=8, max_hours=64,
                 dropout=0.1, act_now=True, act_delta=False):
        if not act_now:
            raise ValueError("the JEPA model uses the act_now path (current default)")
        super().__init__(n_vitals, n_actions, d, n_layers, n_heads, z_dim, max_hours, dropout,
                         act_now=True, act_delta=act_delta)
        del self.decoder                                                  # no reconstruction of x
        self.d, self.z_dim, self.n_vitals = d, z_dim, n_vitals
        self.predictor = nn.Sequential(nn.Linear(2 * d + z_dim, 2 * d), nn.GELU(),
                                       nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        # Target encoder: EMA copies of the token layers and the transformer, never trained by gradient.
        self.target = nn.ModuleDict(dict(obs_proj=copy.deepcopy(self.obs_proj), act_proj=copy.deepcopy(self.act_proj),
                                         time_emb=copy.deepcopy(self.time_emb),
                                         transformer=copy.deepcopy(self.transformer)))
        for p in self.target.parameters():
            p.requires_grad_(False)

    # ---- target encoder

    def _online_target_pairs(self):
        for name in ("obs_proj", "act_proj", "time_emb", "transformer"):
            yield getattr(self, name), self.target[name]

    @torch.no_grad()
    def ema_update(self, momentum=EMA_MOMENTUM):
        for online, target in self._online_target_pairs():
            for p_o, p_t in zip(online.parameters(), target.parameters()):
                p_t.mul_(momentum).add_(p_o.detach(), alpha=1.0 - momentum)

    @torch.no_grad()
    def target_embed(self, x, mask, delta, a_prev, hours):
        """Same inputs as embed + encode -> target embeddings (B, L, D), causal, no gradient.
        Dropout is off in the target encoder (it is a fixed teacher)."""
        tg = self.target
        tokens = tg["obs_proj"](torch.cat([x, mask, delta], -1)) + tg["act_proj"](a_prev) + tg["time_emb"](hours)
        L = tokens.shape[1]
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=tokens.device)
        was = tg["transformer"].training
        tg["transformer"].eval()
        out = tg["transformer"](tokens, mask=causal, is_causal=True)
        tg["transformer"].train(was)
        return out

    # ---- predictor

    def predict_embedding(self, h, act_emb, z):
        """h (..., D), act_emb = action_input(a) (..., D), z (..., Z) -> predicted embedding (..., D)."""
        return self.predictor(torch.cat([h, act_emb, z], -1))

    # ---- full teacher-forced pass

    def forward(self, batch, show=False):
        x, mask, delta, a = batch["x"], batch["mask"], batch["delta"], batch["a"]
        B, T, _ = x.shape
        a_prev = torch.cat([torch.zeros_like(a[:, :1]), a[:, :-1]], dim=1)          # (B, T, A)
        hours = torch.arange(T, device=x.device)
        h = self.encode(self.embed(x, mask, delta, a_prev, hours))                    # (B, T, D)
        h_now, a_now, a_before = h[:, :-1], a[:, :-1], a_prev[:, :-1]
        act_emb = self.action_input(a_now, a_before)                                  # (B, T-1, D)
        prior_mu, prior_ls = self.prior(torch.cat([h_now, act_emb], -1))
        post_mu, post_ls = self.posterior(torch.cat([h_now, act_emb, x[:, 1:], mask[:, 1:]], -1))
        z = self.sample(post_mu, post_ls)
        pred = self.predict_embedding(h_now, act_emb, z)                              # (B, T-1, D)
        target = self.target_embed(x, mask, delta, a_prev, hours)[:, 1:]             # (B, T-1, D)
        valid = batch.get("valid", torch.ones(B, T, device=x.device))
        return dict(h=h, act_emb=act_emb, prior_mu=prior_mu, prior_log_sigma=prior_ls, post_mu=post_mu,
                    post_log_sigma=post_ls, z=z, pred=pred, target=target, x_target=x[:, 1:],
                    mask_target=mask[:, 1:], valid_target=valid[:, 1:])


# ---------------------------------------------------------------- loss pieces

def variance_term(e):
    """e (N, D) -> mean over dims of relu(1 - std): pushes every dim to spread across the batch."""
    std = torch.sqrt(e.var(0) + 1e-4)
    return torch.relu(VAR_TARGET_STD - std).mean()


def covariance_term(e):
    """e (N, D) -> sum of squared off-diagonal covariances / D: decorrelates the dims."""
    n, d = e.shape
    e = e - e.mean(0)
    cov = (e.T @ e) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / d


def embedding_std(e):
    """e (N, D) -> per-dimension std (D,), for the collapse check."""
    return e.float().std(0)


# ---------------------------------------------------------------- probe and evaluation adapter

class ProbeHead(GaussianHead):
    """Predicted embedding (D) -> mean and log sigma of the vitals (V), z-units. Same small MLP
    as the VAE decoder head (Linear(D, D) -> GELU -> Linear(D, 2V)), trained on the frozen model."""


class JEPAForEval(nn.Module):
    """Model + probe behind the interface of WorldModel that the shared evaluators use.

    The evaluators pass a latent through prior_from -> sample -> decoder(cat[z, h]). The JEPA
    predictor also needs the current action, so the latent seen by the evaluators is
    [z, act_now(a_t)]: the action part has a fixed log sigma of -30 (no noise) in prior and
    posterior alike, so sampling leaves it unchanged and its KL is exactly 0. decoder() splits
    it again and returns probe(predictor(h, act_now(a_t), z))."""

    def __init__(self, model, probe):
        super().__init__()
        self.model, self.probe = model, probe
        self.z_dim, self.d = model.z_dim, model.d
        # ablate_mimic reads the latent width as prior.net[-1].out_features // 2
        self.prior = SimpleNamespace(net=[SimpleNamespace(out_features=2 * (self.z_dim + self.d))])

    def embed(self, *args):
        return self.model.embed(*args)

    def encode(self, tokens):
        return self.model.encode(tokens)

    sample = staticmethod(WorldModel.sample)

    def _augment(self, mu, ls, act_emb):
        return torch.cat([mu, act_emb], -1), torch.cat([ls, torch.full_like(act_emb, FIXED_LOG_SIGMA)], -1)

    def prior_from(self, h, a, a_prev=None):
        act_emb = self.model.action_input(a, a_prev)
        mu, ls = self.model.prior(torch.cat([h, act_emb], -1))
        return self._augment(mu, ls, act_emb)

    def decoder(self, inp):
        z, act_emb, h = inp.split([self.z_dim, self.d, self.d], -1)
        return self.probe(self.model.predict_embedding(h, act_emb, z))

    def forward(self, batch, show=False):
        out = self.model(batch)
        out["prior_mu"], out["prior_log_sigma"] = self._augment(out["prior_mu"], out["prior_log_sigma"], out["act_emb"])
        out["post_mu"], out["post_log_sigma"] = self._augment(out["post_mu"], out["post_log_sigma"], out["act_emb"])
        return out


def load_jepa(path, n_vitals=5, n_actions=3, device="cpu"):
    state = torch.load(path, map_location="cpu")
    model = JEPAWorldModel(n_vitals=n_vitals, n_actions=n_actions)
    model.load_state_dict(state)
    return model.to(device).eval()


def load_jepa_for_eval(model_path, probe_path, n_vitals=5, n_actions=3, device="cpu"):
    model = load_jepa(model_path, n_vitals, n_actions, "cpu")
    probe = ProbeHead(model.d, model.d, n_vitals)
    probe.load_state_dict(torch.load(probe_path, map_location="cpu"))
    return JEPAForEval(model, probe).to(device).eval()
