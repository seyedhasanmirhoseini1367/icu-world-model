"""
model.py - Stage 3: action-conditioned transformer world model (TSSM-style).

Shape letters: B batch, T hours (48), L any sequence length, V vitals (2),
A action dims (2), D model width (64), Z latent dims (8).

One step, from hour t to hour t+1:
    token_t   = obs_proj([x_t, mask_t, delta_t]) + act_proj(a_{t-1}) + time_emb(t)
    h_t       = causal transformer(token_0 .. token_t)     summary of everything up to hour t
    prior     p(z_{t+1} | h_t, a_t)                        "what comes next if I do a_t"
    posterior q(z_{t+1} | h_t, a_t, x_{t+1})               the same, but allowed to see x_{t+1}
    decoder   p(x_{t+1} | z_{t+1}, h_t)                    turns the latent into vitals

The decoder never sees a_t, so the chosen action can only change the predicted
vitals through z_{t+1}. In training, z is sampled from the posterior and the
prior is pulled towards it (KL term, Stage 4). In imagination (Stage 5) there is
no x_{t+1}, so only the prior is used.

The posterior also gets mask_{t+1}: missing vitals are stored as 0, and without
the mask it could not tell "missing" from "exactly the mean".

Run from the project root:  python src/model.py
"""
import torch
import torch.nn as nn

LOG_SIGMA_MIN, LOG_SIGMA_MAX = -5.0, 3.0   # keeps exp(log_sigma) in a safe range


class GaussianHead(nn.Module):
    """Small MLP that outputs the mean and log standard deviation of a diagonal Gaussian."""

    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, 2 * d_out))

    def forward(self, inp):                                    # inp (..., d_in)
        mu, log_sigma = self.net(inp).chunk(2, dim=-1)         # each (..., d_out)
        return mu, log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)


class WorldModel(nn.Module):

    def __init__(self, n_vitals=2, n_actions=2, d=64, n_layers=2, n_heads=4,
                 z_dim=8, max_hours=64, dropout=0.1):
        super().__init__()
        self.obs_proj = nn.Linear(3 * n_vitals, d)       # [x, mask, delta] (3V) -> D
        self.act_proj = nn.Linear(n_actions, d)          # a_{t-1} (A) -> D
        self.time_emb = nn.Embedding(max_hours, d)       # hour index -> D (learned)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=4 * d, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d),
                                                 enable_nested_tensor=False)
        self.prior = GaussianHead(d + n_actions, d, z_dim)                     # [h, a] -> z
        self.posterior = GaussianHead(d + n_actions + 2 * n_vitals, d, z_dim)  # [h, a, x', mask'] -> z
        self.decoder = GaussianHead(z_dim + d, d, n_vitals)                    # [z, h] -> x'

    # ---- building blocks (Stage 5 calls these one hour at a time) ----

    def embed(self, x, mask, delta, a_prev, hours):
        """x, mask, delta (B, L, V); a_prev (B, L, A); hours (L,) long -> tokens (B, L, D)."""
        obs = torch.cat([x, mask, delta], dim=-1)                    # (B, L, 3V)
        return self.obs_proj(obs) + self.act_proj(a_prev) + self.time_emb(hours)  # (B, L, D)

    def encode(self, tokens):
        """tokens (B, L, D) -> h (B, L, D); h_t only attends to tokens 0..t."""
        L = tokens.shape[1]
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=tokens.device)  # (L, L), -inf above diagonal
        return self.transformer(tokens, mask=causal, is_causal=True)  # (B, L, D)

    @staticmethod
    def sample(mu, log_sigma):
        """Reparameterised sample: mu + sigma * noise, so gradients flow into mu and sigma."""
        return mu + log_sigma.exp() * torch.randn_like(mu)          # same shape as mu

    # ---- full teacher-forced pass used in training ----

    def forward(self, batch, show=False):
        x, mask, delta, a = batch["x"], batch["mask"], batch["delta"], batch["a"]  # each (B, T, 2)
        B, T, _ = x.shape
        a_prev = torch.cat([torch.zeros_like(a[:, :1]), a[:, :-1]], dim=1)  # (B, T, A): a_{t-1}, zeros at t=0
        hours = torch.arange(T, device=x.device)                             # (T,)

        tokens = self.embed(x, mask, delta, a_prev, hours)                   # (B, T, D)
        h = self.encode(tokens)                                              # (B, T, D)

        # Predict hour t+1 from hour t, for t = 0 .. T-2 (the last hour has no next hour).
        h_now, a_now = h[:, :-1], a[:, :-1]                                  # (B, T-1, D), (B, T-1, A)
        x_next, mask_next = x[:, 1:], mask[:, 1:]                            # (B, T-1, V) each

        prior_mu, prior_ls = self.prior(torch.cat([h_now, a_now], -1))       # (B, T-1, Z) each
        post_mu, post_ls = self.posterior(
            torch.cat([h_now, a_now, x_next, mask_next], -1))               # (B, T-1, Z) each
        z = self.sample(post_mu, post_ls)                                    # (B, T-1, Z)
        dec_mu, dec_ls = self.decoder(torch.cat([z, h_now], -1))             # (B, T-1, V) each

        out = dict(h=h, prior_mu=prior_mu, prior_log_sigma=prior_ls,
                   post_mu=post_mu, post_log_sigma=post_ls, z=z,
                   dec_mu=dec_mu, dec_log_sigma=dec_ls,
                   x_target=x_next, mask_target=mask_next)
        if show:
            rows = [("x, mask, delta, a  (input)", x), ("a_prev = a shifted right", a_prev),
                    ("hours", hours),
                    ("obs_proj input [x, mask, delta]", torch.cat([x, mask, delta], -1)),
                    ("tokens = obs_proj + act_proj + time_emb", tokens), ("h = encode(tokens)", h),
                    ("h_now = h[:, :-1]", h_now), ("a_now = a[:, :-1]", a_now),
                    ("x_next = x[:, 1:]", x_next),
                    ("prior input [h_now, a_now]", torch.cat([h_now, a_now], -1)),
                    ("prior mu / log_sigma", prior_mu),
                    ("posterior input [h_now, a_now, x_next, mask_next]",
                     torch.cat([h_now, a_now, x_next, mask_next], -1)),
                    ("posterior mu / log_sigma", post_mu), ("z ~ posterior", z),
                    ("decoder input [z, h_now]", torch.cat([z, h_now], -1)),
                    ("decoder mu / log_sigma", dec_mu)]
            for name, t in rows:
                print(f"  {name:<52} {tuple(t.shape)}")
        return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import make_loaders

    torch.manual_seed(0)
    loaders, _ = make_loaders("policy", batch_size=16)
    batch = next(iter(loaders["train"]))
    model = WorldModel()
    print(f"WorldModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    for name, part in [("obs_proj", model.obs_proj), ("act_proj", model.act_proj),
                       ("time_emb", model.time_emb), ("transformer", model.transformer),
                       ("prior", model.prior), ("posterior", model.posterior),
                       ("decoder", model.decoder)]:
        print(f"  {name:<12} {sum(p.numel() for p in part.parameters()):>7,}")

    print("\nForward pass on one real batch (B=16 patients, T=48 hours):")
    model.eval()                                          # no dropout, so the checks below are exact
    with torch.no_grad():
        out = model(batch, show=True)

        # Check 1: changing x at hour 30 must not change h at hours 0..29.
        b2 = {k: v.clone() for k, v in batch.items()}
        b2["x"][:, 30] += 5.0
        h2 = model(b2)["h"]
        past = torch.allclose(out["h"][:, :30], h2[:, :30], atol=1e-5)
        future = not torch.allclose(out["h"][:, 30:], h2[:, 30:], atol=1e-5)
        print(f"\nCausality: h before hour 30 unchanged {past}, h from hour 30 changed {future}")

        # Check 2: changing the action at hour 30 changes the prior for hour 31,
        # but not h at hour 30 (a_30 only enters the token of hour 31).
        b3 = {k: v.clone() for k, v in batch.items()}
        b3["a"][:, 30, 0] = 1.0 - b3["a"][:, 30, 0]
        out3 = model(b3)
        h_same = torch.allclose(out["h"][:, :31], out3["h"][:, :31], atol=1e-5)
        prior_moves = not torch.allclose(out["prior_mu"][:, 30], out3["prior_mu"][:, 30], atol=1e-5)
        print(f"Action:    h up to hour 30 unchanged {h_same}, prior for hour 31 changed {prior_moves}")
