"""
model_jepa_b.py - Stage 14b: the JEPA model of Stage 14 with the action leak closed.

One change only: the target encoder builds its tokens WITHOUT any action input
(target act_proj not applied), so the target embedding of hour t+1 is made from the vitals,
masks, time gaps and hour alone. Everything else (online encoder with its action term,
predictor, prior, posterior, loss, probe) is model_jepa.JEPAWorldModel unchanged.

The target act_proj module is kept (and still EMA-updated) only so the checkpoint layout is the
same as Stage 14's; it is never used.
New file; model_jepa.py and model.py are only imported.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_jepa as MJ   # noqa: E402


class JEPANoActTarget(MJ.JEPAWorldModel):
    """JEPAWorldModel whose target embedding sees no action."""

    @torch.no_grad()
    def target_embed(self, x, mask, delta, a_prev, hours):
        tg = self.target
        tokens = tg["obs_proj"](torch.cat([x, mask, delta], -1)) + tg["time_emb"](hours)   # no action term
        L = tokens.shape[1]
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=tokens.device)
        was = tg["transformer"].training
        tg["transformer"].eval()
        out = tg["transformer"](tokens, mask=causal, is_causal=True)
        tg["transformer"].train(was)
        return out


def load_jepa_b(path, n_vitals=5, n_actions=3, device="cpu"):
    state = torch.load(path, map_location="cpu")
    model = JEPANoActTarget(n_vitals=n_vitals, n_actions=n_actions)
    model.load_state_dict(state)
    return model.to(device).eval()


def load_jepa_b_for_eval(model_path, probe_path, n_vitals=5, n_actions=3, device="cpu"):
    model = load_jepa_b(model_path, n_vitals, n_actions, "cpu")
    probe = MJ.ProbeHead(model.d, model.d, n_vitals)
    probe.load_state_dict(torch.load(probe_path, map_location="cpu"))
    return MJ.JEPAForEval(model, probe).to(device).eval()
