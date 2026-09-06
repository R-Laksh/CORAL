from types import SimpleNamespace

import torch

from coral.evaluation import edit_regret, exact_min_edits
from coral.models.biological import (
    BPNetCountScore, ConjunctiveConstraint, ESMAlphabet,
    ESMSoftSequenceRegressor, FairESMSoftSequenceRegressor, ThresholdConstraint,
)


class TinyBPNet(torch.nn.Module):
    def forward(self, x, ctl=None):
        return x[:, :2], (2 * x[:, 0] + x[:, 2]).mean(dim=-1, keepdim=True)


def test_bpnet_adapter_preserves_input_gradient():
    x = torch.softmax(torch.randn(3, 7, 4), dim=-1).requires_grad_(True)
    BPNetCountScore(TinyBPNet())(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_threshold_and_conjunction_semantics():
    class Score(torch.nn.Module):
        def forward(self, x): return x[:, 0, 1]
    inc = ThresholdConstraint(Score(), 0.8, "increase")
    dec = ThresholdConstraint(Score(), 0.9, "decrease")
    both = ConjunctiveConstraint([inc, dec])
    x = torch.tensor([[[0.1, 0.9]], [[0.9, 0.1]]])
    assert inc(x)[0] <= 0 and inc(x)[1] > 0 and dec(x)[0] <= 0
    assert torch.allclose(both(x), torch.maximum(inc(x), dec(x)))


class TinyBackbone(torch.nn.Module):
    def __init__(self, vocab=25, d=6):
        super().__init__(); self.emb = torch.nn.Embedding(vocab, d); self.proj = torch.nn.Linear(d, d, bias=False)
    def get_input_embeddings(self): return self.emb
    def forward(self, inputs_embeds, attention_mask=None, return_dict=True):
        return SimpleNamespace(last_hidden_state=self.proj(inputs_embeds))


def test_esm_soft_adapter_has_gradient_and_freezes_backbone():
    backbone = TinyBackbone(); head = torch.nn.Linear(6, 1)
    model = ESMSoftSequenceRegressor(backbone, ESMAlphabet(tuple(range(20)), 20, 21), head)
    x = torch.softmax(torch.randn(2, 5, 20), dim=-1).requires_grad_(True)
    model(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert all(not p.requires_grad for p in backbone.parameters()) and any(p.requires_grad for p in head.parameters())


class TinyFairLayer(torch.nn.Module):
    def __init__(self, d): super().__init__(); self.proj = torch.nn.Linear(d, d, bias=False)
    def forward(self, x, self_attn_padding_mask=None, need_head_weights=False): return self.proj(x), None


class TinyFairESM(torch.nn.Module):
    def __init__(self, vocab=25, d=6):
        super().__init__(); self.embed_tokens = torch.nn.Embedding(vocab, d); self.embed_scale = 1.0
        self.token_dropout = True; self.cls_idx = 20; self.eos_idx = 21
        self.layers = torch.nn.ModuleList([TinyFairLayer(d), TinyFairLayer(d)])
        self.emb_layer_norm_after = torch.nn.LayerNorm(d)


def test_fair_esm_soft_adapter_has_gradient_and_freezes_backbone():
    backbone = TinyFairESM(); head = torch.nn.Linear(6, 1)
    model = FairESMSoftSequenceRegressor(backbone, tuple(range(20)), head)
    x = torch.softmax(torch.randn(2, 5, 20), dim=-1).requires_grad_(True)
    model(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert all(not p.requires_grad for p in backbone.parameters()) and any(p.requires_grad for p in head.parameters())


def test_exact_edit_regret_uses_measured_endpoints_only():
    seqs = ["VDGV", "IDGV", "IAGV", "IAGA", "AAAA"]
    scores = [1.0, 0.5, 1.2, 3.0, 4.0]
    assert exact_min_edits("VDGV", seqs, scores, 2.0) == 3
    r = edit_regret("VDGV", "AAAA", seqs, scores, 2.0)
    assert r.optimal_edits == 3 and r.achieved_edits == 4 and r.regret == 1
