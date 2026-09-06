from types import SimpleNamespace

import torch

from coral.evaluation import edit_regret, exact_min_edits
from coral.models.biological import (
    BPNetCountScore,
    ConjunctiveConstraint,
    ESMAlphabet,
    ESMSoftSequenceRegressor,
    ThresholdConstraint,
)


class TinyBPNet(torch.nn.Module):
    def forward(self, x, ctl=None):
        # x is N,4,L. Return a differentiable count score.
        profile = x[:, :2]
        count = (2 * x[:, 0] + x[:, 2]).mean(dim=-1, keepdim=True)
        return profile, count


def test_bpnet_adapter_preserves_input_gradient():
    x = torch.softmax(torch.randn(3, 7, 4), dim=-1).requires_grad_(True)
    y = BPNetCountScore(TinyBPNet())(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_threshold_and_conjunction_semantics():
    class Score(torch.nn.Module):
        def forward(self, x):
            return x[:, 0, 1]

    inc = ThresholdConstraint(Score(), target=0.8, direction="increase")
    dec = ThresholdConstraint(Score(), target=0.9, direction="decrease")
    both = ConjunctiveConstraint([inc, dec])
    x = torch.tensor([[[0.1, 0.9]], [[0.9, 0.1]]])
    assert inc(x)[0] <= 0 and inc(x)[1] > 0
    assert dec(x)[0] <= 0
    assert torch.allclose(both(x), torch.maximum(inc(x), dec(x)))


class TinyBackbone(torch.nn.Module):
    def __init__(self, vocab=25, d=6):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, d)
        self.proj = torch.nn.Linear(d, d, bias=False)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, inputs_embeds, attention_mask=None, return_dict=True):
        return SimpleNamespace(last_hidden_state=self.proj(inputs_embeds))


def test_esm_soft_adapter_has_gradient_to_amino_acid_probabilities_and_freezes_backbone():
    backbone = TinyBackbone()
    alphabet = ESMAlphabet(tuple(range(20)), cls_token_id=20, eos_token_id=21)
    head = torch.nn.Linear(6, 1)
    model = ESMSoftSequenceRegressor(backbone, alphabet, head, freeze_backbone=True)
    x = torch.softmax(torch.randn(2, 5, 20), dim=-1).requires_grad_(True)
    model(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert all(not p.requires_grad for p in backbone.parameters())
    assert any(p.requires_grad for p in head.parameters())


def test_exact_edit_regret_uses_measured_endpoints_only():
    seqs = ["VDGV", "IDGV", "IAGV", "IAGA", "AAAA"]
    scores = [1.0, 0.5, 1.2, 3.0, 4.0]
    assert exact_min_edits("VDGV", seqs, scores, target=2.0) == 3
    r = edit_regret("VDGV", "AAAA", seqs, scores, target=2.0)
    assert r.optimal_edits == 3
    assert r.achieved_edits == 4
    assert r.regret == 1
