"""Differentiable adapters for frozen biological sequence predictors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

Tensor = torch.Tensor


class BPNetCountScore(torch.nn.Module):
    """Expose a BPNet/ChromBPNet log-count prediction as a scalar score."""

    def __init__(self, model: torch.nn.Module, control: Tensor | None = None):
        super().__init__()
        self.model = model
        if control is not None:
            self.register_buffer("control", control.detach().clone())
        else:
            self.control = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != 4:
            raise ValueError("BPNetCountScore expects (N, L, 4) DNA probabilities")
        x_cf = x.transpose(1, 2)
        ctl = self.control
        if ctl is not None:
            if ctl.ndim == 2:
                ctl = ctl.unsqueeze(0)
            if ctl.shape[0] == 1 and x.shape[0] != 1:
                ctl = ctl.expand(x.shape[0], -1, -1)
        try:
            _, counts = self.model(x_cf, ctl) if ctl is not None else self.model(x_cf)
        except TypeError:
            _, counts = self.model(x_cf)
        return counts.reshape(x.shape[0], -1).mean(dim=-1)


class ThresholdConstraint(torch.nn.Module):
    """Convert a scalar score model into ``g(x) <= 0`` constraints."""

    def __init__(self, score_model: torch.nn.Module, target: float, direction: str = "increase"):
        super().__init__()
        if direction not in {"increase", "decrease"}:
            raise ValueError("direction must be 'increase' or 'decrease'")
        self.score_model = score_model
        self.target = float(target)
        self.direction = direction

    def forward(self, x: Tensor) -> Tensor:
        score = self.score_model(x).reshape(x.shape[0])
        return self.target - score if self.direction == "increase" else score - self.target


class ConjunctiveConstraint(torch.nn.Module):
    """Conjunction of inequality constraints via ``max_i g_i(x)``."""

    def __init__(self, constraints: Sequence[torch.nn.Module]):
        super().__init__()
        if not constraints:
            raise ValueError("at least one constraint is required")
        self.constraints = torch.nn.ModuleList(constraints)

    def forward(self, x: Tensor) -> Tensor:
        vals = [c(x).reshape(x.shape[0]) for c in self.constraints]
        return torch.stack(vals, dim=-1).max(dim=-1).values


@dataclass(frozen=True)
class ESMAlphabet:
    """ESM vocabulary entries corresponding to the 20 amino acids and boundaries."""

    aa_token_ids: tuple[int, ...]
    cls_token_id: int
    eos_token_id: int

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        amino_acids: Iterable[str] = tuple("ACDEFGHIKLMNPQRSTVWY"),
    ) -> "ESMAlphabet":
        aas = tuple(amino_acids)
        ids = tuple(int(tokenizer.convert_tokens_to_ids(a)) for a in aas)
        if len(set(ids)) != len(ids):
            raise ValueError("tokenizer does not map amino acids to unique token IDs")
        cls_id = getattr(tokenizer, "cls_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if cls_id is None or eos_id is None:
            raise ValueError("tokenizer must expose cls_token_id and eos_token_id")
        return cls(ids, int(cls_id), int(eos_id))


class ESMSoftSequenceRegressor(torch.nn.Module):
    """Frozen HuggingFace-style ESM backbone plus a differentiable scalar task head."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        alphabet: ESMAlphabet,
        head: torch.nn.Module,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.alphabet = alphabet
        self.head = head
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != len(self.alphabet.aa_token_ids):
            raise ValueError("ESMSoftSequenceRegressor expects (N, L, alphabet_size)")
        weight = self.backbone.get_input_embeddings().weight
        aa_ids = torch.as_tensor(self.alphabet.aa_token_ids, device=x.device)
        aa_weight = weight.index_select(0, aa_ids).to(dtype=x.dtype)
        residues = x @ aa_weight
        cls = weight[self.alphabet.cls_token_id].to(dtype=x.dtype)[None, None, :]
        eos = weight[self.alphabet.eos_token_id].to(dtype=x.dtype)[None, None, :]
        inputs_embeds = torch.cat(
            [cls.expand(x.shape[0], -1, -1), residues, eos.expand(x.shape[0], -1, -1)], dim=1
        )
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=x.device)
        out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled = out.last_hidden_state[:, 1:-1].mean(dim=1)
        return self.head(pooled).reshape(x.shape[0])


class FairESMSoftSequenceRegressor(torch.nn.Module):
    """Frozen fair-esm ESM2 backbone plus a differentiable scalar task head.

    This follows the embedding-to-transformer path used by ``esm.pretrained.esm2_*``
    directly, allowing relaxed amino-acid probabilities to replace discrete token
    embeddings while leaving the pretrained backbone frozen.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        aa_token_ids: Sequence[int],
        head: torch.nn.Module,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.aa_token_ids = tuple(int(i) for i in aa_token_ids)
        self.head = head
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    @classmethod
    def from_alphabet(
        cls,
        backbone: torch.nn.Module,
        alphabet,
        head: torch.nn.Module,
        amino_acids: Iterable[str] = tuple("ACDEFGHIKLMNPQRSTVWY"),
        freeze_backbone: bool = True,
    ) -> "FairESMSoftSequenceRegressor":
        aa_ids = [int(alphabet.get_idx(aa)) for aa in amino_acids]
        return cls(backbone, aa_ids, head, freeze_backbone=freeze_backbone)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != len(self.aa_token_ids):
            raise ValueError("FairESMSoftSequenceRegressor expects (N, L, alphabet_size)")
        model = self.backbone
        weight = model.embed_tokens.weight
        aa_ids = torch.as_tensor(self.aa_token_ids, dtype=torch.long, device=x.device)
        aa_weight = weight.index_select(0, aa_ids).to(dtype=x.dtype)
        residues = x @ aa_weight
        cls = weight[int(model.cls_idx)].to(dtype=x.dtype)[None, None, :]
        eos = weight[int(model.eos_idx)].to(dtype=x.dtype)[None, None, :]
        hidden = torch.cat(
            [cls.expand(x.shape[0], -1, -1), residues, eos.expand(x.shape[0], -1, -1)],
            dim=1,
        )
        hidden = hidden * float(getattr(model, "embed_scale", 1.0))
        if bool(getattr(model, "token_dropout", False)):
            hidden = hidden * (1.0 - 0.15 * 0.8)
        hidden = hidden.transpose(0, 1)
        for layer in model.layers:
            hidden, _ = layer(hidden, self_attn_padding_mask=None, need_head_weights=False)
        hidden = model.emb_layer_norm_after(hidden).transpose(0, 1)
        pooled = hidden[:, 1:-1].mean(dim=1)
        return self.head(pooled).reshape(x.shape[0])
