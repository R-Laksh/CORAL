"""Frozen pretrained ESM-2 with an input-differentiable functional readout.

The temporary embedding hook preserves Hugging Face's native token-dropout
rescaling and special-token handling. This adapter is for serial batched calls,
not concurrent calls on the same encoder. All sequences have equal length and
contain canonical, unmasked residues. Model weights remain frozen, while input
gradients pass through the whole transformer.
"""
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, EsmModel

from coral.datasets.gb1 import ALPHABET

MODEL_ID = "facebook/esm2_t12_35M_UR50D"
MODEL_REVISION = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"
ESMC_ID = "biohub/ESMC-300M"
ESMC_REVISION = "a59b831785f907e96e6a246b1d142bfb76df31ee"
ESMC_CODE_REVISION = "bf343ba264b650dff7a073643725f9aaa1fdbe8d"


class FrozenESM(nn.Module):
    model_id, model_revision = MODEL_ID, MODEL_REVISION
    def __init__(self, checkpoint, sites=(), device="cpu"):
        super().__init__()
        self.amp = False
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.encoder = EsmModel.from_pretrained(
            checkpoint, add_pooling_layer=False, local_files_only=True).eval()
        self.encoder.requires_grad_(False)
        self.register_buffer("token_map", torch.tensor(
            tokenizer.convert_tokens_to_ids(list(ALPHABET))))
        self.bos, self.eos = tokenizer.cls_token_id, tokenizer.eos_token_id
        expected = [self.bos] + self.token_map.tolist() + [self.eos]
        if tokenizer(ALPHABET)["input_ids"] != expected:
            raise ValueError("Native tokenizer does not match canonical residue mapping")
        self.sites = tuple(int(x) for x in sites)
        self.to(device)

    def tokens(self, ids):
        mapped = self.token_map[ids]
        return torch.cat((torch.full_like(mapped[:, :1], self.bos), mapped,
                          torch.full_like(mapped[:, :1], self.eos)), dim=1)

    def pool(self, hidden):
        residue = hidden[:, 1:-1]
        mean = residue.mean(dim=1)
        if not self.sites:
            return mean
        return torch.cat((mean, residue[:, self.sites].flatten(1)), dim=1)

    def native_features(self, ids):
        tokens = self.tokens(ids)
        with torch.autocast(tokens.device.type, dtype=torch.bfloat16, enabled=self.amp):
            out = self.encoder(input_ids=tokens, attention_mask=torch.ones_like(tokens))
        return self.pool(out.last_hidden_state)

    @property
    def embedding(self):
        return self.encoder.get_input_embeddings()

    def forward(self, probabilities):
        if probabilities.ndim != 3 or probabilities.shape[1] != len(ALPHABET):
            raise ValueError("Expected (batch, 20, length) canonical amino acids")
        tokens = self.tokens(probabilities.detach().argmax(dim=1))
        embedding = self.embedding
        mixed = probabilities.transpose(1, 2) @ embedding.weight[self.token_map]
        edge = embedding(tokens[:, [0, -1]])
        full = torch.cat((edge[:, :1], mixed, edge[:, 1:]), dim=1)
        handle = embedding.register_forward_hook(lambda module, inputs, output: full)
        try:
            with torch.autocast(tokens.device.type, dtype=torch.bfloat16, enabled=self.amp):
                result = self.encoder(input_ids=tokens, attention_mask=torch.ones_like(tokens))
        finally:
            handle.remove()
        return self.pool(result.last_hidden_state)


class FrozenESMC(FrozenESM):
    """Official Biohub ESMC weights and native PyTorch encoder, with autograd."""
    model_id, model_revision = ESMC_ID, ESMC_REVISION

    def __init__(self, checkpoint, sites=(), device="cpu"):
        nn.Module.__init__(self)
        self.amp = False
        from esm.models.esmc import EsmcModel, EsmcTokenizer
        tokenizer = EsmcTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.encoder = EsmcModel.from_pretrained(checkpoint, local_files_only=True).eval()
        self.encoder.requires_grad_(False)
        self.register_buffer("token_map", torch.tensor(
            tokenizer.convert_tokens_to_ids(list(ALPHABET))))
        self.bos, self.eos = tokenizer.cls_token_id, tokenizer.eos_token_id
        expected = [self.bos] + self.token_map.tolist() + [self.eos]
        if tokenizer(ALPHABET)["input_ids"] != expected:
            raise ValueError("Native ESMC tokenizer does not match canonical residue mapping")
        self.sites = tuple(int(x) for x in sites)
        self.to(device)


def load_backbone(family, checkpoint, sites=(), device="cpu"):
    if family not in ("esm2", "esmc"):
        raise ValueError("Expected esm2 or esmc")
    return (FrozenESM if family == "esm2" else FrozenESMC)(checkpoint, sites, device)


class ESMFunctionalPredictor(nn.Module):
    """Train-fitted standardisation + linear head; predicts log1p(GB1 fitness)."""
    def __init__(self, backbone, head_path):
        super().__init__()
        self.backbone = backbone
        with np.load(head_path) as head:
            for key in ("mean", "scale", "coef", "intercept"):
                self.register_buffer(key, torch.tensor(head[key], dtype=torch.float32))
            self.feature_count = len(head["mean"])
        self.to(backbone.token_map.device)
        self.eval()

    def forward(self, probabilities):
        features = self.backbone(probabilities)[:, :self.feature_count]
        return (((features - self.mean) / self.scale) @ self.coef + self.intercept).reshape(-1, 1)
