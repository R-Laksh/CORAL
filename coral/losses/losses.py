import torch
import torch.nn as nn
import torch.nn.functional as F


class HammingTableLoss(nn.Module):
    """Expected base-level Hamming via 6-mer token lookup table."""

    def __init__(self, hamming_table: torch.Tensor):
        super().__init__()
        self.register_buffer("hamming_table", hamming_table)  # (V_pure, V_pure)

    def forward(self, X_hat: torch.Tensor, X_orig: torch.Tensor) -> torch.Tensor:
        orig_local = X_orig.argmax(dim=1)             # (B, L_edit)
        H_rows = self.hamming_table[orig_local]        # (B, L_edit, V_pure)
        P = X_hat.permute(0, 2, 1)                    # (B, L_edit, V_pure)
        return (P * H_rows).sum(dim=-1).sum()


class GCMarginLoss(nn.Module):
    """One-sided margin loss for GC regression."""

    def __init__(self, target_label: int, target_gc: float, squared: bool = True):
        super().__init__()
        self.target_label = int(target_label)
        self.target_gc = target_gc
        self.squared = squared

    def forward(self, pred_gc: torch.Tensor, _unused) -> torch.Tensor:
        s = 1.0 if self.target_label == 1 else -1.0
        viol = F.relu(s * (self.target_gc - pred_gc))
        return viol.square().mean() if self.squared else viol.mean()


class ProbToLogitMarginLoss(nn.Module):
    """Margin loss for binary classifiers operating in logit space.

    Penalises logits that haven't crossed the target probability threshold.
    """

    def __init__(self, target_label: int, margin: float = 0.5,
                 squared: bool = True, eps: float = 1e-6):
        super().__init__()
        self.target_label = int(target_label)
        self.margin = margin
        self.squared = squared
        self.eps = eps

    def forward(self, p: torch.Tensor, _unused) -> torch.Tensor:
        p = p.clamp(self.eps, 1 - self.eps)
        z = torch.logit(p)
        m = torch.logit(
            torch.tensor(self.margin, device=z.device, dtype=z.dtype).clamp(self.eps, 1 - self.eps)
        )
        s = 1.0 if self.target_label == 1 else -1.0
        viol = F.relu(m - s * z)
        return viol.square().mean() if self.squared else viol.mean()
