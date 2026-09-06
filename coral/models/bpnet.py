"""Differentiable ENCODE BPNet counts with verified zero-control semantics."""
import h5py
import numpy as np
import torch
from torch import nn

MODEL_ID = "kundajelab/encode-bpnet-FOXA1-ChIP-seq-HepG2-ENCSR865RXA-ENCSR337KST"
MODEL_REVISION = "adde8fa27ceb7647e35aebe456620ea6b40d4ec3"
MODEL_SHA256 = "c347f6885eec87076df3669dc22a75239f2391982d0ec5dba9f9dd0d48833f3b"
BPNET_LITE_REVISION = "b37e766bd7a2bef1614cf18d8bac38167e6f6ff5"


class ZeroControlBPNet(nn.Module):
    """2114bp -> one total-logcount, for the pinned basepairmodels H5 layout.

    Fixes the conversion of the profile-head bias through the final 1x1 mixer.
    Upstream BasePairNet adds the two biases without applying the mixer. This
    checkpoint has a nonidentity mixer, giving a strand-specific logit offset.

    Zero *log-count* control inputs in the original TensorFlow graph become
    logsumexp(0, 0) = log(2), matching BasePairNet's log(sum(raw_control) + 2).
    This wrapper deliberately exposes only that verified control convention.
    """
    def __init__(self, checkpoint, device="cpu"):
        super().__init__()
        from bpnetlite.bpnet import BasePairNet
        self.model = BasePairNet.from_bpnet(checkpoint)
        with h5py.File(checkpoint, "r") as h5:
            weights = h5["model_weights"]
            read = lambda layer, name: weights[f"{layer}/{layer}/{name}:0"][:]
            bias = read("main_profile_head", "bias")
            mixer = read("profile_predictions", "kernel")[0, :len(bias)]
            corrected = bias @ mixer + read("profile_predictions", "bias")
        with torch.no_grad():
            self.model.fconv.bias.copy_(torch.tensor(corrected))
        self.model.eval().requires_grad_(False)
        self.to(device)

    def full_outputs(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (4, 2114):
            raise ValueError("Expected (batch, 4, 2114) [A,C,G,T]")
        control = torch.zeros((len(x), 2, 2114), dtype=x.dtype, device=x.device)
        return self.model(x, control)

    def forward(self, x):
        return self.full_outputs(x)[1]


def verify_against_tf(model, reference_path):
    """Compare exported TF predictions and eight independent count directions."""
    with np.load(reference_path) as reference:
        x = torch.tensor(reference["x"].transpose(0, 2, 1), requires_grad=True)
        profile, counts = model.full_outputs(x)
        gradient = torch.autograd.grad(counts[2:10].sum(), x)[0][2:10]
        directions = torch.tensor(reference["directions"].transpose(0, 2, 1))
        analytic = (gradient * directions).sum((1, 2)).numpy()
        finite = ((reference["counts"][10:18] - reference["counts"][18:26]) /
                  (2 * float(reference["epsilon"]))).flatten()
        result = {"profile_max_abs_error": float(np.abs(profile.detach().numpy().transpose(0, 2, 1) - reference["profile"]).max()),
                  "counts_max_abs_error": float(np.abs(counts.detach().numpy() - reference["counts"]).max()),
                  "gradient_max_abs_error": float(np.abs(analytic - finite).max()),
                  "torch_directions": analytic.tolist(), "tf_finite_differences": finite.tolist(),
                  "epsilon": float(reference["epsilon"]), "sequences_including_perturbations": len(x)}
    if result["profile_max_abs_error"] > 1e-4 or result["counts_max_abs_error"] > 1e-5:
        raise AssertionError(f"BPNet conversion failed: {result}")
    if result["gradient_max_abs_error"] > .002:
        raise AssertionError(f"BPNet directional-gradient check failed: {result}")
    return result
