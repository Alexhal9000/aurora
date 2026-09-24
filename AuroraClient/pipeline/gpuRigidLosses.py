"""
Custom FireANTs similarity losses for GPU rigid.

NGF (Haber/Modersitzki): match normalized image gradients — robust when
intensity overlap is poor after centroid paste.

NMI: 2 I / (H_a + H_b) from FireANTs Parzen MI — less bin-count sensitive than raw MI.
"""

from __future__ import annotations

import torch
from torch import nn

from fireants.losses.mi import GlobalMutualInformationLoss


class NormalizedMutualInformationLoss(GlobalMutualInformationLoss):
    """Minimize −NMI with NMI = 2 I(X;Y) / (H(X)+H(Y))."""

    def forward_util_singlechannel(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        maxval = max(pred.max(), target.max())
        if maxval > 1:
            pred = pred / maxval
            target = target / maxval
        wa, pa, wb, pb = self.parzen_windowing(pred, target)
        pab = torch.bmm(wa.permute(0, 2, 1), wb.to(wa)).div(wa.shape[1])
        papb = torch.bmm(pa.permute(0, 2, 1), pb.to(pa))
        mi = torch.sum(
            pab * torch.log((pab + self.smooth_nr) / (papb + self.smooth_dr) + self.smooth_dr),
            dim=(1, 2),
        )
        ha = -torch.sum(pa.squeeze(1) * torch.log(pa.squeeze(1) + self.smooth_nr), dim=1)
        hb = -torch.sum(pb.squeeze(1) * torch.log(pb.squeeze(1) + self.smooth_nr), dim=1)
        nmi = 2.0 * mi / (ha + hb + self.smooth_dr)
        if self.reduction == "sum":
            return torch.sum(nmi).neg()
        if self.reduction == "none":
            return nmi.neg()
        return torch.mean(nmi).neg()


class NormalizedGradientFieldLoss(nn.Module):
    """
    Differentiable NGF: 1 − ⟨n(∇moved), n(∇fixed)⟩² averaged over voxels.

    First channel is intensity; optional last channel is a mask (FireANTs convention).
    """

    def __init__(self, eps: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.eps = float(eps)
        self.reduction = reduction

    def get_image_padding(self) -> int:
        return 0

    def _intensity_and_mask(self, moved: torch.Tensor, fixed: torch.Tensor):
        if moved.shape[1] > 1:
            return moved[:, :1], fixed[:, :1], fixed[:, -1:]
        return moved, fixed, None

    def _normalized_grad(self, image: torch.Tensor) -> torch.Tensor:
        grads = torch.gradient(image, dim=tuple(range(2, image.ndim)))
        stacked = torch.cat(grads, dim=1)
        norm = torch.sqrt((stacked * stacked).sum(dim=1, keepdim=True) + self.eps * self.eps)
        return stacked / norm

    def forward(self, moved: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        moved_i, fixed_i, mask = self._intensity_and_mask(moved, fixed)
        n_m = self._normalized_grad(moved_i)
        n_f = self._normalized_grad(fixed_i)
        alignment = (n_m * n_f).sum(dim=1, keepdim=True)
        residual = 1.0 - alignment * alignment
        if mask is not None:
            w = (mask > 0.5).to(residual.dtype)
            denom = w.sum().clamp_min(1.0)
            return (residual * w).sum() / denom
        if self.reduction == "sum":
            return residual.sum()
        return residual.mean()


class NgfNmiLoss(nn.Module):
    """NGF plus a smaller NMI term (edges first, histogram second)."""

    def __init__(self, ngf_weight: float = 1.0, nmi_weight: float = 0.25, eps: float = 1e-3):
        super().__init__()
        self.ngf_weight = float(ngf_weight)
        self.nmi_weight = float(nmi_weight)
        self.ngf = NormalizedGradientFieldLoss(eps=eps)
        self.nmi = NormalizedMutualInformationLoss(kernel_type="gaussian", num_bins=32)

    def get_image_padding(self) -> int:
        return 0

    def forward(self, moved: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        moved_i, fixed_i, _mask = self.ngf._intensity_and_mask(moved, fixed)
        return self.ngf_weight * self.ngf(moved, fixed) + self.nmi_weight * self.nmi(moved_i, fixed_i)


def build_gpu_rigid_custom_loss(loss_name: str):
    name = str(loss_name or "ngf").lower()
    if name == "nmi":
        return NormalizedMutualInformationLoss(kernel_type="gaussian", num_bins=32)
    if name == "ngf":
        return NormalizedGradientFieldLoss()
    if name in ("ngf_nmi", "nmi_ngf"):
        return NgfNmiLoss()
    return None
