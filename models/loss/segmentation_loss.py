import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7) -> None:
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # probs = torch.sigmoid(logits)
        probs = logits
        targets = targets.float()

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth + self.eps)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float | None = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 4:
            raise ValueError(f"Expected logits/probs [B,C,H,W], got {tuple(logits.shape)}")

        # Single-channel fallback: BCE-style focal for backward compatibility.
        if logits.size(1) == 1:
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)
            targets = targets.float()

            # If input already looks like probability map, use it directly.
            if torch.all((logits >= 0) & (logits <= 1)):
                probs = logits.clamp(min=self.eps, max=1.0 - self.eps)
                bce = -(targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs))
            else:
                bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

            p_t = torch.exp(-bce)
            if self.alpha is None:
                alpha_t = 1.0
            else:
                alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal = alpha_t * ((1.0 - p_t) ** self.gamma) * bce
            if self.reduction == "mean":
                return focal.mean()
            if self.reduction == "sum":
                return focal.sum()
            return focal

        # Multi-class CE-style focal (works for 2-class mutually exclusive segmentation).
        if targets.dim() == 4:
            targets = targets[:, 0]
        targets = targets.long()

        # If input already looks like probabilities, use it; otherwise apply softmax.
        if torch.all((logits >= 0) & (logits <= 1)):
            probs = logits
        else:
            probs = torch.softmax(logits, dim=1).clamp(min=self.eps, max=1.0)

        idx = targets.unsqueeze(1)  # [B,1,H,W], stays on GPU
        pt = torch.gather(probs, dim=1, index=idx).squeeze(1).clamp(min=self.eps, max=1.0)
        logpt = torch.log(pt)

        if self.alpha is None:
            alpha_t = 1.0
        else:
            alpha_t = torch.where(targets == 1, torch.as_tensor(self.alpha, device=targets.device, dtype=pt.dtype), torch.as_tensor(1.0 - self.alpha, device=targets.device, dtype=pt.dtype))

        focal = -alpha_t * ((1.0 - pt) ** self.gamma) * logpt
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        smooth: float = 1.0,
        eps: float = 1e-7,
        alpha: float | None = 0.0,
        gamma: float = 2.0,
        reduction: str = "mean",
        multi_layer_reduction: str = "sum",
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.multi_layer_reduction = multi_layer_reduction
        if self.multi_layer_reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported multi_layer_reduction: {self.multi_layer_reduction}")

        self.dice_loss = DiceLoss(smooth=smooth, eps=eps)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)

    def _single_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        if logits.dim() != 4:
            raise ValueError(f"Expected logits [B,C,H,W], got {tuple(logits.shape)}")
        if targets.dim() != 4:
            raise ValueError(f"Expected targets [B,1,H,W], got {tuple(targets.shape)}")

        # Backward compatibility: single-channel model output.
        if logits.size(1) == 1:
            dice = self.dice_loss(logits, targets)
            focal = self.focal_loss(logits, targets)
            return self.dice_weight * dice + self.focal_weight * focal

        if logits.size(1) != 2:
            raise ValueError(f"Expected channel size 1 or 2, got {logits.size(1)}")

        normal_target = 1.0 - targets
        anomaly_target = targets
        class_targets = (targets[:, 0] > 0.5).long()

        focal = self.focal_loss(logits, class_targets)
        dice_norm = self.dice_loss(logits[:, 0:1], normal_target)
        dice_anom = self.dice_loss(logits[:, 1:2], anomaly_target)

        return self.focal_weight * focal + self.dice_weight * (dice_norm + dice_anom)

    def forward(self, logits: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...], targets: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, (list, tuple)):
            if len(logits) == 0:
                raise ValueError("Empty logits list is not allowed.")
            losses = [self._single_loss(layer_logits, targets) for layer_logits in logits]
            loss_stack = torch.stack(losses, dim=0)
            if self.multi_layer_reduction == "sum":
                return loss_stack.sum()
            return loss_stack.mean()

        return self._single_loss(logits, targets)