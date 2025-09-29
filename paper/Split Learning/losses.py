import torch
import torch.nn as nn
from torch import Tensor


class GroupedLoss(nn.Module):
    def __init__(self, base_loss, n_classes):
        super().__init__()
        self.base_loss = base_loss
        self.n_classes = n_classes

    def forward(self, preds, targets):
        grouped_loss = torch.zeros((self.n_classes, ), device=preds.device, dtype=preds.dtype, requires_grad=preds.requires_grad)
        losses = self.base_loss(preds, targets)
        grouped_loss = grouped_loss.index_add(dim=0, index=targets, source=losses)
        return grouped_loss


class ImportanceLoss(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return _loss_fn_reduction(y_pred, y_true).mul(self.tau).exp().mean().sub(1).divide(self.tau)


_loss_fn = nn.CrossEntropyLoss()
_loss_fn_reduction = nn.CrossEntropyLoss(reduction='none')
