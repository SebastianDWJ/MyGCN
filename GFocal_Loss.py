import torch
import torch.nn as nn
import torch.nn.functional as F


class GFocalLoss(nn.Module):
    def __init__(self, beta=2.0, use_sigmoid=True):
        super().__init__()
        self.beta = beta
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        r"""
        Focal loss
        :param pred: shape=(B,  HW)
        :param label: shape=(B, HW)
        """
        if self.use_sigmoid:
            pred = self.sigmoid(pred)
        pred = pred.view(-1)
        label = target.view(-1)
        pos = torch.nonzero(label > 0).squeeze(1)
        pos_num = max(pos.numel(), 1.0)
        mask = ~(label == -1)
        pred = pred[mask]
        label = label[mask]
        scale_factor = (pred - label).abs().pow(self.beta)
        loss = F.binary_cross_entropy(pred, label, reduction='none') * scale_factor
        return loss.sum() / pos_num