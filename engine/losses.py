"""
D4 损失函数：CE(BCEWithLogits) / Dice / BCE+Dice。
输入 logits: (B, 1, H, W)，target: (B, 1, H, W) 取值 0/1。
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """软 Dice 损失：1 - Dice 系数。内部对 logits 做 sigmoid。"""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        # 展平成 (B, H*W)，按样本逐个算 Dice
        prob = prob.flatten(1)
        target = target.flatten(1).float()

        inter = (prob*target).sum(dim=1)
        union = prob.sum(dim=1)+target.sum(dim=1)

        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEPlusDice(nn.Module):
    """BCE + Dice 联合损失，权重各 1。"""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        lce_loss = self.bce(logits,target)
        dice_loss = self.dice(logits,target)
        return lce_loss+dice_loss


def get_loss(loss_type: str) -> nn.Module:
    """按配置名返回损失实例。loss_type: bce / dice / bce_dice。"""
    _LOSS_MAP_={
        "bce":nn.BCEWithLogitsLoss(),
        "dice":DiceLoss(),
        "bce_dice":BCEPlusDice(),
    }
    if loss_type not in _LOSS_MAP_:
        raise KeyError(f"不支持损失函数{loss_type},可选{list(_LOSS_MAP_.keys())}")
    return _LOSS_MAP_[loss_type]
