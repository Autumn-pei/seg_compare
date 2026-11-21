"""
D4 分割指标：Dice / IoU / PA(像素准确率)。
输入 pred, target: (B, 1, H, W)，pred 为二值 0/1，target 为 0/1。
"""

import torch

SMOOTH = 1e-5


def binarize(prob: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """概率按阈值二值化，返回 0/1 浮点张量。"""
    return (prob >= threshold).float()


def dice_coeff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.flatten(1)
    target = target.flatten(1).float()
    inter = (pred * target).sum(1)
    return ((2 * inter + SMOOTH) / (pred.sum(1) + target.sum(1) + SMOOTH)).mean()


def iou_coeff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """IoU = 交集 / 并集。并集 = pred + target - 交集。"""
    pred = pred.flatten(1)
    target = target.flatten(1).float()

    inter = (pred * target).sum(1)
    union = pred.sum(1)+target.sum(1)-inter
    return ((inter+SMOOTH)/(union+SMOOTH)).mean()


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PA = 预测正确的像素数 / 总像素数。"""
    # TODO(你填)：逐像素比较 pred 与 target，统计相等比例
    pred = pred.flatten(1)
    target = target.flatten(1).float()

    correct= (pred==target).sum(1)
    total_pixel=target.shape[1]

    return ((correct+SMOOTH)/(total_pixel+SMOOTH)).mean()
