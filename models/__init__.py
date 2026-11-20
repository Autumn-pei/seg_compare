"""
模型工厂：训练循环通过 build_model(name, cfg) 按名字取模型。
新增模型时：在 _MODELS 字典里登记一行即可，训练代码不用改。
"""

import torch.nn as nn

from models.unet import UNet

# 模型名 -> 模型类。后续 UNet++ / Attention UNet / DeepLabV3+ / nnUNet 在这里登记。
_MODELS = {
    "unet": UNet,
}


def build_model(name: str, cfg: dict) -> nn.Module:
    """
    根据名字实例化模型。
    cfg 至少包含: in_channels, num_classes。
    """
    if name not in _MODELS:
        raise KeyError(f"未知模型 '{name}'，可选: {list(_MODELS.keys())}")
    return _MODELS[name](
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
    )
