"""
模型工厂：训练循环通过 build_model(name, cfg) 按名字取模型。
"""

import torch.nn as nn

from models.unet import UNet
from models.unetplusplus import UNetPlusPlus
from models.attention_unet import AttentionUNet
from models.deeplabv3plus import DeepLabV3Plus
from models.transunet import TransUNet
from models.segformer import SegFormer

# 模型名 -> 模型类
_MODELS = {
    "unet": UNet,
    "unetpp": UNetPlusPlus,
    "attention_unet": AttentionUNet,
    "deeplabv3plus": DeepLabV3Plus,
    "transunet": TransUNet,
    "segformer": SegFormer,
}


def build_model(name: str, cfg: dict) -> nn.Module:
    """
    根据名字实例化模型。
    """
    if name not in _MODELS:
        raise KeyError(f"未知模型 '{name}'，可选: {list(_MODELS.keys())}")
    return _MODELS[name](
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
    )
