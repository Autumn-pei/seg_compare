"""
维度测试：校验每个模型 输入 (1,3,256,256) -> 输出 (1,1,256,256)。
新增模型后，把名字加到 MODELS 列表里。
"""

import sys
import pathlib

import torch

# 把项目根目录加入搜索路径，保证直接运行本脚本能 import 到 models
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from models import build_model

CFG = {"in_channels": 3, "num_classes": 1}
INPUT_SHAPE = (1, 3, 256, 256)
TARGET_SHAPE = (1, 1, 256, 256)

MODELS = ["unet"]


def check_shape(name: str) -> None:
    net = build_model(name, CFG).eval()
    x = torch.randn(*INPUT_SHAPE)
    with torch.no_grad():
        y = net(x)
    assert y.shape == TARGET_SHAPE, f"[{name}] 输出形状 {tuple(y.shape)}，期望 {TARGET_SHAPE}"
    print(f"[{name}] {tuple(x.shape)} -> {tuple(y.shape)} OK")


if __name__ == "__main__":
    for name in MODELS:
        check_shape(name)
    print("全部维度测试通过 ✅")
