"""
D6 测试集评估：加载最优权重，在 test 子集上计算 Dice / IoU / PA,存入json,并保存预测叠加图。
用法: python engine/evaluator.py --cfg configs/unet.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# 把项目根目录加入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import SegDataset, load_split_records
from models import build_model
from engine.metrics import dice_coeff, iou_coeff, pixel_accuracy, binarize

BASE_CFG = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
NUM_VIS = 4  # 保存前几张样本做预测叠加图


def load_config(cfg_path: str) -> dict:
    """先读 base.yaml，再用模型专属 yaml 覆盖同名项（与 train.py 保持一致）。"""
    with open(BASE_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}
    cfg.update(model_cfg)
    return cfg


@torch.no_grad()
def evaluate(model, loader, device):
    """在给定 loader 上前向一遍，返回平均指标与前 NUM_VIS 个样本张量。"""
    model.eval()
    dice_sum = iou_sum = pa_sum = 0.0
    samples = []  # 元素: (img, gt, pred)，每个形状 (C, H, W)，已在 CPU

    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        prob = torch.sigmoid(logits)
        pred = binarize(prob)

        dice_sum += dice_coeff(pred, mask).item() * img.size(0)
        iou_sum += iou_coeff(pred, mask).item() * img.size(0)
        pa_sum += pixel_accuracy(pred, mask).item() * img.size(0)

        # 收集前 NUM_VIS 个样本用于出叠加图（移到 CPU）
        for i in range(img.size(0)):
            if len(samples) < NUM_VIS:
                samples.append((img[i].cpu(), mask[i].cpu(), pred[i].cpu()))

    n = len(loader.dataset)
    metrics = {
        # 总和 ÷ 总样本数 = 整个测试集平均指标
        "dice": dice_sum / n,
        "iou": iou_sum / n,
        "pa": pa_sum / n,
    }
    return metrics, samples


def save_overlay(samples, out_dir: Path, model_name: str) -> None:
    """把样本排成 行=样本、列=原图/GT/预测 的网格图保存。"""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(samples), 3, figsize=(9, 3 * len(samples)))
    if len(samples) == 1:
        axes = axes[None, :]

    for row, (img, gt, pred) in enumerate(samples):
        # img: (3,H,W) [0,1] -> (H,W,3) 才能 imshow；gt/pred: (1,H,W) 去掉通道维
        img_np = img.permute(1, 2, 0).numpy()
        gt_np = gt.squeeze(0).numpy()
        pred_np = pred.squeeze(0).numpy()

        axes[row, 0].imshow(img_np)
        axes[row, 1].imshow(gt_np, cmap="gray")
        axes[row, 2].imshow(pred_np, cmap="gray")
        axes[row, 0].set_title("image")
        axes[row, 1].set_title("GT")
        axes[row, 2].set_title("pred")
        for col in range(3):
            axes[row, col].axis("off")

    fig.suptitle(model_name)
    fig.tight_layout()
    fig.savefig(out_dir / "overlay.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="模型配置 yaml 路径")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"评估设备: {device} | 模型: {cfg['model']}")

    # 测试集：不 增强、打乱
    test_recs = load_split_records(cfg["split_json"], "test")
    test_ds = SegDataset(test_recs, cfg["data_root"], tuple(cfg["image_size"]), augment=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    model = build_model(cfg["model"], cfg).to(device)
    ckpt_path = Path(cfg["checkpoint_dir"]) / f"{cfg['model']}_best.pth"

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    metrics, samples = evaluate(model, test_loader, device)

    # 存入测试集指标 json
    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = {"model": cfg["model"], "n": len(test_ds), **metrics}
    with open(log_dir / f"{cfg['model']}_test.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    save_overlay(samples, Path(cfg["pred_dir"]) / cfg["model"], cfg["model"])
    print(f"[{cfg['model']}] test 指标: {metrics}")
    print(f"叠加图 -> {Path(cfg['pred_dir']) / cfg['model'] / 'overlay.png'}")


if __name__ == "__main__":
    main()
