"""
D4 统一训练循环：所有模型共用。
用法: python engine/train.py --cfg configs/unet.yaml
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# 把项目根目录加入搜索路径，保证直接运行本脚本能 import 到 data / models / engine
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from torch.optim import AdamW 
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import SegDataset, load_split_records
from models import build_model
from engine.losses import get_loss
from engine.metrics import dice_coeff, iou_coeff, pixel_accuracy, binarize

BASE_CFG = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(cfg_path: str) -> dict:
    """先读 base.yaml，再用模型专属 yaml 覆盖同名项。"""
    with open(BASE_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f) or {}
    cfg.update(model_cfg)
    return cfg


def build_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    size = tuple(cfg["image_size"])
    train_recs = load_split_records(cfg["split_json"], "train")
    val_recs = load_split_records(cfg["split_json"], "val")

    train_ds = SegDataset(train_recs, cfg["data_root"], size, augment=True)
    val_ds = SegDataset(val_recs, cfg["data_root"], size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    return train_loader, val_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    """训练一个 epoch，返回平均训练损失。"""
    model.train()
    total_loss = 0.0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)

        # 清空梯度 -> 前向 -> 算损失 -> 反向 -> 更新参数
        optimizer.zero_grad()
        logits = model(img)
        loss = loss_fn(logits,mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, loss_fn, device) -> dict:
    """验证：返回平均损失与 Dice / IoU / PA。"""
    model.eval()
    total_loss = 0.0
    dice_sum = iou_sum = pa_sum = 0.0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        loss = loss_fn(logits, mask)
        total_loss += loss.item() * img.size(0)

        prob = torch.sigmoid(logits)
        pred = binarize(prob)
        # 累加 dice / iou / pa（用 batch 样本数加权）
        dice_sum += dice_coeff(pred, mask).item() * img.size(0)
        iou_sum += iou_coeff(pred,mask).item()*img.size(0)
        pa_sum += pixel_accuracy(pred,mask).item()*img.size(0)

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "dice": dice_sum / n,
        "iou": iou_sum / n,
        "pa": pa_sum / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="模型配置 yaml 路径")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 模型: {cfg['model']}")

    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg["model"], cfg).to(device)
    loss_fn = get_loss(cfg["loss_type"])

    # 优化器 AdamW + 余弦学习率调度
    optimizer = AdamW(model.parameters(), lr=cfg["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_dice = -1.0
    no_improve = 0
    history = []

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val_metrics,
        })
        print(f"epoch {epoch+1}/{cfg['epochs']} | train_loss {train_loss:.4f} | "
              f"val_loss {val_metrics['loss']:.4f} | dice {val_metrics['dice']:.4f} | "
              f"iou {val_metrics['iou']:.4f} | pa {val_metrics['pa']:.4f} | "
              f"{time.time()-t0:.1f}s")

        # 按验证 Dice 保存最优权重；patience 早停
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            no_improve = 0
            #保存最好权重
            torch.save(model.state_dict(),ckpt_dir/f"{cfg['model']}_best.pth")
        else:
            no_improve += 1
            # 早停判断：连续 patience 轮 Dice 不提升则停止
            if no_improve >= cfg["patience"]:
                print(f"早停触发，连续 {cfg['patience']} 轮 Dice 没有提升")
                break

    # 落盘日志 json
    log_path = log_dir / f"{cfg['model']}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"model": cfg["model"], "best_dice": best_dice, "history": history},
                  f, ensure_ascii=False, indent=2)
    print(f"训练结束，最优验证 Dice: {best_dice:.4f} | 日志 -> {log_path}")


if __name__ == "__main__":
    main()
