"""
D6 出图：读取 results/logs/*.json 训练历史，绘制多模型对比曲线到 results/curves/。
用法: python scripts/make_figures.py
"""

import json
import sys
from pathlib import Path

import yaml
import matplotlib
matplotlib.use("Agg")  #只保存图片，不弹窗
import matplotlib.pyplot as plt

# 把项目根目录加入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "configs" / "base.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

log_dir = ROOT / cfg["log_dir"]
curve_dir = ROOT / cfg["curve_dir"]
curve_dir.mkdir(parents=True, exist_ok=True)


def load_train_logs() -> dict:
    """读取所有 <model>.json（排除 <model>_test.json），返回 {model: history}。"""
    logs = {}
    for p in sorted(log_dir.glob("*.json")):
        if p.stem.endswith("_test"):
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        logs[data["model"]] = data["history"]
    return logs


def plot_metric(ax, logs: dict, key: str, title: str, ylabel: str) -> None:
    """在一个子图上，为每个模型画一条 key 指标随 epoch 变化的曲线。"""
    for model, history in logs.items():
        epochs = [h["epoch"] for h in history]
        values = [h[key] for h in history]
        ax.plot(epochs, values, marker="o", markersize=3, label=model)

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)


def main() -> None:
    logs = load_train_logs()
    if not logs:
        print("还没有训练日志，请先运行 engine/train.py")
        return

    # 2 行 3 列：train_loss / val_loss / dice / iou / pa，最后一格留给测试集对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plot_metric(axes[0, 0], logs, "train_loss", "Train Loss", "loss")
    plot_metric(axes[0, 1], logs, "loss", "Val Loss", "loss")
    plot_metric(axes[0, 2], logs, "dice", "Val Dice", "Dice")
    plot_metric(axes[1, 0], logs, "iou", "Val IoU", "IoU")
    plot_metric(axes[1, 1], logs, "pa", "Val PA", "PA")
    axes[1, 2].axis("off")  # 测试集柱状图在 make_table 阶段再补

    fig.tight_layout()
    out = curve_dir / "metrics_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"曲线已保存 -> {out}")


if __name__ == "__main__":
    main()
