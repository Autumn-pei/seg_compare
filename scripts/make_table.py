"""
D6 汇总表：合并训练日志（最优 Val Dice）与测试集指标，生成 markdown 总表。
用法: python scripts/make_table.py
"""

import json
import sys
from pathlib import Path

import yaml

# 把项目根目录加入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import build_model

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "configs" / "base.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

log_dir = ROOT / cfg["log_dir"]
table_path = ROOT / cfg["result_dir"] / "results_table.md"


def fmt(x) -> str:
    """数值保留 4 位小数；缺失则显示 —。"""
    return "—" if x is None else f"{x:.4f}"


def count_params(name: str) -> float:
    """构建模型并统计参数量（单位 M）。"""
    model = build_model(name, {"in_channels": cfg["in_channels"], "num_classes": cfg["num_classes"]})
    return sum(p.numel() for p in model.parameters()) / 1e6


def main() -> None:
    # 训练日志: <model>.json；测试日志: <model>_test.json
    train_files = {p.stem: p for p in log_dir.glob("*.json") if not p.stem.endswith("_test")}
    test_files = {p.stem[:-5]: p for p in log_dir.glob("*_test.json")}

    rows = []
    for _, p in sorted(train_files.items()):
        with open(p, encoding="utf-8") as f:
            train_data = json.load(f)
        model = train_data["model"]

        test_data = {}
        if model in test_files:
            with open(test_files[model], encoding="utf-8") as f:
                test_data = json.load(f)

        rows.append({
            "model": model,
            "params": count_params(model),
            "val_dice": train_data.get("best_dice"),
            "test_dice": test_data.get("dice"),
            "test_iou": test_data.get("iou"),
            "test_pa": test_data.get("pa"),
        })

    # 按 Test Dice 降序排序；没有测试指标（None）的排到最后。
    rows.sort(key=lambda r: (r["test_dice"] is None, -(r["test_dice"] or 0.0)))

    # 组装 markdown 表格
    lines = [
        "| Model | Params(M) | Val Best Dice | Test Dice | Test IoU | Test PA |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        # 按表头列顺序拼一行 markdown；数值列套 fmt，参数量保留 2 位小数
        lines.append(
            f"| {r['model']} | {r['params']:.2f} | {fmt(r['val_dice'])} | "
            f"{fmt(r['test_dice'])} | {fmt(r['test_iou'])} | {fmt(r['test_pa'])} |"
        )

    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n总表已保存 -> {table_path}")


if __name__ == "__main__":
    main()
