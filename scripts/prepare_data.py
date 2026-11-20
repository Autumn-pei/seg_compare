"""
D1-3 数据预处理脚本
职责：读取 data/raw 下的 ISIC 2017 官方三子集，统一缩放到 image_size，
      存到 data/processed/，并生成固定划分 data/splits/train_val_test.json。
"""

import json
import os
from pathlib import Path

import cv2
import yaml

# 1. 读取全局配置（base.yaml）。路径字段：data_root / raw_dir / processed_dir / split_json / image_size。这里先把配置加载进来。

CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


def load_config(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# 官方三子集：子集名 -> (原图目录名, 掩码目录名)
SPLIT_DIRS = {
    "train": ("ISIC-2017_Training_Data", "ISIC-2017_Training_Part1_GroundTruth"),
    "val":   ("ISIC-2017_Validation_Data", "ISIC-2017_Validation_Part1_GroundTruth"),
    "test":  ("ISIC-2017_Test_v2_Data", "ISIC-2017_Test_v2_Part1_GroundTruth"),
}


def pair_samples(img_dir: Path, mask_dir: Path) -> list[tuple[str, str]]:
    """
    配对一个子集里的原图与掩码。
    返回：[(image_name, mask_name), ...]，只保留两边都存在的样本。
    """
    #收集原图前缀
    img_stems={}
    for f in img_dir.glob("*.jpg"):
        stem=f.stem
        img_stems[stem]=f.name
    #收集掩码前缀
    mask_stems={}
    for f in mask_dir.glob("*_segmentation.png"):
        stem=f.stem.replace("_segmentation","")
        mask_stems[stem]=f.name
    #交集：两边都有的前缀 排序 返回元组列表
    common_stems=sorted(set(img_stems.keys())&set(mask_stems.keys()))
    pairs=[]
    for s in common_stems:
        pairs.append((img_stems[s],mask_stems[s]))
    return pairs


def process_one(image_path: Path, mask_path: Path,
                out_img_path: Path, out_mask_path: Path, size: tuple[int, int]) -> None:
    """
    处理单张样本：读图 -> 缩放 -> 掩码二值化 -> 存盘。
    """
    #读图
    img = cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"读取失败：{image_path}")
        return
    
    #缩放
    img_resize = cv2.resize(img,size,interpolation=cv2.INTER_LINEAR)
    mask_resize= cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)
    #掩码二值化
    mask_bin=(mask_resize>0).astype("uint8")
    #保存
    cv2.imwrite(str(out_img_path),img_resize)
    cv2.imwrite(str(out_mask_path),mask_bin)


def main() -> None:
    cfg = load_config(CFG_PATH)
    # 1.加载配置 从 cfg 取出配置路径与尺寸
    data_root = Path(cfg["data_root"])
    raw_dir = Path(cfg["raw_dir"])
    processed_dir = Path(cfg["processed_dir"])
    split_json = Path(cfg["split_json"])
    size_w, size_h = cfg["image_size"]   # image_size: [256, 256]

    # 2.创建输出文件夹
    for split in SPLIT_DIRS:
        (processed_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (processed_dir / split / "masks").mkdir(parents=True, exist_ok=True)

    # 3.准备一个空字典 放存入json的内容
    split_records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    # 4.循环:依次处理train val test这三组
    for split, (img_dir_name, mask_dir_name) in SPLIT_DIRS.items():
        
        img_dir = Path(raw_dir) / img_dir_name
        mask_dir = Path(raw_dir) / mask_dir_name

        pairs = pair_samples(img_dir, mask_dir)
        print(f"[{split}] 配对到 {len(pairs)} 个样本")

        #循环每一对（原图，掩码）
        for img_name, mask_name in pairs:
            stem = Path(img_name).stem  # 公共前缀，如 ISIC_0000000
            out_img = Path(processed_dir) / split / "images" / f"{stem}.jpg"
            out_mask = Path(processed_dir) / split / "masks" / f"{stem}.png"

            process_one(img_dir / img_name, mask_dir / mask_name,
                        out_img, out_mask, (size_w, size_h))

            # 记录相对 data_root 的路径（json 里不存绝对路径）
            split_records[split].append({
                "image": os.path.relpath(out_img, data_root).replace("\\", "/"),
                "mask":  os.path.relpath(out_mask, data_root).replace("\\", "/"),
            })

    # 5.全部图片处理完后，把记录写入json
    Path(split_json).parent.mkdir(parents=True, exist_ok=True)
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump(split_records, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in split_records.values())
    print(f"完成：共 {total} 个样本，划分文件 -> {split_json}")


if __name__ == "__main__":
    main()
