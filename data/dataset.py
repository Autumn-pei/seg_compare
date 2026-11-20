"""
D2 数据加载：从 processed/ + split json 读取 (image, mask) 张量。
张量约定：
- image: (3, H, W) float32，归一化到 [0,1]，通道在前
- mask:  (1, H, W) float32，取值 0/1
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_split_records(split_json: str, split: str) -> list[dict]:
    """读取划分 json，返回指定子集（train/val/test）的记录列表。"""
    with open(split_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[split]


class SegDataset(Dataset):
    def __init__(self, records: list[dict], data_root: str,
                 image_size: tuple[int, int] = (256, 256), augment: bool = False):
        self.records = records
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        img_path = self.data_root / rec["image"]
        mask_path = self.data_root / rec["mask"]

        # 1) 读图：原图彩色，掩码灰度
        img = cv2.imread(str(img_path),cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)

        # 2) 训练集增强：随机水平翻转（图和掩码必须同步翻转）
        if self.augment:
            if np.random.random()<0.5:
                img=img[:,::-1,:]
                mask=mask[:,::-1]

        # 3) 归一化 + 换轴
        #    img  : 像素 / 255.0 -> float32 -> 换轴 (H,W,C) -> (C,H,W)
        img = img/255.0
        img=img.astype(np.float32)
        img_t = torch.from_numpy(img.transpose(2,0,1))

        #    mask : 保持 0/1 -> float32 -> 增加通道维 (H,W) -> (1,H,W)
        mask = mask.astype(np.float32)
        mask = np.expand_dims(mask,axis=0)
        mask_t = torch.from_numpy(mask)
        
        return img_t, mask_t
