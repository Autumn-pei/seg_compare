## TransUNet / SegFormer 接入指南

### 1. 为什么加这两个网络

已有 4 个纯 CNN，这次扩展后形成三类对照：
- 纯 CNN：UNet / UNet++ / Attention UNet / DeepLabV3+
- CNN + Transformer：TransUNet
- 纯 Transformer：SegFormer

三个阶段共用同一份数据划分、同一个训练循环、同一套指标，保证横向对比公平。

### 2. 本次新增 / 改动的文件

新增：
- `models/trans_blocks.py`  Transformer 共享积木（PatchEmbed / Mlp / Attention / TransformerBlock，已写完，不用填）
- `models/transunet.py`      TransUNet 骨架
- `models/segformer.py`      SegFormer 骨架
- `configs/transunet.yaml`
- `configs/segformer.yaml`

改动：
- `models/__init__.py`       工厂登记 transunet / segformer
- `tests/test_shapes.py`     MODELS 列表加入这两个名字

### 3. 结构要点

**TransUNet（CNN 编码 + Transformer 瓶颈 + 解码上采样）**
- CNN 编码器四级下采样到 16×16，保存 e1~e4 跳跃特征；
- 瓶颈：1×1 投影 → 特征图展平成 256 个 token → 加可学习位置编码 → 4 层 TransformerEncoder → 还原回特征图；
- 解码器四级上采样，与对应跳跃特征通道拼接，套路和 UNet 一致。

**SegFormer（分层 Transformer 主干 + 轻量 MLP 解码器）**
- 没有 CNN 卷积主干，4 个 stage 用 PatchEmbed 切 token，分辨率 128 / 64 / 32 / 16，通道 32 / 64 / 160 / 256；
- 注意力带 sr_ratio 空间缩减，降低 K/V 序列长度；
- 解码器：四个尺度先 1×1 对齐到 256 通道，统一上采样到 64×64 拼接融合，再回到原图 256×256。

### 4. 已实现的关键环节

**`models/transunet.py`**
- 瓶颈：特征图展平成 token → 加位置编码 → 过 encoder → reshape 回特征图；
- 解码器：四级上采样 + `torch.cat` 跳跃拼接 + 卷积融合。

**`models/segformer.py`**
- 解码器融合：四个特征 1×1 对齐 → 上采样到同一尺寸 → 通道拼接；
- 输出上采样：融合特征上采样回 256×256 再过输出头。

> 张量形状约定：特征图 `(B,C,H,W)`，token 序列 `(B,N,C)`。

### 5. 怎么跑测试、怎样算通过

填完 TODO 后，在项目根目录执行：

```powershell
D:\Anaconda\envs\seg\python.exe tests\test_shapes.py
```

**通过标准：**
1. 六个模型逐行打印 `(1, 3, 256, 256) -> (1, 1, 256, 256) OK`；
2. 最后一行打印 `全部维度测试通过 ✅`；
3. 没有红色异常堆栈。

**没通过时怎么看：**
- 若形状不对，会抛 `AssertionError`，括号里是实际形状、后面是期望 `(1, 1, 256, 256)`；
- 对照报错那一层的通道数 / 高宽，检查拼接通道维 `dim=1` 和上采样尺寸。

> 两个模型前向已填完并通过维度测试；若后续改动结构导致形状报错，对照下一小节排查通道维与上采样尺寸。

### 6. 后续实验 SOP（测试通过后）

每个模型两步，一次只跑一个：

```powershell
# TransUNet
python engine/train.py --cfg configs/transunet.yaml
python engine/evaluator.py --cfg configs/transunet.yaml

# SegFormer
python engine/train.py --cfg configs/segformer.yaml
python engine/evaluator.py --cfg configs/segformer.yaml
```

两个都跑完后统一出图、出表：

```powershell
python scripts/make_figures.py
python scripts/make_table.py
```

### 7. 常见坑

- 通道拼接用 `torch.cat([...], dim=1)`，别拼成 batch 维；
- 上采样 `F.interpolate(..., mode="bilinear", align_corners=False)`；
- token 是 `(B,N,C)`，特征图是 `(B,C,H,W)`，换轴别搞反；
- 若 `CUDA out of memory`：先关掉占显存的软件；仍爆再把 `base.yaml` 的 `batch_size` 调小，且六个模型必须用同一个值。

### 8. 骨架实测参数量（未训练）

- transunet：约 9.01M
- segformer：约 6.36M
