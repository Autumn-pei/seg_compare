## 全流程复盘（项目顺序）

> 照着这个顺序做，每一步的产物就是下一步的输入；命令里的 python 用 seg 环境解释器。

1. 建环境
    新建 conda 环境 seg（Python 3.11）
    装 torch(cu128) + torchvision + 其余依赖，版本锁进 requirements.txt
    验收：torch.cuda.is_available() 为 True，能在显卡上跑通一次卷积

2. 搭项目骨架 + 接 git
    目录：configs / data / models / engine / scripts / results / notes / tests
    git init，remote 建联 GitHub，先提交 README 和 .gitignore

3. 数据集核对
    ISIC2017 Task1：train 2000 / val 150 / test 600
    按文件名编号配对原图和掩码，排除 *_superpixels.png 杂图，确认没有孤图孤掩码

4. 数据预处理（scripts/prepare_data.py）
    原图用双线性插值、掩码用最近邻插值，统一缩到 256×256
    掩码二值化成 0/1，输出到 processed/，生成 splits/train_val_test.json

5. 数据加载（data/dataset.py）
    SegDataset 读张量：image (3,H,W) 归一化到 [0,1]，mask (1,H,W) 取值 0/1
    训练集 0.5 概率水平翻转，图和掩码必须同步翻转

6. 搭网络
    (1) 共用积木 models/blocks.py：ConvBNReLU / Down / Up
    (2) 基线 UNet：编码器下采样 + 跳跃连接 + 解码器上采样
    (3) 工厂 models/__init__.py：build_model(name, cfg) 按名字取模型，加新网络只登记一行
    (4) 依次加 UNet++（嵌套跳跃连接）、Attention UNet（注意力门）、DeepLabV3+（ASPP 多尺度空洞卷积）
    (5) tests/test_shapes.py：校验输入 (1,3,256,256) -> 输出 (1,1,256,256)

7. 损失 / 指标（engine/losses.py、metrics.py）
    损失：BCE / Dice / BCE+Dice，get_loss 工厂按名字取
    指标：Dice / IoU / PA，概率按 0.5 阈值二值化

8. 训练循环（engine/train.py）
    合并 base.yaml + 模型 yaml，固定 seed
    AdamW + 余弦退火，逐轮训练 / 验证，按 val Dice 存最优权重，patience 早停
    每轮历史写入 results/logs/<模型>.json

9. 测试集评估（engine/evaluator.py）
    加载 best 权重，在 test 集上算 Dice / IoU / PA，写 <模型>_test.json
    保存 原图 | GT | 预测 叠加图到 results/preds/<模型>/

10. 出图出表
    scripts/make_figures.py：读 logs 画 loss / Dice / IoU / PA 曲线 -> results/curves/
    scripts/make_table.py：汇总参数量 + val / test 指标 -> results/results_table.md

11. 跑批顺序（一次跑一个模型）
    每个模型两步：
      python engine/train.py --cfg configs/<模型>.yaml
      python engine/evaluator.py --cfg configs/<模型>.yaml
    四个都跑完后，再统一：
      python scripts/make_figures.py
      python scripts/make_table.py

12. 结果结论（results/results_table.md）
    DeepLabV3+ 测试集三项指标最高、参数量最小、val->test 掉点最少，精度效率平衡最好
    Attention UNet 验证集最高但测试集回落偏多，有轻度过拟合迹象
    UNet++ 参数最多，相对纯 UNet 提升有限；基线 UNet 四项最低，作对照
    局限：单 seed、从头训练、256 分辨率、小数据集，模型间差距 1~3 个点属同量级
