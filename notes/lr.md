## 学习心得

流程
1.搭项目文件骨架
    configs
    data
    engine
    models
    test
    results
2.下载文件/数据集
 原始：放入raw

3.建一个github仓库 和git 做匹配
初始化 git init
建联 git remote add origin https://github.com/你的用户名/仓库名.git

4.做数据预处理 
编写prepare_data.py 将图片统一备用
dataset.py 做具体裁剪反转...操作 并将图片转为张良 方便后序运行

5.开始搭建网络 
（1）unet
 先写固定的模块 blocks 将他们拼在一起组成unet

 6.编写损失函数
 BCE DICE 以及综合起来

