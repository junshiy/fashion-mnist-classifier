# 基于 TensorFlow 的 Fashion MNIST 服装图像分类工具

基于 TensorFlow-Keras 搭建卷积神经网络（CNN），对 Fashion-MNIST 服饰数据集进行图像分类，并提供 Tkinter 可视化界面，支持测试集序号查询、随机测试与本地图片识别。本项目为课程设计项目，**两人分工完成**：本人负责模型训练模块（`train_model.py` 与模型文件），另一位同学负责 GUI 界面模块（`gui_app.py`）。

## 功能

- **模型训练**：加载 Fashion-MNIST 数据集，完成预处理、数据增强、CNN 建模、训练回调与最优模型保存；
- **序号查询 / 随机测试**：输入 0~9999 的图片序号或随机抽样，展示 28×28 灰度图、真实类别、预测类别与置信度；
- **本地图片识别**：选择本地 PNG/JPG 图片，经灰度化、缩放、归一化、色彩反转后送入模型推理，输出预测类别与置信度。

## 数据集与模型

- **Fashion-MNIST**：60000 张训练图像 + 10000 张测试图像，均为 28×28 灰度图，共 10 个服饰类别（T恤、裤子、套头衫、连衣裙、外套、凉鞋、衬衫、运动鞋、包、短靴）；
- **CNN 结构**：三层卷积模块（32 / 64 / 128 卷积核 + 批归一化 + 最大池化 + Dropout）+ 全连接层，输出层 10 个神经元；
- **训练策略**：ImageDataGenerator 数据增强（旋转、平移、翻转、缩放）、AdamW 优化器（学习率 1e-3 + 权重衰减 1e-4）、SparseCategoricalCrossentropy 损失；EarlyStopping（patience=6）、ReduceLROnPlateau（patience=3）、ModelCheckpoint 自动保存最优模型；
- **测试结果**：在独立测试集上准确率 **91.04%**。

## 运行环境

- 课程原始环境：Python 3.8 + PyCharm 2022.1.3 + TensorFlow 2.x（Conda 虚拟环境 dl_basic）
- 依赖库：`tensorflow`、`numpy`（GUI 部分使用 Python 内置的 `tkinter`，无需额外安装）

## 使用方式

1. **训练模型**（可选，仓库已附训练好的 `fashion_mnist_model.h5`）：

   ```bash
   python train_model.py
   ```

   运行后在同目录生成 `fashion_mnist_model.h5`。

2. **启动图形界面**：

   ```bash
   python gui_app.py
   ```

   确保 `fashion_mnist_model.h5` 与脚本位于同一目录。启动后可输入序号查询、随机测试，或选择 `samples/` 目录下的示例图片（及任意本地图片）进行识别。

## 目录结构

```
fashion-mnist-classifier/
├── train_model.py            # 模型训练脚本（本人负责）
├── gui_app.py                # Tkinter 图形界面（队友负责）
├── fashion_mnist_model.h5    # 训练好的模型权重
├── samples/                  # 20 张本地测试图片示例
└── README.md
```

## 技术栈

Python · TensorFlow / Keras · CNN · NumPy · Tkinter
