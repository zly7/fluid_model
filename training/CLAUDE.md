# Training 模块设计文档

## 任务概述
基于已完成的 @models (FluidDecoder) 和 @data (数据处理) 模块，设计完整的训练框架。

## 核心架构选择
**使用 🤗 Transformers Trainer** - 成熟稳定的训练框架，提供自动混合精度、分布式训练、实验跟踪等功能。

## 主要模块设计

### 1. HuggingFace 集成适配

### 2. 训练配置和控制
- **TrainingArguments** - 完整的训练参数配置
- **实验跟踪** - WandB/TensorBoard 集成
- **混合精度训练** - FP16 自动优化
- **梯度裁剪和累积** - 稳定训练的关键技术

### 3. 评估指标系统
- **设备分组评估**: B(球阀)、C(压缩机)、H(管段)、N(节点)、P(管道)、R(调节阀)、T&E(气源分输)
- **掩码损失**: 只对设备变量计算损失，忽略边界条件

## 文件结构
```
training/
├── __init__.py                 # 包初始化
├── config.py                  # 训练配置类
├── inference.py               # 推理接口
│── train.py                   # 主训练脚本,训练中会有评估，并且有可视化记录
│── evaluate.py                # 评估脚本
└── utils.py                   # 训练工具函数
```

## 核心训练流程
1. **数据准备** - 加载训练/验证数据集，应用归一化
2. **训练配置** - 设置 TrainingArguments 参数
3. **训练执行** - Trainer 自动化训练循环
4. **模型保存** - 保存最佳检查点和最终模型，最后report to SwanLab的 Swanlog