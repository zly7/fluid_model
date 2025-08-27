# 天然气管网数值模拟系统

## 项目简介

本项目是针对"赛题3：数据与机理融合的天然气管网数值模拟"开发的完整解决方案。该系统能够深度融合物理机理与多源数据，实现高保真、低时延、可扩展、易维护的管网水热力仿真核心引擎。

## 系统特性

- **多模型架构**：支持标准神经网络、物理信息神经网络(PINN)、注意力机制网络
- **数据与机理融合**：结合数据驱动方法和物理约束
- **完整的训练流程**：从数据预处理到模型评估的全流程
- **可视化支持**：丰富的图表和交互式仪表板
- **模块化设计**：易于扩展和维护

## 项目结构

```
fluid_model/
├── data/                    # 数据处理模块
│   ├── __init__.py
│   ├── dataset.py          # 数据集定义
│   ├── loader.py           # 数据加载器
│   └── dataset/            # 数据文件
│       ├── train/          # 训练数据
│       └── test/           # 测试数据
├── models/                  # 模型定义模块
│   ├── __init__.py
│   ├── neural_network.py   # 神经网络模型
│   └── utils.py            # 模型工具函数
├── training/                # 训练模块
│   ├── __init__.py
│   ├── trainer.py          # 训练器
│   ├── loss.py             # 损失函数
│   └── optimizer.py        # 优化器和调度器
├── inference/               # 推理模块
│   ├── __init__.py
│   ├── predictor.py        # 预测器
│   └── evaluator.py        # 评估器
├── visualization/           # 可视化模块
│   ├── __init__.py
│   ├── plotter.py          # 绘图工具
│   └── dashboard.py        # 交互式仪表板
├── main.py                  # 主程序入口
├── requirements.txt         # 依赖包列表
└── README.md               # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
# 使用默认配置训练
python main.py --mode train

# 指定数据目录和输出目录
python main.py --mode train --data_dir /path/to/data --output_dir /path/to/results

# 指定GPU设备
python main.py --mode train --device cuda
```

### 2. 测试模型

```bash
# 使用训练好的模型进行预测
python main.py --mode test --model_path checkpoints/best_model.pth

# 指定测试数据和输出目录
python main.py --mode test --model_path checkpoints/best_model.pth --data_dir /path/to/data --output_dir /path/to/results
```

### 3. 启动可视化仪表板

```bash
python main.py --mode dashboard
```

## 模型配置

系统支持多种模型配置：

### 标准神经网络 (FluidNet)
- 多层全连接网络
- 支持多种激活函数
- 批量归一化和Dropout

### 物理信息神经网络 (PhysicsInformedNet)
- 融合物理约束损失
- 质量守恒、动量守恒、能量守恒约束
- 自适应物理权重

### 注意力机制网络 (AttentionFluidNet)
- Transformer编码器架构
- 多头自注意力机制
- 位置编码

## 数据格式

### 输入数据格式
- **Boundary.csv**: 边界条件文件，包含时间序列和各设备参数

### 输出数据格式
- **B.csv**: 球阀数据
- **T&E.csv**: 气源和分输点数据
- **H.csv**: 管段数据
- **C.csv**: 压缩机数据
- **N.csv**: 节点数据
- **R.csv**: 调节阀数据
- **P.csv**: 管道数据

## 主要功能

### 数据处理
- 自动数据加载和预处理
- 支持时间序列数据
- 数据归一化和标准化

### 模型训练
- 多种优化器支持（Adam, AdamW, SGD等）
- 学习率调度策略
- 早停机制
- 模型检查点保存

### 模型评估
- 多种评估指标（MSE, MAE, R², MAPE等）
- 预测结果可视化
- 模型性能报告生成

### 可视化分析
- 训练曲线图
- 预测vs真实值对比
- 误差分布分析
- 相关性分析
- 交互式仪表板

## 性能优化建议

1. **数据处理优化**
   - 使用适当的批次大小
   - 启用数据并行加载
   - 数据预处理缓存

2. **模型优化**
   - 选择合适的网络架构
   - 调整学习率和优化器参数
   - 使用混合精度训练

3. **硬件加速**
   - 使用GPU进行训练和推理
   - 多GPU并行训练
   - 模型量化和剪枝

## 技术规格

- **编程语言**: Python 3.8+
- **深度学习框架**: PyTorch 1.10+
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib, Seaborn, Plotly
- **Web界面**: Streamlit

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证。详见 `LICENSE` 文件。

## 联系方式

如有问题或建议，请提交 Issue 或联系开发团队。