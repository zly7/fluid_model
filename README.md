# 流体管网预测模型 (Fluid Pipeline Network Prediction Model)

基于深度学习的长输干线管网水热力仿真预测系统，支持分钟级实时预测。

## 项目概述

本项目针对我国某长输干线管网拓扑系统，通过深度学习模型实现对管网各节点、管道、设备的水热力参数预测。系统包含7个气源、122个分输点、262个管道、572个节点、23台压缩机、343个球阀及10个调节阀。

### 系统特点

- **多模型架构**: 支持Decoder、CNN、LSTM三种深度学习模型
- **分钟级预测**: 实现分钟级别的高频预测
- **完整数据流**: 从数据预处理到模型训练到可视化的完整pipeline  
- **统一接口**: 所有模型兼容transformers训练框架
- **实时可视化**: 基于Streamlit的Web界面

## 技术架构

### 数据维度
- **输入维度**: [B, T, 6712]
  - B: 批次大小
  - T: 时间步长（默认3分钟）
  - 6712: 539个边界变量 + 6174个设备变量
- **预测维度**: [B, T, 6174] （设备变量预测）

### 设备类型
| 简称 | 名称 | 数量 | 参数类型 |
|------|------|------|----------|
| T | 气源 | 7 | p, q, t |
| E | 用户/分输 | 122 | p, q, t |
| N | 节点 | 572 | p, q, t |
| P | 管道 | 230 | p_in, p_out, q_in, q_out, t_in, t_out, inv |
| H | 管段 | 32 | p_in, p_out, q_in, q_out, t_in, t_out |
| C | 压缩机 | 23 | p_in, p_out, q_in, q_out, t_in, t_out, pwr |
| B | 球阀 | 343 | p_in, p_out, q_in, q_out, t_in, t_out |
| R | 调节阀 | 10 | p_in, p_out, q, t_in, t_out |

## 安装和使用

### 环境要求
```bash
pip install -r requirements.txt
```

### 核心依赖
- PyTorch >= 2.0.0
- transformers（训练框架兼容）
- pandas, numpy, scikit-learn
- streamlit（Web界面）
- tensordict（数据结构）

### 快速开始

#### 1. 数据预处理
```python
from data import DataProcessor, create_data_loaders

# 创建数据处理器
processor = DataProcessor(data_dir="path/to/data")
train_loader, val_loader = create_data_loaders(
    data_dir="path/to/data",
    batch_size=32,
    sequence_length=3
)
```

#### 2. 模型训练
```bash
# 使用默认配置训练Decoder模型
python train.py --config configs/quick_test.json

# 训练LSTM模型
python train.py --config configs/quick_test_lstm.json

# 训练CNN模型  
python train.py --config configs/quick_test_cnn.json
```

#### 3. 可视化和监控
```bash
# 启动Streamlit Web界面
streamlit run visualization/streamlit_app.py

# 启动完整功能界面
streamlit run visualization/streamlit_app_full.py
```

## 项目结构

```
flued_model/
├── data/                   # 数据处理模块
│   ├── dataset.py         # 数据集类
│   ├── normalizer.py      # 数据归一化
│   ├── loader.py          # 数据加载器
│   └── processor.py       # 数据预处理
├── models/                # 模型架构
│   ├── base.py           # 基础模型类
│   ├── decoder/          # Decoder模型
│   ├── cnn/              # CNN模型
│   └── lstm/             # LSTM模型
├── configs/              # 配置文件
│   ├── default.json      # 默认训练配置
│   ├── production.json   # 生产环境配置
│   └── models/           # 模型配置文件
├── training/             # 训练框架
│   ├── trainer.py        # 训练器
│   ├── callbacks.py      # 训练回调
│   └── utils.py          # 训练工具
├── visualization/        # 可视化模块
│   ├── streamlit_app.py  # Web界面
│   ├── plots.py          # 图表绘制
│   └── multi_variable_viewer.py
└── train.py             # 训练脚本
```

## 模型架构

### 1. FluidDecoder（纯Decoder架构）
- **架构**: Transformer Decoder-only
- **特点**: 自回归预测，适合序列生成任务
- **输入处理**: 线性投影 + 位置编码
- **注意力**: Multi-head self-attention
- **参数规模**: nano(~1M) / small(~5M) / medium(~20M)

### 2. FluidCNN（卷积神经网络）
- **架构**: 1D卷积 + 残差连接
- **特点**: 快速训练，适合时序模式识别
- **卷积核**: 多尺度时间卷积
- **池化**: 全局平均池化
- **输出**: 全连接层映射到目标维度

### 3. FluidLSTM（长短期记忆网络）
- **架构**: 双向LSTM + 注意力机制
- **特点**: 优秀的时序建模能力
- **处理方式**: 分别处理边界和设备变量
- **输出**: 注意力加权的特征融合
- **参数统计**: nano配置约275万参数

## 数据格式

### 输入数据
- **Boundary.csv**: 边界条件，30分钟更新一次
  - 参数类型: SNQ(设定流量), SP(设定压力), ST(运行状态), SPD(下游设定压力), FR(阀门开度)
- **设备CSV文件**: B.csv, C.csv, H.csv, N.csv, P.csv, R.csv, T&E.csv
  - 更新频率: 每分钟
  - 参数类型: 压力、流量、温度、管存、功率等

### 预测目标
模型预测下一分钟各设备的状态参数，支持多步预测。

## 训练配置

### 配置文件说明
- `quick_test.json`: 快速测试配置（小数据集，短训练）
- `production.json`: 生产环境配置（完整数据集）
- `full_training.json`: 完整训练配置（长训练周期）

### 模型配置
```json
{
  "model_type": "decoder",
  "d_model": 256,
  "n_heads": 8,
  "n_layers": 6,
  "dropout": 0.1,
  "max_sequence_length": 10
}
```

## 性能监控

- **TensorBoard**: 训练过程可视化
- **SwanLab**: 实验管理和追踪
- **自定义回调**: 学习率调度、早停、模型保存

## 可视化功能

### Web界面特性
- **多变量查看器**: 支持6000+变量的实时查看
- **交互式图表**: 基于Plotly的动态图表
- **数据格式介绍**: 详细的数据结构说明
- **预测结果对比**: 多模型预测结果对比

### 启动方式
```bash
# 基础界面
streamlit run visualization/streamlit_app.py

# 完整功能界面
streamlit run visualization/streamlit_app_full.py
```

## 开发规范

### 代码结构
- 采用Package结构，使用`__init__.py`
- 不超过5个主要Package
- 统一的配置管理系统
- 完整的类型注解

### 错误处理
- 不进行过度的try-catch包装
- 让错误直接暴露，便于调试
- 在适当位置进行错误处理

## 许可证

本项目用于学术研究目的。

## 贡献指南

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 联系方式

如有问题请提交Issue或联系项目维护者。