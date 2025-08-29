# FluidDecoder Training System

基于 🤗 Transformers 的 FluidDecoder 模型训练系统，专为天然气管网流体动力学预测设计。

## ✨ 特性

- **完整的 HuggingFace 集成** - 兼容 `transformers.Trainer` 和生态系统
- **高级训练控制** - 混合精度、分布式训练、梯度累积
- **专业指标评估** - 设备分组指标、时序指标、可视化监控
- **灵活的推理接口** - 单步预测、批量预测、自回归生成
- **完善的实验跟踪** - TensorBoard、WandB 集成

## 🚀 快速开始

### 基础训练

```python
from transformers import Trainer, TrainingArguments
from training import (
    FluidDecoderForTraining, FluidDecoderConfig,
    FluidDataCollator, compute_fluid_metrics
)
from data import create_data_loaders

# 1. 创建模型
config = FluidDecoderConfig(d_model=256, n_heads=8, n_layers=6)
model = FluidDecoderForTraining(config)

# 2. 加载数据
train_loader, eval_loader = create_data_loaders("data/dataset")
train_dataset, eval_dataset = train_loader.dataset, eval_loader.dataset

# 3. 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=100,
    per_device_train_batch_size=32,
    learning_rate=1e-4,
    evaluation_strategy="steps",
    eval_steps=500,
    fp16=True,
)

# 4. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=FluidDataCollator(),
    compute_metrics=compute_fluid_metrics,
)

# 5. 开始训练
trainer.train()
```

### 命令行训练

```bash
# 基础训练
python training/scripts/train.py --model_size medium --epochs 50

# 自定义参数
python training/scripts/train.py \
    --d_model 512 --n_heads 16 --n_layers 12 \
    --batch_size 16 --learning_rate 5e-5 \
    --fp16 --gradient_checkpointing \
    --wandb --wandb_project "fluid-dynamics"

# 从检查点恢复
python training/scripts/train.py --resume_from_checkpoint "./results/checkpoint-1000"
```

## 🏗️ 架构设计

### 核心组件

```
training/
├── hf_integration/          # HuggingFace 集成
│   ├── model_wrapper.py     # FluidDecoderForTraining
│   ├── data_collator.py     # FluidDataCollator  
│   ├── metrics.py           # compute_fluid_metrics
│   └── callbacks.py         # 训练回调函数
├── scripts/                 # 训练脚本
│   ├── train.py            # 主训练脚本
│   └── example_usage.py    # 使用示例
├── inference.py            # 推理接口
└── README.md              # 本文档
```

### 数据流

```
[B, T, V=6712] 输入数据
       ↓
FluidDataCollator 批处理
       ↓  
FluidDecoderForTraining 前向传播
       ↓
[B, T, V=6712] 预测输出
       ↓
compute_fluid_metrics 指标计算
```

## 📊 评估指标

### 基础指标
- **MSE**: 均方误差 (主要优化目标)
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **R²**: 决定系数
- **MAPE**: 平均绝对百分比误差

### 设备分组指标
```python
equipment_metrics = {
    'b_mse': 0.0023,    # 球阀 MSE
    'c_mse': 0.0015,    # 压缩机 MSE
    'h_mse': 0.0031,    # 管段 MSE
    'n_mse': 0.0019,    # 节点 MSE
    'p_mse': 0.0027,    # 管道 MSE
    'r_mse': 0.0012,    # 调节阀 MSE
    # ... 对应的 MAE, RMSE 指标
}
```

## ⚙️ 配置选项

### 模型配置

```python
config = FluidDecoderConfig(
    d_model=256,                    # 隐藏层维度
    n_heads=8,                     # 注意力头数
    n_layers=6,                    # Decoder层数
    d_ff=1024,                     # 前馈网络维度
    input_dim=6712,                # 输入维度 (固定)
    output_dim=6712,               # 输出维度 (固定)
    boundary_dims=538,             # 边界条件维度
    dropout=0.1,                   # Dropout概率
    activation="gelu",             # 激活函数
    max_time_positions=10,         # 最大时间位置
    max_variable_positions=6712,   # 最大变量位置
)
```

### 预定义模型大小

| 大小 | d_model | n_heads | n_layers | d_ff | 参数量 |
|------|---------|---------|----------|------|--------|
| small | 128 | 4 | 3 | 512 | ~2M |
| medium | 256 | 8 | 6 | 1024 | ~8M |
| large | 512 | 16 | 12 | 2048 | ~32M |

### 训练参数

```python
training_args = TrainingArguments(
    # 训练控制
    num_train_epochs=100,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    
    # 优化器
    optim="adamw_torch",
    learning_rate=1e-4,
    weight_decay=1e-5,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    
    # 性能优化
    fp16=True,                     # 混合精度
    gradient_checkpointing=True,   # 内存优化
    dataloader_num_workers=4,
    
    # 评估和保存
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
    
    # 实验跟踪
    report_to=["wandb", "tensorboard"],
)
```

## 📈 监控和可视化

### 训练回调

```python
from training.hf_integration.callbacks import create_training_callbacks

callbacks = create_training_callbacks(
    save_plots=True,        # 保存指标图表
    monitor_memory=False,   # 监控内存使用
)

trainer = Trainer(..., callbacks=callbacks)
```

### 生成的图表

训练过程会自动生成以下可视化：

1. **训练和验证损失曲线**
2. **MSE 和 MAE 趋势**
3. **R² 决定系数变化**
4. **设备分组指标对比**

图表保存在 `{output_dir}/training_metrics_*.png`

### 实验跟踪

```python
# WandB 集成
trainer = Trainer(..., args=TrainingArguments(
    report_to=["wandb"],
    run_name="experiment_v1",
))

# TensorBoard 集成 (自动启用)
# 查看: tensorboard --logdir ./logs
```

## 🔮 推理接口

### 基础推理

```python
from training import FluidInference

# 加载模型
inference = FluidInference(
    model_path="./results/final_model",
    normalizer_path="./data/normalizer.pkl",  # 可选
    device="cuda"
)

# 单样本预测
result = inference.predict_single(
    input_data,        # [T, V=6712]
    denormalize=True   # 反归一化结果
)
print(f"预测结果形状: {result['predictions'].shape}")
```

### 批量推理

```python
# 批量预测
results = inference.predict_batch(
    input_batch,       # [B, T, V=6712]  
    batch_size=32,
    denormalize=True
)
```

### 自回归生成

```python
# 多步预测
results = inference.predict_autoregressive(
    initial_input,     # [T, V=6712]
    steps=10,          # 预测10步
    denormalize=True
)
```

## 🛠️ 高级功能

### 分布式训练

```bash
# 多GPU训练
python -m torch.distributed.launch --nproc_per_node=4 \
    training/scripts/train.py --batch_size 8

# 多机训练
python -m torch.distributed.launch \
    --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr="192.168.1.1" --master_port=12355 \
    training/scripts/train.py
```

### 超参数搜索

```python
from transformers import Trainer
import optuna

def objective(trial):
    # 定义超参数搜索空间
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    
    # 创建配置和训练
    config = FluidDecoderConfig(d_model=d_model)
    model = FluidDecoderForTraining(config)
    
    training_args = TrainingArguments(
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        # ... 其他参数
    )
    
    trainer = Trainer(model=model, args=training_args, ...)
    trainer.train()
    
    # 返回优化目标
    eval_result = trainer.evaluate()
    return eval_result['eval_mse']

# 运行优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

### 自定义指标

```python
from training.hf_integration.metrics import create_metrics_computer

# 创建自定义指标计算器
compute_metrics = create_metrics_computer(
    include_equipment=True,   # 包含设备指标
    include_temporal=False,   # 不包含时序指标
)

trainer = Trainer(..., compute_metrics=compute_metrics)
```

## 🐛 调试和故障排除

### 常见问题

1. **内存不足**
   ```python
   training_args = TrainingArguments(
       gradient_checkpointing=True,
       per_device_train_batch_size=8,  # 减小批次大小
       gradient_accumulation_steps=4,  # 增加梯度累积
   )
   ```

2. **训练不收敛**
   ```python
   training_args = TrainingArguments(
       learning_rate=5e-5,      # 降低学习率
       warmup_steps=1000,       # 增加预热步数
       lr_scheduler_type="linear",  # 使用线性衰减
   )
   ```

3. **评估指标异常**
   - 检查数据预处理和归一化
   - 验证预测掩码设置
   - 查看训练日志中的警告信息

### 调试模式

```bash
# 启用调试模式
python training/scripts/train.py --debug --epochs 1

# 使用小数据集测试
python training/scripts/example_usage.py
```

## 📝 最佳实践

### 训练策略

1. **渐进式训练**: 先用小模型和少量数据验证流程
2. **学习率调优**: 使用学习率查找器确定最佳学习率
3. **批次大小**: 根据GPU内存选择合适的批次大小
4. **早停策略**: 设置合理的早停耐心值避免过拟合

### 模型配置

1. **模型大小选择**: 
   - 原型阶段: `small`
   - 实验阶段: `medium`  
   - 生产阶段: `large`

2. **超参数设置**:
   - `d_model`: 通常设为64的倍数
   - `n_heads`: 应能被`d_model`整除
   - `d_ff`: 通常为`d_model`的4倍

### 数据处理

1. **数据归一化**: 确保使用正确的归一化参数
2. **掩码设置**: 边界条件不应参与损失计算
3. **时序对齐**: 确保输入和目标的时序对应关系

## 📚 参考资料

- [HuggingFace Transformers 文档](https://huggingface.co/docs/transformers/)
- [PyTorch 分布式训练指南](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Weights & Biases 集成](https://docs.wandb.ai/guides/integrations/huggingface)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进训练系统！

## 📄 许可证

本项目基于 MIT 许可证开源。