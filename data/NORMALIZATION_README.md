# 数据归一化系统使用指南

## 概述

重新设计的数据归一化系统提供了更灵活、高效和可控的数据标准化功能。系统分离了归一化逻辑和数据加载逻辑，支持多种归一化方法，并允许在训练、推理和可视化等不同场景中灵活控制。

## 系统架构

### 核心组件

1. **DataNormalizer**: 独立的归一化管理器
2. **compute_normalization_stats.py**: 统计量预计算工具
3. **FluidDataset**: 简化的数据集类（移除了内置归一化）
4. **create_collate_fn**: 支持归一化的数据加载函数

### 数据流

```
原始数据 -> DataProcessor -> FluidDataset -> DataLoader -> 归一化 -> 模型
                                            ↑
                                    collate_fn + DataNormalizer
```

## 使用方法

### 1. 预计算归一化统计量

在开始训练之前，首先需要计算归一化统计量：

```bash
cd data/
python compute_normalization_stats.py --data_dir /path/to/data --method standard --max_samples 100 --validate
```

**参数说明：**
- `--data_dir`: 数据根目录路径
- `--method`: 归一化方法（standard, minmax, robust）
- `--max_samples`: 最大样本数（控制内存使用，可选）
- `--sequence_length`: 序列长度（默认3）
- `--validate`: 验证统计量的正确性
- `--force`: 强制重新计算（即使文件已存在）

### 2. 训练时使用归一化

```python
from data.dataset import FluidDataset, create_dataloader_with_normalization

# 创建数据集
train_dataset = FluidDataset(
    data_dir="/path/to/data",
    split='train',
    sequence_length=3
)

val_dataset = FluidDataset(
    data_dir="/path/to/data", 
    split='val',
    sequence_length=3
)

# 创建带归一化的DataLoader
train_loader = create_dataloader_with_normalization(
    train_dataset,
    batch_size=32,
    shuffle=True,
    normalizer_method='standard',
    apply_normalization=True  # 训练时开启归一化
)

val_loader = create_dataloader_with_normalization(
    val_dataset,
    batch_size=64,
    shuffle=False,
    normalizer_method='standard',
    apply_normalization=True
)

# 训练循环
for batch in train_loader:
    inputs = batch['input']      # 已归一化的输入 [B, T, 6712]
    targets = batch['target']    # 已归一化的目标 [B, T, 6712]
    mask = batch['prediction_mask']  # 预测mask [B, 6712]
    normalized = batch['normalized']  # True，表示已归一化
    
    # 模型训练...
```

### 3. 可视化时跳过归一化

```python
# 创建可视化DataLoader（不归一化）
vis_loader = create_dataloader_with_normalization(
    val_dataset,
    batch_size=1,
    shuffle=False,
    normalizer_method='standard',
    apply_normalization=False  # 可视化时关闭归一化
)

# 数据仍然是原始尺度，方便可视化
for batch in vis_loader:
    inputs = batch['input']      # 原始尺度数据 [B, T, 6712]
    normalized = batch['normalized']  # False，表示未归一化
    
    # 可视化处理...
```

### 4. 推理时的反归一化

```python
from data.normalizer import DataNormalizer

# 加载归一化器
normalizer = DataNormalizer("/path/to/data", method='standard')
normalizer.load_stats()

# 模型推理
model.eval()
with torch.no_grad():
    for batch in val_loader:
        inputs = batch['input']  # 已归一化
        
        # 模型预测
        predictions = model(inputs)  # 归一化尺度的预测
        
        # 反归一化到原始尺度
        predictions_original = normalizer.inverse_transform(predictions)
        targets_original = normalizer.inverse_transform(batch['target'])
        
        # 计算原始尺度的损失
        loss = criterion(predictions_original, targets_original)
```

## 归一化方法

### Standard (Z-score) 归一化
```
normalized = (data - mean) / std
```

**特点：**
- 结果均值为0，标准差为1
- 适合正态分布数据
- 对异常值敏感

### MinMax 归一化
```
normalized = (data - min) / (max - min)
```

**特点：**
- 结果范围为[0, 1]
- 保持原始数据分布形状
- 对异常值非常敏感

### Robust 归一化
```
normalized = (data - median) / IQR
```
其中 IQR = Q75 - Q25

**特点：**
- 基于分位数，对异常值鲁棒
- 适合包含异常值的数据
- 结果不限于特定范围

## 文件存储

归一化统计量存储在 `{data_dir}/normalization_stats/` 目录下：

```
data/
├── normalization_stats/
│   ├── standard_stats.npz    # 标准化统计量
│   ├── minmax_stats.npz      # 最大最小值统计量
│   └── robust_stats.npz      # 鲁棒性统计量
```

每个文件包含：
- 归一化方法
- 维度信息
- 相应的统计量（均值、标准差等）

## 性能优化

### 内存控制
- 使用 `max_samples` 参数限制用于统计量计算的样本数
- 使用 `max_sequences_per_sample` 控制每个样本的序列数

### 计算效率
- 统计量只需计算一次，然后持久化存储
- 归一化在DataLoader中批量进行，利用GPU加速
- 支持流式计算，避免加载全部数据到内存

### 缓存优化
- DataLoader支持 `pin_memory` 和多进程数据加载
- 统计量文件使用压缩存储（npz格式）

## 故障排除

### 1. 统计量文件不存在
```
WARNING - Normalizer not found. Consider running compute_normalization_stats.py first
```
**解决方案：** 先运行统计量预计算脚本

### 2. 维度不匹配
```
ERROR - Dimension mismatch: expected 6712, got 6713
```
**解决方案：** 检查数据处理管道，确保维度一致

### 3. 数据类型错误
```
ERROR - Input data must be float32 for normalization
```
**解决方案：** 确保输入数据为float32类型

### 4. 常数列警告
```
WARNING - Found 5 constant columns, setting std=1
```
**说明：** 这是正常的，系统自动处理常数列以避免除零错误

## 迁移指南

### 从旧系统迁移

1. **更新Dataset创建**：
```python
# 旧代码
dataset = FluidDataset(data_dir, normalize=True, scaler_type='standard')

# 新代码
dataset = FluidDataset(data_dir)  # 不再需要归一化参数
```

2. **更新DataLoader创建**：
```python
# 旧代码
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# 新代码
loader = create_dataloader_with_normalization(
    dataset, batch_size=32, normalizer_method='standard'
)
```

3. **预计算统计量**：
```bash
python compute_normalization_stats.py --data_dir /path/to/data --method standard
```

### 兼容性

- 保持了主要API接口
- TensorDict格式不变
- 批次数据格式不变（增加了normalized标记）

## 最佳实践

1. **选择归一化方法**：
   - 数据正态分布 → standard
   - 需要固定范围 → minmax  
   - 存在异常值 → robust

2. **训练阶段**：
   - 始终开启归一化
   - 使用训练+验证集计算统计量
   - 监控归一化后的数据分布

3. **推理阶段**：
   - 使用相同的归一化方法
   - 预测后进行反归一化
   - 在原始尺度计算指标

4. **可视化阶段**：
   - 关闭归一化使用原始数据
   - 或者对归一化数据进行反变换

5. **版本管理**：
   - 保存统计量文件的版本
   - 确保训练和推理使用相同统计量
   - 定期更新统计量（当数据分布改变时）