# PRP: Gas Pipeline Network Fluid Dynamics Decoder-Only Transformer Model

## Project Overview

Implementation of a decoder-only transformer architecture for predicting gas pipeline network fluid dynamics parameters at minute-level granularity. The system processes boundary conditions (gas sources, distribution points, compressor states, valve settings) to predict comprehensive pipeline system states including pressures, flows, temperatures, and equipment power consumption.

## Technical Context

### Data Architecture
- **Training Data**: 264 complete examples with boundary conditions + outputs
- **Test Data**: 30 examples with boundary conditions only  
- **Input**: Boundary.csv (TIME + 500+ parameters including T_xxx:SNQ, C_xxx:ST/SP_out, R_xxx:ST/SPD, B_xxx:FR)
- **Output**: 7 CSV files (B.csv, T&E.csv, H.csv, C.csv, N.csv, R.csv, P.csv) with equipment-specific parameters
- **Prediction**: Minute-level autoregressive forecasting from 30-minute boundary condition inputs

### Network Topology
- 572 nodes, 262 pipelines/segments, 343 ball valves, 23 compressors, 10 control valves
- 7 gas sources, 122 distribution points  
- Sparse connectivity graph (each node connects to few others)
- Fixed prediction dimension enabling minute-level forecasting

## Research-Based Architecture

### Decoder-Only Design (Based on Google TimesFM 2024)
- Implement code directly based on the decoder-only large model architecture with multi-head attention that you've learned from history.
- **Architecture**: Stacked transformer layers with self-attention and feedforward blocks
- **Token Creation**: MLP block converts time-series patches to transformer tokens

### PyTorch Implementation Best Practices (2024)
**Multi-Head Attention**:
- Use PyTorch's optimized `MultiheadAttention` with `need_weights=False` for best performance
- Leverages `scaled_dot_product_attention()` optimization
- **Reference**: https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

**Residual Connections**:
- Critical for gradient flow in deep transformers
- Prevents information loss after attention layers  

**Layer Normalization**:
- Pre-layer normalization (`norm_first=True`) recommended for training stability
- Addresses high gradient issues during early iterations
- 
### Positional Encoding Strategy
**Sinusoidal Fixed Encoding**:
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        # Standard sinusoidal implementation for time series
        # Reference: https://github.com/wzlxjtu/PositionalEncoding2D
```

**Time Series Considerations**:
- Use fixed positional encodings for consistent time step representation

## Implementation Blueprint

### Package Structure (Corrected - Start Fresh)
```
fluid_model/
├── data/           # 🆕 To implement (removed existing code)
│   ├── processor.py        # Raw data loading and preprocessing
│   ├── dataset.py          # PyTorch Dataset implementation
│   ├── loader.py           # DataLoader creation
│   └── __init__.py
├── visualization/  # 🆕 Critical for data understanding
│   ├── streamlit_app.py    # Interactive data exploration
│   ├── plotly_charts.py    # Chart generation functions
│   └── __init__.py
├── models/         # 🆕 To implement
│   ├── transformer.py      # DecoderOnlyTransformer
│   ├── positional.py       # PositionalEncoding
│   ├── embeddings.py       # Input/Output embeddings
│   └── __init__.py
├── training/       # 🆕 To implement  
│   ├── trainer.py          # Training/inference logic
│   ├── loss.py             # Loss functions
│   └── __init__.py
├── utils/          # 🆕 To implement
│   ├── metrics.py          # Evaluation metrics
│   ├── tensorboard_utils.py # TensorBoard logging
│   └── __init__.py
└── main.py         # 🆕 Entry point
```

### Core Architecture Components

#### 1. DecoderOnlyTransformer (models/transformer.py)
```python
class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only transformer for gas pipeline prediction
    - Input: Boundary conditions (normalized float values)
    - Output: Multi-target predictions for all equipment types
    - Architecture: Similar to GPT but for multivariate time series
    """
```

#### 2. Input Embedding Pipeline (models/embeddings.py)
```python 
class InputEmbedding(nn.Module):
    """
    MLP-based embedding: input_dim → 128 → 256 → d_model
    Handles normalization and non-linear transformation
    """
```

#### 3. Multi-Output Head (models/transformer.py)
```python
class MultiTargetHead(nn.Module):
    """
    Separate output heads for each equipment type:
    - B.csv: Ball valves (p_in, p_out, q_in, q_out, t_in, t_out)
    - C.csv: Compressors (+ pwr parameter)
    - etc.
    """
```

### Training Strategy (Based on 2024 Research)

#### Teacher Forcing vs Autoregressive
- **Training**: Teacher forcing with true previous outputs
- **Inference**: Autoregressive generation  
- **Validation**: Autoregressive mode to match inference
- **Reference**: https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/

#### Masking Strategy
- Causal masking for autoregressive behavior
- Look-ahead mask prevents future information leakage
- **Reference**: https://stackoverflow.com/questions/57099613/how-is-teacher-forcing-implemented-for-the-transformer-training

### TensorBoard Integration (2024 Best Practices)

#### Monitoring Setup
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/fluid_transformer')
# Track loss, learning rate, gradients, model graph
```

#### Key Metrics to Track
- Training/validation loss per equipment type
- Gradient norms and weight histograms
- Learning rate scheduling
- Model graph visualization
- **Reference**: https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

## Critical Context for Implementation


### Key Technical Insights
1. **Data Flow**: Boundary conditions (30-min intervals) → Equipment predictions (1-min intervals)
2. **Multi-target Learning**: 7 different equipment types with varying parameter counts
3. **Sparse Network**: Topology learning through self-attention (no pre-defined adjacency)
4. **Autoregressive**: Each minute predicts next minute's state

### External Dependencies
- PyTorch ≥ 2.6 (for optimized attention)
- pandas, numpy (data processing)
- streamlit (interactive data exploration)
- plotly (advanced visualizations)
- TensorBoard (experiment tracking)
- Optional: transformers library (training utilities)

## Implementation Task List

### Phase 1: Data Understanding and Preprocessing (CRITICAL FIRST STEP)
1. **Create data package** with `__init__.py`
2. **Implement raw data processor** (data/processor.py)
   - Load CSV files from train/test directories  
   - Parse boundary conditions and equipment outputs
   - Handle time series alignment (30min → 1min mapping)
   - Data cleaning and validation
3. **Create data visualization tools** (visualization/)
    可视化这里- B.csv (球阀): 2,058 (2,059-1)
  - C.csv (压缩机): 161 (162-1)
  - H.csv (管段): 192 (193-1)
  - N.csv (节点): 1,716 (1,717-1)
  - P.csv (管道): 1,610 (1,611-1)
  - R.csv (调节阀): 50 (51-1)
  - T&E.csv (气源和分输点): 387 (388-1)
  - 这些所有列都带有timestamp，可视化的时候表格就是每个指标一个格子，然后选择是第X个算例就可以了
   - **Streamlit dashboard** (visualization/streamlit_app.py)
     - Interactive data exploration interface
     - Time series plotting for boundary conditions
     - Equipment parameter distribution analysis
     - Correlation heatmaps between parameters
   - **Plotly chart functions** (visualization/plotly_charts.py)
     - Time series line plots with zoom capabilities
     - Multi-dimensional scatter plots
     - Parameter distribution histograms
     - Equipment-wise parameter comparisons
4. **Data validation and insights**
   - Verify data format understanding through visualization
   - Identify data patterns, outliers, missing values
   - Understand temporal relationships and dependencies
   - Validate 30-min boundary → 1-min prediction mapping

### Phase 2: PyTorch Data Infrastructure
5. **Implement PyTorch Dataset** (data/dataset.py)
   - Custom Dataset class for boundary→equipment mapping
   - Handle variable sequence lengths and equipment counts
   - Normalization and preprocessing pipelines
6. **Implement DataLoader creation** (data/loader.py)
   - Train/validation/test splits
   - Batch collation for variable-length sequences
   - Data augmentation if beneficial
  Boundary.csv: 539列已经去除TIMELINE列
  其它目标文件总维度: 6,174 (去除TIME列)
  - B.csv (球阀): 2,058 (2,059-1)
  - C.csv (压缩机): 161 (162-1)
  - H.csv (管段): 192 (193-1)
  - N.csv (节点): 1,716 (1,717-1)
  - P.csv (管道): 1,610 (1,611-1)
  - R.csv (调节阀): 50 (51-1)
  - T&E.csv (气源和分输点): 387 (388-1)
  其它，B,C,H,N,P,R,T&E列总共6,174列，所以每次dataloader应该准备539+6174=6713列

### Phase 3: Core Model Architecture
7. **Create models package** with `__init__.py`
8. **Implement PositionalEncoding** (models/positional.py)
   - Sinusoidal encoding with configurable max_len
   - Support for time series-specific positioning
9. **Implement InputEmbedding** (models/embeddings.py) 
   - MLP: input_dim → 128 → 256 → d_model
   - Include normalization and GELU activation
10. **Implement DecoderOnlyTransformer** (models/transformer.py)
    - Multi-head self-attention with causal masking
    - Pre-layer normalization architecture
    - Residual connections
    - Configurable depth and head count
11. **Implement MultiTargetHead** (models/transformer.py)
    - Separate linear heads for each equipment type
    - Handle different output dimensions per type

### Phase 4: Training Infrastructure  
12. **Create training package** with `__init__.py`
13. **Implement loss functions** (training/loss.py)
    - MSE loss for continuous predictions
    - Weighted loss for different equipment importance
14. **Implement Trainer class** (training/trainer.py)
    - Training loop with teacher forcing
    - Autoregressive validation
    - Checkpoint saving/loading
    - TensorBoard logging integration

### Phase 5: Utilities and Monitoring
15. **Create utils package** with `__init__.py`  
16. **Implement evaluation metrics** (utils/metrics.py)
    - MAE, RMSE, MAPE per equipment type
    - Sequence-level accuracy metrics
17. **Implement TensorBoard utilities** (utils/tensorboard_utils.py)
    - Scalar/histogram logging functions
    - Loss curves and gradient monitoring
    - Model architecture graph visualization

### Phase 6: Integration and Entry Point
18. **Create main.py** - Training/inference entry point
    - Argument parsing for hyperparameters
    - Model instantiation and data loading
    - Training execution with experiment tracking
    - Inference mode for test set predictions

## Validation Gates

### Phase 1 Validation: Data Understanding
```bash
# Install dependencies
pip install pandas numpy streamlit plotly

# Verify data loading and visualization
python -c "from data.processor import DataProcessor; dp = DataProcessor('data'); print('✅ Data loading successful')"

# Launch Streamlit dashboard for data exploration
streamlit run visualization/streamlit_app.py
```

### Phase 2 Validation: PyTorch Data Pipeline  
```bash
# Test PyTorch dataset and dataloader
python -c "
from data.dataset import FluidDataset
from data.loader import create_data_loaders

# Test dataset creation
train_loader, val_loader, test_loader = create_data_loaders('data', batch_size=2)
batch = next(iter(train_loader))
print(f'✅ Batch shape: {batch[0].shape if isinstance(batch, tuple) else batch.shape}')
"
```

### Phase 3+ Validation: Model and Training
```bash
# Install remaining dependencies  
pip install torch tensorboard

# Basic model instantiation test
python -c "
from models.transformer import DecoderOnlyTransformer

model = DecoderOnlyTransformer(
    input_dim=500, d_model=256, nhead=8, 
    num_decoder_layers=6, max_seq_len=1500
)
print('✅ Model instantiation successful')
"

# Quick training verification (1 epoch, small batch)
python main.py --epochs 1 --batch_size 2 --d_model 128 --nhead 4 --num_layers 2

# Launch TensorBoard to verify logging
tensorboard --logdir=runs/ --port=6006
```

## Error Handling Strategy

### Common Implementation Pitfalls
1. **Data Understanding**: Misunderstanding time series alignment (30min boundary → 1min predictions)
2. **CSV Structure**: Different equipment types have different parameter counts and naming
3. **Dimension Mismatches**: Carefully track tensor shapes through embedding → transformer → output heads
4. **Masking Errors**: Ensure causal mask prevents future information leakage  
5. **Memory Issues**: Large sequences (1441 time steps) may require gradient checkpointing
6. **Equipment Mapping**: Handle variable equipment counts across different CSV files

### Debugging Approach
1. **Start with Data**: CRITICAL - Use Streamlit dashboard to verify data understanding
2. **Visualize First**: Plot time series, distributions, correlations before modeling
3. **Start Small**: Begin with minimal model (d_model=64, 1 layer) 
4. **Incremental Building**: Add complexity after each component works
5. **Shape Debugging**: Add print statements for tensor dimensions
6. **Gradient Monitoring**: Use TensorBoard to catch vanishing/exploding gradients

## Success Metrics

### Functional Requirements  
- [ ] **Phase 1**: Data processor loads all CSV files correctly
- [ ] **Phase 1**: Streamlit dashboard displays data insights clearly  
- [ ] **Phase 1**: Data structure and temporal relationships validated through visualization
- [ ] **Phase 2**: PyTorch Dataset and DataLoader work without errors
- [ ] **Phase 3+**: Model trains without errors on sample data
- [ ] **Phase 3+**: Predictions generated for all equipment types
- [ ] **Phase 3+**: TensorBoard logging captures training progress
- [ ] **Phase 3+**: Autoregressive inference produces realistic outputs

### Performance Benchmarks (Optional)
- Training converges within 50 epochs
- Validation loss decreases consistently  
- Prediction accuracy competitive with baseline methods

## Confidence Score: 9/10

This PRP provides comprehensive context for one-pass implementation success:

**Strengths:**
- ✅ Thorough research-based architecture decisions
- ✅ Clear understanding of existing codebase patterns
- ✅ Detailed task breakdown with validation gates
- ✅ Modern best practices from 2024 research
- ✅ Practical implementation guidance

**Risk Mitigation:**
- 📋 Incremental testing approach
- 📋 Clear error handling strategy  
- 📋 Fallback to simpler architectures if needed
- 📋 Existing data infrastructure reduces implementation risk

The combination of thorough research, existing code analysis, and structured implementation plan makes this highly likely to succeed in a single implementation pass.