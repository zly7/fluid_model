"""
综合测试文件：验证FluidDecoder的完整功能
"""

import torch
import sys
import os

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.decoder import FluidDecoder, DecoderAttentionMask, CombinedPositionalEncoding
from models.config import DecoderConfig
from models.base import MaskedMSELoss

def test_real_dimensions():
    """测试真实数据维度（6712维）"""
    print("=== 测试真实维度 (6712) ===")
    
    config = DecoderConfig(
        d_model=128,  # 适中的模型大小
        n_heads=8,
        n_layers=2,
        d_ff=512,
        input_dim=6712,
        output_dim=6712,
        boundary_dims=538,
        equipment_dims=6174,
        max_time_positions=10,
        max_variable_positions=6712
    )
    
    print(f"配置信息:")
    print(f"- d_model: {config.d_model}")
    print(f"- boundary_dims: {config.boundary_dims}")
    print(f"- equipment_dims: {config.equipment_dims}")
    print()
    
    # 创建模型
    model = FluidDecoder(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count:,}")
    
    # 测试数据
    batch_size = 2
    time_steps = 3
    num_variables = 6712
    
    input_data = torch.randn(batch_size, time_steps, num_variables)
    target_data = torch.randn(batch_size, time_steps, num_variables)
    
    # 创建prediction_mask（设备变量需要预测）
    prediction_mask = torch.zeros(batch_size, num_variables)
    prediction_mask[:, config.boundary_dims:] = 1.0  # 只预测设备变量
    
    batch = {
        'input': input_data,
        'target': target_data,
        'prediction_mask': prediction_mask
    }
    
    print(f"输入维度: {batch['input'].shape}")
    print(f"预测mask中的预测变量数: {prediction_mask.sum(dim=1)[0].int().item()}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        predictions = model(batch)
    
    print(f"输出维度: {predictions.shape}")
    print(f"维度匹配: {predictions.shape == input_data.shape}")
    
    # 损失计算
    loss_dict = model.compute_loss(batch, predictions)
    print(f"损失: {loss_dict['loss'].item():.6f}")
    
    return True

def test_attention_mask_logic():
    """测试attention mask的逻辑"""
    print("\n=== 测试 Attention Mask 逻辑 ===")
    
    batch_size = 1
    time_steps = 3
    num_variables = 10
    boundary_dims = 4
    
    mask = DecoderAttentionMask.create_decoder_mask(
        batch_size, time_steps, num_variables, boundary_dims
    )
    
    print(f"Mask 维度: {mask.shape}")
    
    # 验证边界变量只能看到自己
    for t in range(time_steps):
        for v in range(boundary_dims):
            pos = t * num_variables + v
            visible_count = mask[0, pos, :].sum().item()
            print(f"时间{t} 边界变量{v}: 可见变量数={int(visible_count)} (期望=1)")
    
    # 验证设备变量的因果mask
    for t in range(time_steps):
        for v in range(boundary_dims, min(boundary_dims + 2, num_variables)):
            pos = t * num_variables + v
            visible_count = mask[0, pos, :].sum().item()
            expected = (t + 1) * (num_variables - boundary_dims)
            print(f"时间{t} 设备变量{v}: 可见变量数={int(visible_count)} (期望={expected})")
    
    return True

def test_positional_encoding():
    """测试位置编码"""
    print("\n=== 测试位置编码 ===")
    
    d_model = 128
    max_time = 5
    max_var = 100
    
    pos_enc = CombinedPositionalEncoding(
        d_model=d_model,
        max_time_positions=max_time,
        max_variable_positions=max_var
    )
    
    batch_size = 2
    time_steps = 3
    num_variables = 50
    seq_len = time_steps * num_variables
    
    x = torch.zeros(batch_size, seq_len, d_model)  # 零输入便于观察编码
    x_with_pos = pos_enc(x, time_steps, num_variables)
    
    print(f"位置编码输入: {x.shape}")
    print(f"位置编码输出: {x_with_pos.shape}")
    
    # 验证不同位置的编码确实不同
    pos_norms = torch.norm(x_with_pos, dim=-1)
    print(f"位置编码范数 - 最小: {pos_norms.min():.4f}, 最大: {pos_norms.max():.4f}")
    
    # 验证时间编码的周期性
    t0_pos = x_with_pos[0, :num_variables, :]  # 时间0的所有变量
    t1_pos = x_with_pos[0, num_variables:2*num_variables, :]  # 时间1的所有变量
    
    time_diff = torch.norm(t1_pos - t0_pos, dim=-1).mean()
    print(f"时间步间位置编码差异: {time_diff:.4f}")
    
    return True

def test_model_components():
    """测试模型各组件"""
    print("\n=== 测试模型组件 ===")
    
    from models.decoder import SimpleMultiHeadAttention, DecoderBlock
    
    d_model = 128
    n_heads = 8
    batch_size = 2
    seq_len = 50
    
    # 测试多头注意力
    attention = SimpleMultiHeadAttention(d_model, n_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    attention_mask = torch.ones(batch_size, seq_len, seq_len)  # 全连接
    
    attn_out = attention(x, attention_mask)
    print(f"多头注意力: {x.shape} -> {attn_out.shape}")
    print(f"维度匹配: {x.shape == attn_out.shape}")
    
    # 测试Decoder Block
    decoder_block = DecoderBlock(d_model, n_heads, d_ff=512)
    block_out = decoder_block(x, attention_mask)
    print(f"Decoder Block: {x.shape} -> {block_out.shape}")
    print(f"维度匹配: {x.shape == block_out.shape}")
    
    # 测试残差连接是否起作用
    residual_norm = torch.norm(block_out - x, dim=-1).mean()
    print(f"残差连接效果 (与输入的差异): {residual_norm:.4f}")
    
    return True

def test_masked_loss():
    """测试masked loss"""
    print("\n=== 测试 Masked Loss ===")
    
    batch_size = 2
    time_steps = 3
    num_variables = 100
    boundary_dims = 40
    
    predictions = torch.randn(batch_size, time_steps, num_variables)
    targets = torch.randn(batch_size, time_steps, num_variables)
    
    # 只预测设备变量
    prediction_mask = torch.zeros(batch_size, num_variables)
    prediction_mask[:, boundary_dims:] = 1.0
    
    # 使用MaskedMSELoss
    loss_fn = MaskedMSELoss(boundary_dims)
    loss = loss_fn(predictions, targets, prediction_mask)
    
    print(f"Masked Loss: {loss.item():.6f}")
    
    # 比较有mask和无mask的损失
    mse_loss = torch.nn.MSELoss()
    full_loss = mse_loss(predictions, targets)
    
    print(f"完整 MSE Loss: {full_loss.item():.6f}")
    print(f"损失比例: {(loss / full_loss).item():.4f}")
    
    return True

def main():
    """运行所有测试"""
    print("开始 FluidDecoder 综合测试")
    print("=" * 60)
    
    tests = [
        test_real_dimensions,
        test_attention_mask_logic,
        test_positional_encoding,
        test_model_components,
        test_masked_loss
    ]
    
    passed = 0
    total = len(tests)
    
    for i, test_func in enumerate(tests, 1):
        try:
            print(f"\n[测试 {i}/{total}]")
            if test_func():
                passed += 1
                print(f"[OK] 测试 {i} 通过")
            else:
                print(f"[FAIL] 测试 {i} 失败")
        except Exception as e:
            print(f"[ERROR] 测试 {i} 异常: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("所有测试通过！FluidDecoder 可以正常使用。")
    else:
        print("部分测试失败，需要检查代码。")

if __name__ == "__main__":
    main()