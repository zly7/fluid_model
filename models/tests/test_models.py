"""
测试文件：验证模型的输入输出维度和功能
"""

import torch
import torch.nn as nn
import sys
import os

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FluidDecoder, DecoderConfig, BaseModel
from models.base import MaskedMSELoss


def test_decoder_model():
    """测试 FluidDecoder 模型的维度和功能"""
    print("=" * 60)
    print("测试 FluidDecoder 模型")
    print("=" * 60)
    
    # 模型配置
    config = DecoderConfig(
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        input_dim=6712,
        output_dim=6712,
        max_time_positions=10,
        max_variable_positions=6712
    )
    
    print(f"配置信息:")
    print(f"- d_model: {config.d_model}")
    print(f"- n_heads: {config.n_heads}")
    print(f"- n_layers: {config.n_layers}")
    print(f"- input_dim: {config.input_dim}")
    print(f"- output_dim: {config.output_dim}")
    print(f"- boundary_dims: {config.boundary_dims}")
    print()
    
    # 创建模型
    model = FluidDecoder(config)
    
    # 模型信息
    model_info = model.get_model_info()
    print(f"模型信息:")
    print(f"- 总参数量: {model_info['total_parameters']:,}")
    print(f"- 可训练参数量: {model_info['trainable_parameters']:,}")
    print()
    
    # 测试数据维度
    batch_size = 2
    time_steps = 3
    num_variables = 6712
    
    print(f"测试维度:")
    print(f"- batch_size: {batch_size}")
    print(f"- time_steps: {time_steps}")
    print(f"- num_variables: {num_variables}")
    print()
    
    # 创建测试数据
    input_data = torch.randn(batch_size, time_steps, num_variables)
    target_data = torch.randn(batch_size, time_steps, num_variables)
    
    # 创建 prediction_mask (设备变量需要预测，边界变量不需要)
    prediction_mask = torch.zeros(batch_size, num_variables)
    # 前538维是边界变量，设为0（不预测）
    # 后面的维度是设备变量，设为1（预测）
    prediction_mask[:, config.boundary_dims:] = 1.0
    
    # 创建 batch 字典
    batch = {
        'input': input_data,
        'target': target_data,
        'prediction_mask': prediction_mask
    }
    
    print(f"输入维度检查:")
    print(f"- input: {batch['input'].shape}")
    print(f"- target: {batch['target'].shape}")
    print(f"- prediction_mask: {batch['prediction_mask'].shape}")
    print(f"- 边界变量数量: {config.boundary_dims}")
    print(f"- 设备变量数量: {config.equipment_dims}")
    print()
    
    # 前向传播测试
    model.eval()
    with torch.no_grad():
        predictions = model(batch)
    
    print(f"前向传播结果:")
    print(f"- 预测输出维度: {predictions.shape}")
    print(f"- 预期维度: ({batch_size}, {time_steps}, {num_variables})")
    print(f"- 维度匹配: {predictions.shape == (batch_size, time_steps, num_variables)}")
    print()
    
    # 损失计算测试
    loss_dict = model.compute_loss(batch, predictions)
    
    print(f"损失计算结果:")
    for loss_name, loss_value in loss_dict.items():
        print(f"- {loss_name}: {loss_value.item():.6f}")
    print()
    
    # 测试 MaskedMSELoss
    print("测试 MaskedMSELoss:")
    masked_loss_fn = MaskedMSELoss(boundary_dims=config.boundary_dims)
    masked_loss = masked_loss_fn(predictions, target_data, prediction_mask)
    print(f"- MaskedMSELoss: {masked_loss.item():.6f}")
    print()
    
    print("[OK] FluidDecoder 模型测试通过！")
    return True


def test_attention_mask():
    """测试注意力mask的生成"""
    print("=" * 60)
    print("测试注意力 Mask 生成")
    print("=" * 60)
    
    from models.decoder import DecoderAttentionMask
    
    batch_size = 1
    time_steps = 3
    num_variables = 10  # 简化测试
    boundary_dims = 4
    
    # 生成 attention mask
    attention_mask = DecoderAttentionMask.create_decoder_mask(
        batch_size=batch_size,
        time_steps=time_steps,
        num_variables=num_variables,
        boundary_dims=boundary_dims
    )
    
    print(f"Attention mask 维度: {attention_mask.shape}")
    print(f"预期维度: ({batch_size}, {time_steps * num_variables}, {time_steps * num_variables})")
    print()
    
    # 可视化 mask（只显示第一个batch）
    mask_2d = attention_mask[0].cpu().numpy()
    
    print("Attention mask 可视化 (1=可见, 0=不可见):")
    print("行: query位置, 列: key位置")
    print("变量索引: [B0-B3: boundary, E4-E9: equipment] × 3个时间步")
    print()
    
    # 打印mask矩阵
    seq_len = time_steps * num_variables
    for i in range(min(seq_len, 15)):  # 只显示前15行
        row = ""
        for j in range(min(seq_len, 15)):  # 只显示前15列
            row += f"{int(mask_2d[i, j])} "
        
        # 添加位置说明
        t_i = i // num_variables
        v_i = i % num_variables
        var_type = "B" if v_i < boundary_dims else "E"
        row += f"  <- t{t_i}_{var_type}{v_i}"
        
        print(row)
    
    print()
    
    # 验证mask的逻辑
    print("验证 mask 逻辑:")
    
    # 检查边界变量的mask（只能看到自己）
    for t in range(time_steps):
        for v in range(boundary_dims):
            pos = t * num_variables + v
            visible_positions = torch.sum(attention_mask[0, pos, :]).item()
            print(f"- 时间{t} 边界变量{v} 可见位置数: {int(visible_positions)} (应该为1)")
    
    print()
    
    # 检查设备变量的mask
    for t in range(time_steps):
        for v in range(boundary_dims, min(boundary_dims + 3, num_variables)):
            pos = t * num_variables + v
            visible_positions = torch.sum(attention_mask[0, pos, :]).item()
            expected_positions = (t + 1) * (num_variables - boundary_dims)
            print(f"- 时间{t} 设备变量{v} 可见位置数: {int(visible_positions)} (应该为{expected_positions})")
    
    print()
    print("[OK] Attention mask 测试通过！")
    return True


def test_positional_encoding():
    """测试位置编码"""
    print("=" * 60)
    print("测试组合位置编码")
    print("=" * 60)
    
    from models.decoder import CombinedPositionalEncoding
    
    d_model = 128
    max_time_positions = 5
    max_variable_positions = 10
    
    # 测试正弦余弦编码
    pos_encoding = CombinedPositionalEncoding(
        d_model=d_model,
        max_time_positions=max_time_positions,
        max_variable_positions=max_variable_positions,
        time_encoding_type="sinusoidal",
        variable_encoding_type="sinusoidal"
    )
    
    batch_size = 2
    time_steps = 3
    num_variables = 8
    seq_len = time_steps * num_variables
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 添加位置编码
    x_with_pos = pos_encoding(x, time_steps, num_variables)
    
    print(f"位置编码测试:")
    print(f"- d_model: {d_model}")
    print(f"- 输入维度: {x.shape}")
    print(f"- 输出维度: {x_with_pos.shape}")
    print(f"- 维度匹配: {x.shape == x_with_pos.shape}")
    print()
    
    # 验证位置编码不是零向量
    pos_diff = torch.norm(x_with_pos - x, dim=-1)
    print(f"位置编码影响检查:")
    print(f"- 最小差异: {pos_diff.min().item():.6f}")
    print(f"- 最大差异: {pos_diff.max().item():.6f}")
    print(f"- 平均差异: {pos_diff.mean().item():.6f}")
    print()
    
    print("[OK] 位置编码测试通过！")
    return True


def test_model_components():
    """测试模型各个组件"""
    print("=" * 60)
    print("测试模型组件")
    print("=" * 60)
    
    # 测试多头注意力
    from models.decoder import SimpleMultiHeadAttention
    
    d_model = 256
    n_heads = 8
    batch_size = 2
    seq_len = 20
    
    attention = SimpleMultiHeadAttention(d_model, n_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建简单mask
    attention_mask = torch.ones(batch_size, seq_len, seq_len)
    
    # 前向传播
    attn_output = attention(x, attention_mask)
    
    print(f"多头注意力测试:")
    print(f"- 输入维度: {x.shape}")
    print(f"- 输出维度: {attn_output.shape}")
    print(f"- 维度匹配: {x.shape == attn_output.shape}")
    print()
    
    # 测试 Decoder Block
    from models.decoder import DecoderBlock
    
    decoder_block = DecoderBlock(d_model, n_heads, d_ff=1024)
    block_output = decoder_block(x, attention_mask)
    
    print(f"Decoder Block 测试:")
    print(f"- 输入维度: {x.shape}")
    print(f"- 输出维度: {block_output.shape}")
    print(f"- 维度匹配: {x.shape == block_output.shape}")
    print()
    
    print("[OK] 模型组件测试通过！")
    return True


def main():
    """运行所有测试"""
    print("开始模型测试...")
    print("=" * 80)
    
    try:
        # 运行所有测试
        tests = [
            test_decoder_model,
            test_attention_mask,
            test_positional_encoding,
            test_model_components
        ]
        
        passed_tests = 0
        for i, test_func in enumerate(tests, 1):
            print(f"\n{'='*20} 测试 {i}/{len(tests)} {'='*20}")
            try:
                if test_func():
                    passed_tests += 1
                    print(f"[OK] 测试 {i} 通过")
                else:
                    print(f"[FAIL] 测试 {i} 失败")
            except Exception as e:
                print(f"[ERROR] 测试 {i} 出现异常: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 80)
        print(f"测试总结: {passed_tests}/{len(tests)} 测试通过")
        
        if passed_tests == len(tests):
            print("所有测试通过！模型可以正常使用。")
        else:
            print("部分测试失败，请检查代码。")
            
    except Exception as e:
        print(f"测试运行出现严重错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()