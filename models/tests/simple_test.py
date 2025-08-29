"""
简化的模型测试文件
"""

import torch
import sys
import os

# 添加路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.decoder import FluidDecoder
    from models.config import DecoderConfig
    print("模块导入成功")
except ImportError as e:
    print(f"导入错误: {e}")
    exit(1)

def test_simple():
    """简单的维度测试"""
    print("开始简单测试...")
    
    # 使用小配置
    config = DecoderConfig(
        d_model=64,  # 减小模型大小
        n_heads=4,
        n_layers=2,
        d_ff=128,
        input_dim=100,  # 减小输入维度
        output_dim=100,
        boundary_dims=40,  # 边界变量数量，小于总变量数
        equipment_dims=60,  # 设备变量数量
        max_time_positions=3,
        max_variable_positions=100
    )
    
    print(f"配置: d_model={config.d_model}, n_heads={config.n_heads}")
    
    # 创建模型
    try:
        model = FluidDecoder(config)
        print("模型创建成功")
    except Exception as e:
        print(f"模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试数据
    batch_size = 1
    time_steps = 2
    num_variables = 100
    
    input_data = torch.randn(batch_size, time_steps, num_variables)
    target_data = torch.randn(batch_size, time_steps, num_variables)
    prediction_mask = torch.ones(batch_size, num_variables)  # 简化mask
    
    batch = {
        'input': input_data,
        'target': target_data,
        'prediction_mask': prediction_mask
    }
    
    print(f"输入维度: {batch['input'].shape}")
    
    # 前向传播测试
    try:
        model.eval()
        with torch.no_grad():
            predictions = model(batch)
        print(f"输出维度: {predictions.shape}")
        print("前向传播成功")
        return True
    except Exception as e:
        print(f"前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple()
    if success:
        print("测试通过！")
    else:
        print("测试失败！")