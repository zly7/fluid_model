"""
Test transformers compatibility for FluidDecoder model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from models.decoder import FluidDecoder
from models.config import DecoderConfig

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")


def test_transformers_forward_signature():
    """测试transformers兼容的forward签名。"""
    print("Testing transformers forward signature...")
    
    config = DecoderConfig(
        d_model=128,
        n_heads=4,
        n_layers=2,
        sequence_length=3
    )
    model = FluidDecoder(config).to(device)
    
    # 测试数据
    batch_size = 1
    time_steps = 2
    num_variables = 6712
    
    input_tensor = torch.randn(batch_size, time_steps, num_variables).to(device)
    labels = torch.randn(batch_size, time_steps, num_variables).to(device)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # 测试1: 只有输入，无labels
    print("\n1. Testing forward without labels...")
    output = model(input_ids=input_tensor)
    
    assert isinstance(output, dict), "Output should be a dictionary"
    assert 'logits' in output, "Output should contain 'logits' key"
    assert output['logits'].shape == (batch_size, time_steps, num_variables), f"Expected shape {(batch_size, time_steps, num_variables)}, got {output['logits'].shape}"
    print(f"✓ Output logits shape: {output['logits'].shape}")
    
    # 测试2: 有输入和labels
    print("\n2. Testing forward with labels...")
    output_with_loss = model(input_ids=input_tensor, labels=labels)
    
    assert isinstance(output_with_loss, dict), "Output should be a dictionary"
    assert 'loss' in output_with_loss, "Output should contain 'loss' key"
    assert 'logits' in output_with_loss, "Output should contain 'logits' key"
    assert output_with_loss['logits'].shape == (batch_size, time_steps, num_variables), f"Expected shape {(batch_size, time_steps, num_variables)}, got {output_with_loss['logits'].shape}"
    assert isinstance(output_with_loss['loss'], torch.Tensor), "Loss should be a tensor"
    assert output_with_loss['loss'].ndim == 0, "Loss should be a scalar"
    print(f"✓ Output logits shape: {output_with_loss['logits'].shape}")
    print(f"✓ Loss value: {output_with_loss['loss'].item():.6f}")
    
    # 测试3: 兼容原有的batch字典格式
    print("\n3. Testing backward compatibility with batch dict...")
    batch_dict = {
        'input': input_tensor,
        'target': labels
    }
    output_dict = model(input_ids=batch_dict)
    
    assert isinstance(output_dict, dict), "Output should be a dictionary"
    assert 'loss' in output_dict, "Output should contain 'loss' key"
    assert 'logits' in output_dict, "Output should contain 'logits' key"
    print(f"✓ Backward compatibility maintained")
    print(f"✓ Loss with dict input: {output_dict['loss'].item():.6f}")
    
    print("\n✅ All transformers compatibility tests passed!")


def test_loss_computation():
    """测试损失计算。"""
    print("\nTesting loss computation...")
    
    config = DecoderConfig(
        d_model=64,
        n_heads=2,
        n_layers=1
    )
    model = FluidDecoder(config).to(device)
    
    batch_size = 1
    time_steps = 1
    num_variables = 6712
    
    predictions = torch.randn(batch_size, time_steps, num_variables).to(device)
    labels = torch.randn(batch_size, time_steps, num_variables).to(device)
    
    # 测试无mask的损失
    loss = model.compute_loss(predictions, labels)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be a scalar"
    print(f"✓ MSE loss without mask: {loss.item():.6f}")
    
    # 测试有mask的损失
    prediction_mask = torch.zeros(batch_size, num_variables).to(device)
    prediction_mask[:, :100] = 1  # 只预测前100个变量
    
    masked_loss = model.compute_loss(predictions, labels, prediction_mask)
    assert isinstance(masked_loss, torch.Tensor), "Masked loss should be a tensor"
    assert masked_loss.ndim == 0, "Masked loss should be a scalar"
    print(f"✓ MSE loss with mask: {masked_loss.item():.6f}")
    
    print("✅ Loss computation tests passed!")


def test_model_parameters():
    """测试模型参数。"""
    print("\nTesting model parameters...")
    
    config = DecoderConfig(
        d_model=256,
        n_heads=8,
        n_layers=4
    )
    model = FluidDecoder(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    assert total_params > 0, "Model should have parameters"
    assert trainable_params == total_params, "All parameters should be trainable"
    
    print("✅ Model parameter tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING TRANSFORMERS COMPATIBILITY")
    print("=" * 60)
    
    try:
        test_transformers_forward_signature()
        test_loss_computation()
        test_model_parameters()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Model is transformers compatible!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()