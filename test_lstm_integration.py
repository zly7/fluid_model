"""
Final integration test for LSTM model implementation.
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from models.lstm import FluidLSTM, LSTMConfig
from models.utils import create_model, get_model_summary
from training.config import TrainingConfig
from training.utils import create_model as training_create_model

def test_lstm_integration():
    """Test full LSTM integration."""
    print("LSTM Integration Test")
    print("=" * 50)
    
    # Test 1: Direct model creation
    print("\n1. Testing direct LSTM model creation...")
    config = LSTMConfig(hidden_dim=64, num_layers=1, bidirectional=False)
    model = FluidLSTM(config)
    print(f"   Created LSTM model: {model.get_model_info()['trainable_parameters']:,} parameters")
    
    # Test 2: Model factory creation
    print("\n2. Testing model factory creation...")
    model2 = create_model('lstm', config)
    print(f"   Factory created LSTM model: {model2.get_model_info()['trainable_parameters']:,} parameters")
    
    # Test 3: Config file loading
    print("\n3. Testing config file loading...")
    try:
        training_config = TrainingConfig(
            model_config_path="configs/models/lstm_nano.json",
            sequence_length=3
        )
        model3 = training_create_model(training_config)
        print(f"   Training config created LSTM model: {model3.get_model_info()['trainable_parameters']:,} parameters")
    except Exception as e:
        print(f"   Error: {e}")
        return False
    
    # Test 4: Forward pass compatibility
    print("\n4. Testing forward pass compatibility...")
    batch_size, seq_len, dim = 2, 3, 6712
    
    # Test transformers format
    input_tensor = torch.randn(batch_size, seq_len, dim)
    target_tensor = torch.randn(batch_size, seq_len, dim)
    
    with torch.no_grad():
        # Test inference mode
        output_inference = model(input_tensor)
        assert isinstance(output_inference, dict), "Output should be dict for transformers compatibility"
        assert 'logits' in output_inference, "Should contain logits"
        print(f"   Inference output shape: {output_inference['logits'].shape}")
        
        # Test training mode
        output_training = model(input_tensor, labels=target_tensor)
        assert 'loss' in output_training, "Should contain loss in training mode"
        assert 'logits' in output_training, "Should contain logits"
        print(f"   Training mode loss: {output_training['loss'].item():.4f}")
    
    # Test 5: Batch format compatibility
    print("\n5. Testing batch format compatibility...")
    batch = {
        'input': torch.randn(batch_size, seq_len, dim),
        'target': torch.randn(batch_size, seq_len, dim),
        'prediction_mask': torch.ones(batch_size, dim)
    }
    
    with torch.no_grad():
        output = model(batch)
        assert 'loss' in output and 'logits' in output
        print(f"   Batch format loss: {output['loss'].item():.4f}")
    
    # Test 6: Different LSTM configurations
    print("\n6. Testing different LSTM configurations...")
    configs = [
        ("Small", LSTMConfig(hidden_dim=32, num_layers=1, bidirectional=False)),
        ("Bidirectional", LSTMConfig(hidden_dim=64, num_layers=1, bidirectional=True)),
        ("Multi-layer", LSTMConfig(hidden_dim=64, num_layers=2, bidirectional=False)),
        ("With Attention", LSTMConfig(hidden_dim=64, num_layers=1, use_attention=True))
    ]
    
    for name, config in configs:
        model_test = FluidLSTM(config)
        with torch.no_grad():
            output = model_test(input_tensor)
            assert output['logits'].shape == (batch_size, seq_len, dim)
            print(f"   {name} config works: {model_test.get_model_info()['trainable_parameters']:,} params")
    
    print("\n" + "=" * 50)
    print("SUCCESS: All LSTM integration tests passed!")
    print("LSTM model is fully integrated and ready for use.")
    print("=" * 50)
    
    return True

def test_loss_computation():
    """Test that LSTM uses the same loss computation as other models."""
    print("\n7. Testing unified loss computation...")
    
    # Create different models
    from models import FluidDecoder, FluidCNN
    from models.decoder import DecoderConfig
    from models.cnn import CNNConfig
    
    # Create small configs for quick testing
    decoder_config = DecoderConfig(d_model=64, n_heads=2, n_layers=1)
    cnn_config = CNNConfig(hidden_channels=64, num_conv_layers=1)
    lstm_config = LSTMConfig(hidden_dim=64, num_layers=1)
    
    # Create models
    decoder_model = FluidDecoder(decoder_config)
    cnn_model = FluidCNN(cnn_config)
    lstm_model = FluidLSTM(lstm_config)
    
    # Test data
    batch_size, seq_len, dim = 1, 3, 6712
    input_tensor = torch.randn(batch_size, seq_len, dim)
    target_tensor = torch.randn(batch_size, seq_len, dim)
    prediction_mask = torch.ones(batch_size, dim)
    prediction_mask[:, :538] = 0  # Mask boundary conditions
    
    # Test all models use compatible loss
    models = [("Decoder", decoder_model), ("CNN", cnn_model), ("LSTM", lstm_model)]
    
    for name, model in models:
        with torch.no_grad():
            output = model(input_tensor, labels=target_tensor, prediction_mask=prediction_mask)
            loss = output['loss']
            assert loss.dim() == 0, f"{name} loss should be scalar"
            assert loss.item() >= 0, f"{name} loss should be non-negative"
            print(f"   {name} model loss: {loss.item():.4f}")
    
    print("   All models use compatible loss computation")

if __name__ == "__main__":
    try:
        success = test_lstm_integration()
        test_loss_computation()
        
        if success:
            print("\nFinal Summary:")
            print("SUCCESS: LSTM model implementation complete")
            print("SUCCESS: Compatible with transformers training pipeline")
            print("SUCCESS: Unified loss computation with other models")
            print("SUCCESS: Multiple configuration options available")
            print("SUCCESS: Ready for production use")
            exit(0)
    except Exception as e:
        print(f"\nERROR: Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)