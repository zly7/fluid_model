"""
Test LSTM model compatibility with transformers and dimensions.
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.lstm import FluidLSTM, LSTMConfig


class TestLSTMDimensions:
    """Test LSTM model input/output dimensions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 2
        self.time_steps = 3
        self.input_dim = 6712
        self.output_dim = 6712
        
        # Create test config
        self.config = LSTMConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=128,
            num_layers=1,
            bidirectional=False,
            dropout_rate=0.1,
            boundary_hidden_dim=64,
            equipment_hidden_dim=128,
            projection_hidden_dim=256
        )
        
        # Create model
        self.model = FluidLSTM(self.config)
        self.model.eval()
    
    def test_model_creation(self):
        """Test basic model creation."""
        assert self.model is not None
        assert isinstance(self.model, FluidLSTM)
        print("OK LSTM model created successfully")
    
    def test_input_output_dimensions(self):
        """Test input/output dimension compatibility."""
        # Create test input
        batch_input = torch.randn(self.batch_size, self.time_steps, self.input_dim)
        
        with torch.no_grad():
            # Test forward pass
            output = self.model(batch_input)
            
            # Check output format
            assert isinstance(output, dict), "Output should be a dict for transformers compatibility"
            assert 'logits' in output, "Output should contain 'logits' key"
            
            logits = output['logits']
            expected_shape = (self.batch_size, self.time_steps, self.output_dim)
            assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
            
        print(f"OK Input/output dimensions correct: {logits.shape}")
    
    def test_transformers_compatibility(self):
        """Test compatibility with transformers format."""
        # Create test batch
        batch_input = torch.randn(self.batch_size, self.time_steps, self.input_dim)
        batch_target = torch.randn(self.batch_size, self.time_steps, self.output_dim)
        
        with torch.no_grad():
            # Test without labels (inference mode)
            output_inference = self.model(batch_input)
            assert isinstance(output_inference, dict)
            assert 'logits' in output_inference
            assert 'loss' not in output_inference
            
            # Test with labels (training mode)
            output_training = self.model(batch_input, labels=batch_target)
            assert isinstance(output_training, dict)
            assert 'logits' in output_training
            assert 'loss' in output_training
            
            # Check loss is scalar
            loss = output_training['loss']
            assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
            assert loss.item() >= 0, "Loss should be non-negative"
            
        print("OK Transformers compatibility verified")
    
    def test_batch_format_compatibility(self):
        """Test compatibility with original batch format."""
        # Create batch in original format
        batch = {
            'input': torch.randn(self.batch_size, self.time_steps, self.input_dim),
            'target': torch.randn(self.batch_size, self.time_steps, self.output_dim),
            'prediction_mask': torch.ones(self.batch_size, self.output_dim)
        }
        
        with torch.no_grad():
            # Test with batch format
            output = self.model(batch)
            
            assert isinstance(output, dict)
            assert 'logits' in output
            assert 'loss' in output  # Should include loss since target is provided
            
            logits = output['logits']
            expected_shape = (self.batch_size, self.time_steps, self.output_dim)
            assert logits.shape == expected_shape
            
        print("OK Batch format compatibility verified")
    
    def test_prediction_mask(self):
        """Test prediction mask functionality."""
        # Create test data with prediction mask
        batch_input = torch.randn(self.batch_size, self.time_steps, self.input_dim)
        batch_target = torch.randn(self.batch_size, self.time_steps, self.output_dim)
        
        # Create prediction mask (predict only equipment variables, not boundary)
        prediction_mask = torch.zeros(self.batch_size, self.output_dim)
        prediction_mask[:, 538:] = 1  # Only equipment variables
        
        with torch.no_grad():
            # Test with prediction mask
            output = self.model(batch_input, labels=batch_target, prediction_mask=prediction_mask)
            
            assert 'loss' in output
            assert 'logits' in output
            
            # Loss should be computed only on masked variables
            loss = output['loss']
            assert loss.item() >= 0
            
        print("OK Prediction mask functionality verified")
    
    def test_model_parameters(self):
        """Test model parameter counts and info."""
        # Get model info
        info = self.model.get_model_info()
        
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
        
        # Check LSTM-specific info
        assert 'hidden_dim' in info
        assert 'num_layers' in info
        assert 'bidirectional' in info
        assert info['hidden_dim'] == self.config.hidden_dim
        assert info['num_layers'] == self.config.num_layers
        
        print(f"OK Model has {info['total_parameters']:,} total parameters")
        print(f"OK Model has {info['trainable_parameters']:,} trainable parameters")
    
    def test_different_config_sizes(self):
        """Test different model configurations."""
        configs = [
            # Small model
            LSTMConfig(
                hidden_dim=64,
                num_layers=1,
                bidirectional=False,
                boundary_hidden_dim=32,
                equipment_hidden_dim=64
            ),
            # Medium model with attention
            LSTMConfig(
                hidden_dim=256,
                num_layers=2,
                bidirectional=True,
                use_attention=True,
                attention_heads=4,
                boundary_hidden_dim=128,
                equipment_hidden_dim=256
            )
        ]
        
        for i, config in enumerate(configs):
            model = FluidLSTM(config)
            model.eval()
            
            # Test forward pass
            batch_input = torch.randn(2, 3, 6712)
            with torch.no_grad():
                output = model(batch_input)
                assert 'logits' in output
                assert output['logits'].shape == (2, 3, 6712)
            
            print(f"OK Config {i+1} works correctly")
    
    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        self.model.train()
        
        # Create test data
        batch_input = torch.randn(self.batch_size, self.time_steps, self.input_dim, requires_grad=True)
        batch_target = torch.randn(self.batch_size, self.time_steps, self.output_dim)
        
        # Forward pass
        output = self.model(batch_input, labels=batch_target)
        loss = output['loss']
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        param_count_with_grad = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter: {name}"
                param_count_with_grad += 1
        
        assert param_count_with_grad > 0, "No parameters received gradients"
        print(f"OK Gradients flow properly to {param_count_with_grad} parameters")


def main():
    """Run all tests."""
    print("Running LSTM Model Compatibility Tests")
    print("=" * 50)
    
    test_class = TestLSTMDimensions()
    test_class.setup_method()
    
    # Run all tests
    try:
        test_class.test_model_creation()
        test_class.test_input_output_dimensions()
        test_class.test_transformers_compatibility()
        test_class.test_batch_format_compatibility()
        test_class.test_prediction_mask()
        test_class.test_model_parameters()
        test_class.test_different_config_sizes()
        test_class.test_gradient_flow()
        
        print("\n" + "=" * 50)
        print("SUCCESS: All LSTM model tests passed!")
        
    except Exception as e:
        print(f"\nERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)