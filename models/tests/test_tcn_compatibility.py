"""
Test TCN model compatibility and functionality.
"""

import unittest
import torch
import numpy as np
import logging
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.tcn import FluidTCN, TCNConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTCNCompatibility(unittest.TestCase):
    """Test TCN model compatibility with transformers training interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.sequence_length = 3
        self.input_dim = 6712
        self.output_dim = 6712
        
        # Create test config
        self.config = TCNConfig(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            sequence_length=self.sequence_length,
            num_channels=[32, 64, 32],
            kernel_size=3,
            dropout=0.1,
            use_residual=True,
            use_norm=False,
            activation="relu",
            boundary_hidden_dim=32,
            equipment_hidden_dim=64,
            projection_hidden_dim=64
        )
        
        self.model = FluidTCN(self.config)
        self.model.eval()  # Set to evaluation mode for consistent results
        
    def test_model_initialization(self):
        """Test that TCN model initializes properly."""
        logger.info("Testing TCN model initialization...")
        
        # Check model was created
        self.assertIsInstance(self.model, FluidTCN)
        
        # Check model has expected attributes
        self.assertTrue(hasattr(self.model, 'boundary_projection'))
        self.assertTrue(hasattr(self.model, 'equipment_projection'))
        self.assertTrue(hasattr(self.model, 'boundary_tcn'))
        self.assertTrue(hasattr(self.model, 'equipment_tcn'))
        self.assertTrue(hasattr(self.model, 'feature_fusion'))
        self.assertTrue(hasattr(self.model, 'output_projection'))
        
        logger.info("Model initialization test passed")
        
    def test_forward_pass_dict_input(self):
        """Test forward pass with dictionary input (original batch format)."""
        logger.info("Testing forward pass with dictionary input...")
        
        # Create test batch in original format
        batch = {
            'input': torch.randn(self.batch_size, self.sequence_length, self.input_dim),
            'target': torch.randn(self.batch_size, self.sequence_length, self.output_dim),
            'prediction_mask': torch.ones(self.batch_size, self.output_dim)
        }
        
        # Forward pass without labels (inference mode)
        with torch.no_grad():
            output = self.model(batch)
        
        # Check output format
        self.assertIsInstance(output, dict)
        self.assertIn('logits', output)
        self.assertEqual(output['logits'].shape, (self.batch_size, self.sequence_length, self.output_dim))
        
        # Forward pass with labels (training mode)
        with torch.no_grad():
            output_with_loss = self.model(batch, labels=batch['target'])
        
        # Check output format with loss
        self.assertIsInstance(output_with_loss, dict)
        self.assertIn('loss', output_with_loss)
        self.assertIn('logits', output_with_loss)
        self.assertTrue(torch.is_tensor(output_with_loss['loss']))
        self.assertEqual(output_with_loss['logits'].shape, (self.batch_size, self.sequence_length, self.output_dim))
        
        logger.info("Dictionary input test passed")
        
    def test_forward_pass_tensor_input(self):
        """Test forward pass with tensor input (transformers format)."""
        logger.info("Testing forward pass with tensor input...")
        
        # Create test input as tensor
        input_tensor = torch.randn(self.batch_size, self.sequence_length, self.input_dim)
        target_tensor = torch.randn(self.batch_size, self.sequence_length, self.output_dim)
        prediction_mask = torch.ones(self.batch_size, self.output_dim)
        
        # Forward pass without labels
        with torch.no_grad():
            output = self.model(input_ids=input_tensor)
        
        # Check output format
        self.assertIsInstance(output, dict)
        self.assertIn('logits', output)
        self.assertEqual(output['logits'].shape, (self.batch_size, self.sequence_length, self.output_dim))
        
        # Forward pass with labels
        with torch.no_grad():
            output_with_loss = self.model(input_ids=input_tensor, labels=target_tensor, 
                                        prediction_mask=prediction_mask)
        
        # Check output format with loss
        self.assertIsInstance(output_with_loss, dict)
        self.assertIn('loss', output_with_loss)
        self.assertIn('logits', output_with_loss)
        self.assertTrue(torch.is_tensor(output_with_loss['loss']))
        
        logger.info("Tensor input test passed")
        
    def test_gradient_computation(self):
        """Test that gradients can be computed properly."""
        logger.info("Testing gradient computation...")
        
        # Enable training mode
        self.model.train()
        
        # Create test data
        input_tensor = torch.randn(self.batch_size, self.sequence_length, self.input_dim, requires_grad=True)
        target_tensor = torch.randn(self.batch_size, self.sequence_length, self.output_dim)
        prediction_mask = torch.ones(self.batch_size, self.output_dim)
        
        # Forward pass
        output = self.model(input_ids=input_tensor, labels=target_tensor, prediction_mask=prediction_mask)
        loss = output['loss']
        
        # Backward pass
        loss.backward()
        
        # Check that gradients were computed
        self.assertIsNotNone(input_tensor.grad)
        
        # Check that model parameters have gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} should have gradient")
        
        logger.info("Gradient computation test passed")
        
    def test_model_info(self):
        """Test model info generation."""
        logger.info("Testing model info generation...")
        
        info = self.model.get_model_info()
        
        # Check that info is a dictionary
        self.assertIsInstance(info, dict)
        
        # Check for expected keys
        expected_keys = [
            'model_name', 'input_dim', 'output_dim', 'total_parameters',
            'num_channels', 'kernel_size', 'boundary_hidden_dim', 
            'equipment_hidden_dim', 'projection_hidden_dim'
        ]
        
        for key in expected_keys:
            self.assertIn(key, info, f"Info should contain {key}")
        
        logger.info("Model info test passed")
        
    def test_different_input_sizes(self):
        """Test model with different batch sizes and sequence lengths."""
        logger.info("Testing different input sizes...")
        
        test_cases = [
            (1, 3),  # single sample
            (4, 3),  # larger batch
            (2, 5),  # longer sequence
        ]
        
        for batch_size, seq_len in test_cases:
            input_tensor = torch.randn(batch_size, seq_len, self.input_dim)
            
            with torch.no_grad():
                output = self.model(input_ids=input_tensor)
            
            # Check output shape
            expected_shape = (batch_size, seq_len, self.output_dim)
            self.assertEqual(output['logits'].shape, expected_shape, 
                           f"Failed for batch_size={batch_size}, seq_len={seq_len}")
        
        logger.info("Different input sizes test passed")
        
    def test_config_serialization(self):
        """Test config serialization to/from dict."""
        logger.info("Testing config serialization...")
        
        # Convert to dict
        config_dict = self.config.to_dict()
        self.assertIsInstance(config_dict, dict)
        
        # Check key fields
        self.assertEqual(config_dict['input_dim'], self.input_dim)
        self.assertEqual(config_dict['output_dim'], self.output_dim)
        self.assertEqual(config_dict['num_channels'], [32, 64, 32])
        
        # Create new config from dict
        new_config = TCNConfig.from_dict(config_dict)
        
        # Check that configs are equivalent
        self.assertEqual(self.config.input_dim, new_config.input_dim)
        self.assertEqual(self.config.output_dim, new_config.output_dim)
        self.assertEqual(self.config.num_channels, new_config.num_channels)
        
        logger.info("Config serialization test passed")
        
    def test_boundary_equipment_separation(self):
        """Test that boundary and equipment features are processed correctly."""
        logger.info("Testing boundary/equipment feature separation...")
        
        # Create input where first 538 dims are different from rest
        input_tensor = torch.zeros(self.batch_size, self.sequence_length, self.input_dim)
        
        # Set boundary features to 1
        input_tensor[:, :, :538] = 1.0
        # Set equipment features to -1  
        input_tensor[:, :, 538:] = -1.0
        
        with torch.no_grad():
            output = self.model(input_ids=input_tensor)
        
        # Output should be different from zero (model should process the features)
        self.assertNotEqual(torch.sum(output['logits']).item(), 0.0)
        
        logger.info("Boundary/equipment separation test passed")


def run_tests():
    """Run all TCN compatibility tests."""
    print("=" * 60)
    print("RUNNING TCN MODEL COMPATIBILITY TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTCNCompatibility)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("ALL TCN TESTS PASSED!")
        print(f"Ran {result.testsRun} tests successfully")
    else:
        print("SOME TCN TESTS FAILED!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)