#!/usr/bin/env python3
"""
Validation script for Fluid Dynamics Transformer installation.

Tests all components and reports any missing dependencies.
"""

import sys
import importlib
import torch
import numpy as np
from pathlib import Path

def test_import(module_name, description, required=True):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"[OK] {description}")
        return True
    except ImportError as e:
        if required:
            print(f"[FAIL] {description} - FAILED: {e}")
            return False
        else:
            print(f"[SKIP] {description} - OPTIONAL: {e}")
            return True
    except Exception as e:
        print(f"[ERROR] {description} - ERROR: {e}")
        return False

def test_core_functionality():
    """Test core functionality without optional dependencies."""
    print("\n=== Testing Core Functionality ===")
    success = True
    
    try:
        # Test data processing
        print("Testing data components...")
        from data.processor import DataProcessor
        from data.dataset import FluidDataset
        from data.loader import create_data_loaders, AutoregressiveCollator
        print("[OK] Data components functional")
        
        # Test model components
        print("Testing model components...")
        from models.positional import PositionalEncoding, TemporalPositionalEncoding
        from models.embeddings import InputEmbedding, MultiTargetHead
        from models.transformer import DecoderOnlyTransformer
        print("[OK] Model components functional")
        
        # Test loss functions
        print("Testing loss functions...")
        from training.loss import FluidDynamicsLoss, WeightedMSELoss, MultiTargetLoss
        print("[OK] Loss functions functional")
        
        # Test metrics
        print("Testing evaluation metrics...")
        from utils.metrics import (
            calculate_mse, calculate_mae, calculate_r2_score,
            FluidDynamicsEvaluator, EquipmentMetrics
        )
        print("[OK] Evaluation metrics functional")
        
        # Test basic model creation
        print("Testing model instantiation...")
        model = DecoderOnlyTransformer(
            input_dim=539,
            d_model=128,
            nhead=4,
            num_decoder_layers=2,
            total_output_dim=100
        )
        print("[OK] Model instantiation successful")
        
        # Test forward pass with dummy data
        print("Testing model forward pass...")
        dummy_input = torch.randn(2, 10, 539)  # batch_size=2, seq_len=10, input_dim=539
        with torch.no_grad():
            output = model(dummy_input)
        expected_shape = (2, 10, 100)  # batch_size=2, seq_len=10, output_dim=100
        if output.shape == expected_shape:
            print(f"[OK] Model forward pass successful - output shape: {output.shape}")
        else:
            print(f"[FAIL] Model forward pass shape mismatch - expected: {expected_shape}, got: {output.shape}")
            success = False
        
        # Test loss computation
        print("Testing loss computation...")
        equipment_dims = {'B': 30, 'C': 20, 'P': 50}
        loss_fn = FluidDynamicsLoss(equipment_dims)
        dummy_targets = torch.randn(2, 10, 100)
        loss = loss_fn(output, dummy_targets)
        if torch.is_tensor(loss) and loss.dim() == 0:
            print(f"[OK] Loss computation successful - loss value: {loss.item():.6f}")
        else:
            print(f"[FAIL] Loss computation failed - unexpected output: {loss}")
            success = False
        
        # Test metrics calculation
        print("Testing metrics calculation...")
        evaluator = FluidDynamicsEvaluator(equipment_dims)
        results = evaluator.evaluate_predictions(
            output.numpy(), 
            dummy_targets.numpy(), 
            return_detailed=False
        )
        if 'overall' in results and 'mse' in results['overall']:
            print(f"[OK] Metrics calculation successful - MSE: {results['overall']['mse']:.6f}")
        else:
            print(f"[FAIL] Metrics calculation failed")
            success = False
            
    except Exception as e:
        print(f"[FAIL] Core functionality test failed: {e}")
        success = False
    
    return success

def test_optional_components():
    """Test optional components."""
    print("\n=== Testing Optional Components ===")
    
    # Test TensorBoard utilities
    try:
        from utils.tensorboard_utils import TensorBoardLogger
        print("[OK] TensorBoard utilities available")
    except ImportError:
        print("[SKIP] TensorBoard utilities not available (tensorboard package not installed)")
    
    # Test Streamlit visualization
    try:
        import streamlit
        print("[OK] Streamlit available for data visualization")
    except ImportError:
        print("[SKIP] Streamlit not available (streamlit package not installed)")
    
    # Test training with TensorBoard
    try:
        from training.trainer import FluidDynamicsTrainer
        print("[OK] Full training infrastructure available")
    except ImportError as e:
        print(f"[SKIP] Full training infrastructure not available: {e}")

def test_system_requirements():
    """Test system requirements."""
    print("=== System Requirements ===")
    
    # Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"[OK] Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"[FAIL] Python version too old: {python_version.major}.{python_version.minor}.{python_version.micro} (requires 3.8+)")
        return False
    
    # PyTorch
    try:
        print(f"[OK] PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("[SKIP] CUDA not available (CPU-only)")
    except:
        print("[FAIL] PyTorch not properly installed")
        return False
    
    # NumPy
    try:
        print(f"[OK] NumPy version: {np.__version__}")
    except:
        print("[FAIL] NumPy not available")
        return False
    
    return True

def test_file_structure():
    """Test project file structure."""
    print("\n=== File Structure Validation ===")
    
    required_files = [
        "main.py",
        "requirements.txt",
        "data/__init__.py",
        "data/processor.py",
        "data/dataset.py",
        "data/loader.py",
        "models/__init__.py",
        "models/transformer.py",
        "models/positional.py",
        "models/embeddings.py",
        "training/__init__.py",
        "training/loss.py",
        "training/trainer.py",
        "utils/__init__.py",
        "utils/metrics.py",
        "utils/tensorboard_utils.py",
        "visualization/streamlit_app.py",
        "visualization/plots.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"[OK] {file_path}")
        else:
            print(f"[MISSING] {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n[FAIL] Missing files: {missing_files}")
        return False
    else:
        print("\n[OK] All required files present")
        return True

def main():
    """Run all validation tests."""
    print("Fluid Dynamics Transformer - Installation Validation")
    print("=" * 55)
    
    all_tests_passed = True
    
    # Test system requirements
    if not test_system_requirements():
        all_tests_passed = False
    
    # Test file structure
    if not test_file_structure():
        all_tests_passed = False
    
    # Test required dependencies
    print("\n=== Required Dependencies ===")
    required_modules = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
    ]
    
    for module, description in required_modules:
        if not test_import(module, description, required=True):
            all_tests_passed = False
    
    # Test optional dependencies
    print("\n=== Optional Dependencies ===")
    optional_modules = [
        ("tensorboard", "TensorBoard (for training visualization)"),
        ("streamlit", "Streamlit (for data exploration)"),
        ("seaborn", "Seaborn (for enhanced plotting)"),
        ("plotly", "Plotly (for interactive charts)")
    ]
    
    for module, description in optional_modules:
        test_import(module, description, required=False)
    
    # Test core functionality
    if not test_core_functionality():
        all_tests_passed = False
    
    # Test optional components
    test_optional_components()
    
    # Final report
    print("\n" + "=" * 55)
    if all_tests_passed:
        print("[SUCCESS] VALIDATION SUCCESSFUL")
        print("The Fluid Dynamics Transformer is ready to use!")
        print("\nTo install optional dependencies:")
        print("  pip install tensorboard streamlit seaborn plotly")
    else:
        print("[FAIL] VALIDATION FAILED")
        print("Please fix the issues above before using the system.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())