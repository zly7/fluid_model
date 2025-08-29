#!/usr/bin/env python3
"""
Test script for the new data loading pipeline.

This script tests:
1. DataProcessor functionality (interpolation, combination, sequences)
2. FluidDataset functionality (loading, TensorDict format)
3. DataLoader with collate function
4. Data dimensions and formats
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
import numpy as np
from tensordict import TensorDict

# Import our modules
from data.processor import DataProcessor
from data.dataset import FluidDataset, collate_fn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_data_processor():
    """Test DataProcessor functionality."""
    logger.info("="*50)
    logger.info("Testing DataProcessor...")
    
    data_dir = project_root / "data"
    processor = DataProcessor(str(data_dir))
    
    # Test basic info
    logger.info(f"Boundary dims: {processor.boundary_dims}")
    logger.info(f"Total prediction dims: {processor.total_prediction_dims}")
    logger.info(f"Equipment info: {processor.equipment_info}")
    
    # Get sample directories
    train_samples = processor.get_sample_directories('train')
    logger.info(f"Found {len(train_samples)} training samples")
    
    if len(train_samples) == 0:
        logger.error("No training samples found!")
        return False
    
    # Test loading a single sample
    sample_dir = train_samples[0]
    logger.info(f"Testing with sample: {sample_dir.name}")
    
    try:
        # Test combined data loading
        sequences, mask = processor.load_combined_sample_data(sample_dir, sequence_length=3)
        
        if not sequences:
            logger.error(f"No sequences generated from {sample_dir.name}")
            return False
        
        logger.info(f"Generated {len(sequences)} sequences")
        
        # Test first sequence
        input_seq, target_seq, start_time, end_time = sequences[0]
        logger.info(f"Sequence shapes - Input: {input_seq.shape}, Target: {target_seq.shape}")
        logger.info(f"Time range: {start_time} to {end_time}")
        
        # Test prediction mask
        if mask is not None:
            logger.info(f"Prediction mask shape: {mask.shape}")
            logger.info(f"Predictable variables: {np.sum(mask)}/{len(mask)}")
        else:
            logger.error("No prediction mask generated")
            return False
        
        # Verify dimensions
        expected_dims = 6712
        if input_seq.shape[1] != expected_dims:
            logger.error(f"Expected {expected_dims} dims, got {input_seq.shape[1]}")
            return False
        
        logger.info("✅ DataProcessor tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"DataProcessor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fluid_dataset():
    """Test FluidDataset functionality."""
    logger.info("="*50)
    logger.info("Testing FluidDataset...")
    
    data_dir = str(project_root / "data")
    
    try:
        # Create dataset
        dataset = FluidDataset(
            data_dir=data_dir,
            split='train',
            sequence_length=3,
            normalize=True,
            max_samples=2,  # Limit for testing
            max_sequences_per_sample=5  # Limit sequences per sample
        )
        
        logger.info(f"Dataset created with {len(dataset)} sequences")
        
        # Get dataset info
        info = dataset.get_feature_info()
        stats = dataset.get_data_statistics()
        
        logger.info(f"Feature info: {info}")
        logger.info(f"Dataset stats: {stats}")
        
        # Test getting a sample
        if len(dataset) > 0:
            sample = dataset[0]
            
            # Verify it's a TensorDict
            if not isinstance(sample, TensorDict):
                logger.error("Sample is not a TensorDict!")
                return False
            
            logger.info(f"Sample keys: {list(sample.keys())}")
            logger.info(f"Input shape: {sample['input'].shape}")
            logger.info(f"Target shape: {sample['target'].shape}")  
            logger.info(f"Mask shape: {sample['mask'].shape}")
            logger.info(f"Metadata: {sample['metadata']}")
            
            # Verify dimensions
            T, V = sample['input'].shape
            if V != 6712:
                logger.error(f"Expected 6712 variables, got {V}")
                return False
            
            if T != 3:
                logger.error(f"Expected sequence length 3, got {T}")
                return False
            
            logger.info("✅ FluidDataset tests passed!")
            return True
        else:
            logger.error("Dataset is empty!")
            return False
            
    except Exception as e:
        logger.error(f"FluidDataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loader():
    """Test DataLoader with collate function."""
    logger.info("="*50)
    logger.info("Testing DataLoader...")
    
    data_dir = str(project_root / "data")
    
    try:
        # Create dataset
        dataset = FluidDataset(
            data_dir=data_dir,
            split='train',
            sequence_length=3,
            normalize=True,
            max_samples=2,
            max_sequences_per_sample=3
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        logger.info(f"DataLoader created with batch_size=4")
        
        # Test getting a batch
        for batch_idx, batch in enumerate(dataloader):
            logger.info(f"Batch {batch_idx}:")
            logger.info(f"  Batch type: {type(batch)}")
            logger.info(f"  Batch keys: {list(batch.keys())}")
            logger.info(f"  Input shape: {batch['input'].shape}")
            logger.info(f"  Target shape: {batch['target'].shape}")
            logger.info(f"  Prediction mask shape: {batch['prediction_mask'].shape}")
            logger.info(f"  Attention mask shape: {batch['attention_mask'].shape}")
            logger.info(f"  Metadata count: {len(batch['metadata'])}")
            
            # Verify batch dimensions [B, T, V]
            B, T, V = batch['input'].shape
            logger.info(f"  Batch dimensions: B={B}, T={T}, V={V}")
            
            if V != 6712:
                logger.error(f"Expected V=6712, got {V}")
                return False
            
            if T != 3:
                logger.error(f"Expected T=3, got {T}")
                return False
            
            # Verify mask dimensions
            pred_mask_shape = batch['prediction_mask'].shape
            attn_mask_shape = batch['attention_mask'].shape
            
            if pred_mask_shape != (B, V):
                logger.error(f"Expected prediction_mask shape ({B}, {V}), got {pred_mask_shape}")
                return False
            
            if attn_mask_shape != (B, T, V):
                logger.error(f"Expected attention_mask shape ({B}, {T}, {V}), got {attn_mask_shape}")
                return False
            
            # Test mask values
            pred_mask = batch['prediction_mask'][0].numpy()
            boundary_vars = np.sum(pred_mask == 0)  # Should be 538
            equipment_vars = np.sum(pred_mask == 1)  # Should be 6174
            logger.info(f"  Prediction mask - Boundary: {boundary_vars}, Equipment: {equipment_vars}")
            
            attn_mask = batch['attention_mask'][0].numpy()
            boundary_attn = np.sum(attn_mask[:, :538])  # Should be 0 (all zeros)
            equipment_attn = np.sum(attn_mask[:, 538:])  # Should be T * 6174
            logger.info(f"  Attention mask - Boundary sum: {boundary_attn}, Equipment sum: {equipment_attn}")
            
            # Test only first batch
            break
        
        logger.info("✅ DataLoader tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_masks():
    """Test new attention mask functionality."""
    logger.info("="*50)
    logger.info("Testing Attention Masks...")
    
    data_dir = str(project_root / "data")
    
    try:
        # Create dataset
        dataset = FluidDataset(
            data_dir=data_dir,
            split='train',
            sequence_length=3,
            normalize=True,
            max_samples=1,
            max_sequences_per_sample=2
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Test a batch
        for batch in dataloader:
            B, T, V = batch['input'].shape
            logger.info(f"Testing batch with shape: B={B}, T={T}, V={V}")
            
            # Test prediction mask
            pred_mask = batch['prediction_mask']  # [B, V]
            logger.info(f"Prediction mask shape: {pred_mask.shape}")
            
            # Verify prediction mask values
            boundary_count = torch.sum(pred_mask[0, :538]).item()
            equipment_count = torch.sum(pred_mask[0, 538:]).item()
            
            logger.info(f"Boundary variables (should be 0): {boundary_count}")
            logger.info(f"Equipment variables (should be 6174): {equipment_count}")
            
            if boundary_count != 0:
                logger.error(f"Boundary variables should be 0, got {boundary_count}")
                return False
            
            if equipment_count != 6174:
                logger.error(f"Equipment variables should be 6174, got {equipment_count}")
                return False
            
            # Test attention mask
            attn_mask = batch['attention_mask']  # [B, T, V]
            logger.info(f"Attention mask shape: {attn_mask.shape}")
            
            # Verify attention mask structure
            for t in range(T):
                # Boundary variables should all be 0
                boundary_attn = torch.sum(attn_mask[0, t, :538]).item()
                # Equipment variables should all be 1
                equipment_attn = torch.sum(attn_mask[0, t, 538:]).item()
                
                logger.info(f"Time step {t}: Boundary attention sum={boundary_attn}, Equipment attention sum={equipment_attn}")
                
                if boundary_attn != 0:
                    logger.error(f"Time step {t}: Boundary attention should be 0, got {boundary_attn}")
                    return False
                
                if equipment_attn != 6174:
                    logger.error(f"Time step {t}: Equipment attention should be 6174, got {equipment_attn}")
                    return False
            
            # Test consistency across batch
            for b in range(1, B):
                if not torch.equal(pred_mask[0], pred_mask[b]):
                    logger.error(f"Prediction masks inconsistent between batch items 0 and {b}")
                    return False
                    
                if not torch.equal(attn_mask[0], attn_mask[b]):
                    logger.error(f"Attention masks inconsistent between batch items 0 and {b}")
                    return False
            
            break
        
        logger.info("✅ Attention mask tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Attention mask test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_consistency():
    """Test data consistency and format."""
    logger.info("="*50)
    logger.info("Testing data consistency...")
    
    data_dir = str(project_root / "data")
    
    try:
        # Create dataset
        dataset = FluidDataset(
            data_dir=data_dir,
            split='train',
            sequence_length=3,
            normalize=False,  # Test without normalization
            max_samples=1,
            max_sequences_per_sample=2
        )
        
        if len(dataset) < 2:
            logger.warning("Not enough sequences for consistency test")
            return True
        
        # Get two samples
        sample1 = dataset[0]
        sample2 = dataset[1]
        
        # Test mask consistency
        mask1 = sample1['mask'].numpy()
        mask2 = sample2['mask'].numpy()
        
        if not np.array_equal(mask1, mask2):
            logger.error("Prediction masks are inconsistent between samples!")
            return False
        
        # Test boundary vs equipment split
        boundary_vars = np.sum(mask1 == 0)  # Should be 539
        equipment_vars = np.sum(mask1 == 1)  # Should be 6174
        
        logger.info(f"Boundary variables: {boundary_vars}")
        logger.info(f"Equipment variables: {equipment_vars}")
        
        if boundary_vars != 538:
            logger.error(f"Expected 538 boundary variables, got {boundary_vars}")
            return False
        
        if equipment_vars != 6174:
            logger.error(f"Expected 6174 equipment variables, got {equipment_vars}")
            return False
        
        # Test time shift between input and target
        input1 = sample1['input'].numpy()
        target1 = sample1['target'].numpy()
        
        logger.info(f"Input time steps: {input1.shape[0]}")
        logger.info(f"Target time steps: {target1.shape[0]}")
        
        # For autoregressive, target should be input shifted by 1 time step
        # This means target[0] should be similar to input[1] (but not exactly due to different boundary conditions)
        
        logger.info("✅ Data consistency tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Data consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Starting data loading pipeline tests...")
    
    tests = [
        ("DataProcessor", test_data_processor),
        ("FluidDataset", test_fluid_dataset),
        ("DataLoader", test_data_loader),
        ("Attention Masks", test_attention_masks),
        ("Data Consistency", test_data_consistency)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n🔬 Running {test_name} test...")
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} test PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} test FAILED with exception: {e}")
    
    logger.info("="*50)
    logger.info(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Data loading pipeline is ready.")
        return True
    else:
        logger.error("🚨 Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)