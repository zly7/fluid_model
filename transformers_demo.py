"""
Demo of using FluidDecoder with HuggingFace Transformers Trainer.
"""

import torch
from torch.utils.data import Dataset
from models.decoder import FluidDecoder
from models.config import DecoderConfig

# 尝试导入transformers，如果不存在则提示安装
try:
    from transformers import Trainer, TrainingArguments
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers library not available. Install with: pip install transformers")
    TRANSFORMERS_AVAILABLE = False


class SimpleFluidDataset(Dataset):
    """Simple dataset for demo purposes."""
    
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.input_dim = 6712
        self.time_steps = 3
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random data for demo
        input_data = torch.randn(self.time_steps, self.input_dim)
        labels = torch.randn(self.time_steps, self.input_dim)
        
        return {
            'input_ids': input_data,
            'labels': labels
        }


def demo_transformers_trainer():
    """Demo using Transformers Trainer with FluidDecoder."""
    
    if not TRANSFORMERS_AVAILABLE:
        print("Cannot run Transformers demo - library not installed")
        return
    
    print("Setting up FluidDecoder with Transformers Trainer...")
    
    # Create model configuration
    config = DecoderConfig(
        d_model=256,
        n_heads=8,
        n_layers=4,
        learning_rate=1e-4
    )
    
    # Create model
    model = FluidDecoder(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create datasets
    train_dataset = SimpleFluidDataset(num_samples=50)
    eval_dataset = SimpleFluidDataset(num_samples=10)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./fluid_decoder_output',
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        load_best_model_at_end=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")
    
    # Test inference
    print("\nTesting inference...")
    test_input = torch.randn(1, 3, 6712)
    with torch.no_grad():
        output = model(input_ids=test_input)
        print(f"Inference successful! Output shape: {output['logits'].shape}")


def demo_manual_training():
    """Demo manual training loop (when transformers not available)."""
    
    print("Running manual training demo...")
    
    # Create model
    config = DecoderConfig(
        d_model=128,
        n_heads=4,
        n_layers=2
    )
    model = FluidDecoder(config)
    
    # Create dummy data
    batch_size = 4
    time_steps = 3
    num_variables = 6712
    
    input_data = torch.randn(batch_size, time_steps, num_variables)
    labels = torch.randn(batch_size, time_steps, num_variables)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("Training for 5 steps...")
    model.train()
    
    for step in range(5):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(input_ids=input_data, labels=labels)
        loss = output['loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Step {step + 1}, Loss: {loss.item():.6f}")
    
    print("Manual training demo completed!")


if __name__ == "__main__":
    print("=" * 60)
    print("FLUID DECODER TRANSFORMERS DEMO")
    print("=" * 60)
    
    try:
        # Always run manual demo since we don't have all transformers dependencies
        demo_manual_training()
            
    except Exception as e:
        print(f"Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()