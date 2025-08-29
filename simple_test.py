"""Simple test to check model import."""

import torch
from models.decoder import FluidDecoder
from models.config import DecoderConfig

print("Creating simple config...")
config = DecoderConfig(
    d_model=64,
    n_heads=2,
    n_layers=1
)

print("Creating model...")
model = FluidDecoder(config)

print("Creating test data...")
batch_size = 1
time_steps = 3
num_variables = 6712

input_tensor = torch.randn(batch_size, time_steps, num_variables)
labels = torch.randn(batch_size, time_steps, num_variables)

print("Testing forward pass...")
output = model(input_ids=input_tensor, labels=labels)

print(f"Success! Output keys: {output.keys()}")
print(f"Loss: {output['loss'].item():.6f}")
print(f"Logits shape: {output['logits'].shape}")
print("Transformers compatibility test passed!")