"""Final test to confirm transformers compatibility."""

import torch
from models.decoder import FluidDecoder
from models.config import DecoderConfig

print("Creating model...")
config = DecoderConfig(d_model=64, n_heads=2, n_layers=1)
model = FluidDecoder(config)

print("Testing transformers format...")
input_tensor = torch.randn(2, 3, 6712)
labels = torch.randn(2, 3, 6712)

# Test transformers format
output = model(input_ids=input_tensor, labels=labels)
print(f"Output type: {type(output)}")
print(f"Keys: {list(output.keys())}")
print(f"Loss: {output['loss'].item():.4f}")
print(f"Logits shape: {output['logits'].shape}")

# Test without labels
output_no_labels = model(input_ids=input_tensor)
print(f"Without labels - Keys: {list(output_no_labels.keys())}")

print("SUCCESS: Model is transformers compatible!")