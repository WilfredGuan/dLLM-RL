#!/usr/bin/env python3
"""Quick script to check if latent_step_embedding is in the checkpoint"""

import torch
from safetensors import safe_open

checkpoint_path = "/geminicephfs/game-center-recmd/wilfredguan/dLLM-RL/sft_llada_recursive/ckpt/optimized_recursive"

print("Checking checkpoint files for latent_step_embedding...")
print(f"Checkpoint path: {checkpoint_path}")
print()

# Check all safetensors files
for i in range(1, 5):
    file_path = f"{checkpoint_path}/model-{i:05d}-of-00004.safetensors"
    print(f"\nChecking {file_path}...")
    
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            
            # Look for latent_step_embedding
            latent_keys = [k for k in keys if 'latent_step_embedding' in k]
            
            if latent_keys:
                print(f"  ✓ Found latent_step_embedding keys:")
                for key in latent_keys:
                    tensor = f.get_tensor(key)
                    print(f"    - {key}: shape={tensor.shape}, dtype={tensor.dtype}")
                    print(f"      mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}")
                    print(f"      min={tensor.float().min():.6f}, max={tensor.float().max():.6f}")
                    if torch.isnan(tensor).any():
                        print(f"      ⚠️  WARNING: Contains NaN values!")
                    if torch.isinf(tensor).any():
                        print(f"      ⚠️  WARNING: Contains Inf values!")
            else:
                print(f"  - No latent_step_embedding keys found")
                
    except Exception as e:
        print(f"  Error reading file: {e}")

print("\n" + "="*80)
print("Summary:")
print("If latent_step_embedding is NOT found in any file, the checkpoint is missing")
print("the trained recursive parameters, which would cause garbage output!")
print("="*80)
