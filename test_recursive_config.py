#!/usr/bin/env python3
"""
Test script to verify latent recursive configuration and model setup.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

def test_config():
    """Test loading the recursive config"""
    print("=" * 60)
    print("Testing Configuration Loading")
    print("=" * 60)
    
    config = OmegaConf.load("configs/sft_llada_recursive.yaml")
    
    print(f"\n✓ Config loaded successfully")
    print(f"  - Project: {config.experiment.project}")
    print(f"  - Use latent recursive: {config.training.use_latent_recursive}")
    print(f"  - Latent recursive steps: {config.training.latent_recursive_steps}")
    print(f"  - Block size: {config.training.block_size}")
    print(f"  - Use tensorboard: {config.logging.use_tensorboard}")
    print(f"  - Log dir: {config.logging.log_dir}")
    
    return config

def test_model_config():
    """Test model configuration with latent recursive"""
    print("\n" + "=" * 60)
    print("Testing Model Configuration")
    print("=" * 60)
    
    from models.llada.configuration_llada import ModelConfig
    
    model_config = ModelConfig()
    model_config.use_latent_recursive = True
    model_config.max_latent_recursive_steps = 32
    
    print(f"\n✓ ModelConfig created")
    print(f"  - use_latent_recursive: {model_config.use_latent_recursive}")
    print(f"  - max_latent_recursive_steps: {model_config.max_latent_recursive_steps}")
    
    return model_config

def test_decode_efficiency():
    """Test decode efficiency calculation"""
    print("\n" + "=" * 60)
    print("Testing Decode Efficiency Calculation")
    print("=" * 60)
    
    import math
    
    # Test case from your example
    diffusion_time_steps = 128
    max_gen_length = 128
    block_size = 32
    latent_recursive_steps = 16
    
    # Calculate
    original_efficiency = max_gen_length / diffusion_time_steps
    steps_per_block_original = block_size / original_efficiency
    remaining_steps_per_block = steps_per_block_original - latent_recursive_steps
    unmask_speed = block_size / remaining_steps_per_block
    actual_unmask_steps = math.ceil(block_size / unmask_speed)
    total_steps_per_block = latent_recursive_steps + actual_unmask_steps
    
    print(f"\n✓ Decode efficiency calculation:")
    print(f"  - Original efficiency: {original_efficiency:.2f} tokens/step")
    print(f"  - Steps per block (original): {steps_per_block_original:.2f}")
    print(f"  - Latent recursive steps: {latent_recursive_steps}")
    print(f"  - Remaining steps: {remaining_steps_per_block:.2f}")
    print(f"  - Unmask speed: {unmask_speed:.2f} tokens/step")
    print(f"  - Actual unmask steps: {actual_unmask_steps}")
    print(f"  - Total steps per block: {total_steps_per_block:.2f}")
    print(f"  - Match original: {abs(total_steps_per_block - steps_per_block_original) < 0.01} ✓")

def main():
    print("\n" + "=" * 60)
    print("Latent Recursive Configuration Test")
    print("=" * 60)
    
    try:
        # Test 1: Config loading
        config = test_config()
        
        # Test 2: Model config
        model_config = test_model_config()
        
        # Test 3: Decode efficiency
        test_decode_efficiency()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Update pretrained_model path in configs/sft_llada_recursive.yaml")
        print("2. Update dataset name in configs/sft_llada_recursive.yaml")
        print("3. Run training with:")
        print("   python train/sft_llada.py --config configs/sft_llada_recursive.yaml")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
