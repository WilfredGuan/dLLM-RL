#!/usr/bin/env python3
"""Diagnose model loading and configuration"""

import torch
from transformers import AutoTokenizer
import sys
import os

checkpoint_path = "/geminicephfs/game-center-recmd/wilfredguan/dLLM-RL/sft_llada_recursive/ckpt/optimized_recursive"

print("="*80)
print("DIAGNOSTIC SCRIPT FOR MODEL LOADING")
print("="*80)

# Load config
sys.path.insert(0, "/geminicephfs/game-center-recmd/wilfredguan/dLLM-RL")
from sample.llada.configuration_llada import LLaDAConfig
from sample.llada.modeling_recursive_llada import LLaDAModelLM as LLaDAModelLMRecursive

print("\n1. Loading model config...")
model_config = LLaDAConfig.from_pretrained(checkpoint_path)
print(f"   - use_cache: {model_config.use_cache}")
print(f"   - use_latent_recursive: {getattr(model_config, 'use_latent_recursive', 'NOT SET')}")
print(f"   - max_latent_recursive_steps: {getattr(model_config, 'max_latent_recursive_steps', 'NOT SET')}")

print("\n2. Setting recursive config...")
model_config.use_cache = True
model_config.use_latent_recursive = True
model_config.max_latent_recursive_steps = 2

print(f"   - use_latent_recursive: {model_config.use_latent_recursive}")
print(f"   - max_latent_recursive_steps: {model_config.max_latent_recursive_steps}")

print("\n3. Loading model...")
model = LLaDAModelLMRecursive.from_pretrained(
    checkpoint_path,
    config=model_config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

print("\n4. Checking latent_step_embedding...")
if hasattr(model.model, 'latent_step_embedding'):
    emb = model.model.latent_step_embedding
    print(f"   ✓ latent_step_embedding exists")
    print(f"   - Shape: {emb.weight.shape}")
    print(f"   - Dtype: {emb.weight.dtype}")
    print(f"   - Device: {emb.weight.device}")
    print(f"   - Mean: {emb.weight.float().mean():.6f}")
    print(f"   - Std: {emb.weight.float().std():.6f}")
    print(f"   - Has NaN: {torch.isnan(emb.weight).any()}")
    print(f"   - Has Inf: {torch.isinf(emb.weight).any()}")
else:
    print(f"   ✗ latent_step_embedding NOT FOUND!")

print("\n5. Checking freeze configuration...")
metadata_path = os.path.join(checkpoint_path, "metadata.json")
if os.path.exists(metadata_path):
    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    freeze_config = metadata.get('freeze_config', {})
    print(f"   ✓ metadata.json exists")
    print(f"   - freeze_first_half_layers: {freeze_config.get('freeze_first_half_layers')}")
    print(f"   - trainable_layer_start_idx: {freeze_config.get('trainable_layer_start_idx')}")
else:
    print(f"   ✗ metadata.json NOT FOUND")

print("\n6. Setting freeze config on model...")
model.model.freeze_first_half_layers = True
model.model.trainable_layer_start_idx = 16
print(f"   - model.model.freeze_first_half_layers: {model.model.freeze_first_half_layers}")
print(f"   - model.model.trainable_layer_start_idx: {model.model.trainable_layer_start_idx}")

print("\n7. Testing forward_latent_only...")
try:
    # Create dummy input
    hidden_states = torch.randn(1, 10, 4096, dtype=torch.bfloat16)
    
    # Test step 0
    output = model.model.forward_latent_only(
        hidden_states=hidden_states,
        latent_step=0,
        attention_bias=None
    )
    print(f"   ✓ forward_latent_only(step=0) succeeded")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Has NaN: {torch.isnan(output).any()}")
    
    # Test step 1
    output = model.model.forward_latent_only(
        hidden_states=hidden_states,
        latent_step=1,
        attention_bias=None
    )
    print(f"   ✓ forward_latent_only(step=1) succeeded")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Has NaN: {torch.isnan(output).any()}")
    
except Exception as e:
    print(f"   ✗ forward_latent_only FAILED: {e}")

print("\n8. Testing simple generation...")
try:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    test_text = "What is 2+2?"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    # Simple forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    next_token_id = logits[0, -1].argmax().item()
    next_token = tokenizer.decode([next_token_id])
    
    print(f"   ✓ Simple forward pass succeeded")
    print(f"   - Input: '{test_text}'")
    print(f"   - Next token: '{next_token}'")
    print(f"   - Logits shape: {logits.shape}")
    print(f"   - Has NaN in logits: {torch.isnan(logits).any()}")
    
except Exception as e:
    print(f"   ✗ Simple generation FAILED: {e}")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
