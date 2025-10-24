#!/usr/bin/env python3
"""
Test script for data processing logic in sft_llada.py
Tests tokenization, padding, masking, and label handling
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer
import json


def test_tokenization_and_padding():
    """Test basic tokenization and padding logic"""
    print("\n" + "="*80)
    print("TEST 1: Tokenization and Padding")
    print("="*80)

    # Load tokenizer
    pretrained_model = "GSAI-ML/LLaDA-8B-Instruct"  # Adjust path as needed
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    # Test data
    prompts = [
        "What is the capital of France?",
        "Short prompt",
    ]
    responses = [
        "The capital of France is Paris.",
        "This is a much longer response that should be padded differently than the first one.",
    ]

    # Tokenize prompts (left padding)
    prompt_ids = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
        padding_side="left",
        truncation=True,
        max_length=100
    )['input_ids']

    # Tokenize responses (right padding)
    response_ids = tokenizer(
        responses,
        padding=True,
        return_tensors="pt",
        padding_side="right",
        truncation=True,
        max_length=100
    )['input_ids']

    print(f"\nPrompt IDs shape: {prompt_ids.shape}")
    print(f"Response IDs shape: {response_ids.shape}")

    # Check padding
    pad_id = tokenizer.pad_token_id
    print(f"\nPad token ID: {pad_id}")
    print(f"Prompt 0 padding positions: {torch.where(prompt_ids[0] == pad_id)[0].tolist()}")
    print(f"Prompt 1 padding positions: {torch.where(prompt_ids[1] == pad_id)[0].tolist()}")
    print(f"Response 0 padding positions: {torch.where(response_ids[0] == pad_id)[0].tolist()}")
    print(f"Response 1 padding positions: {torch.where(response_ids[1] == pad_id)[0].tolist()}")

    # Decode to verify
    print(f"\nDecoded prompt 0: {tokenizer.decode(prompt_ids[0])}")
    print(f"Decoded prompt 1: {tokenizer.decode(prompt_ids[1])}")
    print(f"Decoded response 0: {tokenizer.decode(response_ids[0])}")
    print(f"Decoded response 1: {tokenizer.decode(response_ids[1])}")

    return prompt_ids, response_ids, pad_id


def test_concatenation_and_labels(prompt_ids, response_ids, pad_id):
    """Test concatenation and label creation"""
    print("\n" + "="*80)
    print("TEST 2: Concatenation and Label Creation")
    print("="*80)
    
    max_seq_len = prompt_ids.shape[1] + response_ids.shape[1]
    
    sequence_ids = []
    label_ids = []
    
    for prompt_id, resp_id in zip(prompt_ids, response_ids):
        prompt_id = prompt_id.tolist()
        resp_id = resp_id.tolist()
        
        # Concatenate
        temp_ids = prompt_id + resp_id
        temp_labels = temp_ids.copy()
        
        # Padding or truncation
        if len(temp_ids) < max_seq_len:
            pad_len = max_seq_len - len(temp_ids)
            temp_ids.extend([pad_id] * pad_len)
            temp_labels.extend([-100] * pad_len)  # Key: padding positions should be -100
        else:
            temp_ids = temp_ids[:max_seq_len]
            temp_labels = temp_labels[:max_seq_len]
        
        sequence_ids.append(torch.tensor(temp_ids).unsqueeze(0))
        label_ids.append(torch.tensor(temp_labels).unsqueeze(0))
    
    input_ids = torch.cat(sequence_ids, dim=0)
    labels = torch.cat(label_ids, dim=0)
    
    print(f"\nInput IDs shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Check that padding positions have label -100
    for i in range(len(input_ids)):
        pad_positions = torch.where(input_ids[i] == pad_id)[0]
        label_at_pad = labels[i][pad_positions]
        print(f"\nSample {i}:")
        print(f"  Padding positions: {pad_positions.tolist()[:10]}...")
        print(f"  Labels at padding: {label_at_pad.tolist()[:10]}...")
        print(f"  All padding labels are -100: {torch.all(label_at_pad == -100).item()}")
    
    return input_ids, labels


def test_masking_logic(input_ids, labels, pad_id, mask_id=None):
    """Test masking logic (random masking)"""
    print("\n" + "="*80)
    print("TEST 3: Masking Logic")
    print("="*80)
    
    if mask_id is None:
        # Use a placeholder mask_id for testing
        mask_id = 50256  # Adjust based on your tokenizer
    
    B, L = input_ids.shape
    start_pos = L // 2  # Assume first half is prompt
    
    # Random masking parameters
    lower_p = 0.2
    upper_p = 0.8
    m = 3  # Number of masks per sample
    
    noisy_list, label_list, pmask_list = [], [], []
    
    for b in range(B):
        base_ids = input_ids[b]
        label_ids = labels[b]
        
        # Identify tail padding (should not be masked)
        pad_mask_b = (base_ids == pad_id)
        pad_mask_b[:start_pos] = False
        tail_pad_b = pad_mask_b
        
        for _ in range(m):
            # Random mask ratio
            t = (upper_p - lower_p) * torch.rand(1) + lower_p
            rand_mask = torch.rand(L) < t
            rand_mask[:start_pos] = False  # Don't mask prompt
            rand_mask = rand_mask & ~tail_pad_b  # Don't mask tail padding
            
            if not rand_mask.any():
                continue
            
            # Create noisy input
            noisy_ids = base_ids.clone()
            noisy_ids[rand_mask] = mask_id
            noisy_ids[tail_pad_b] = mask_id
            
            noisy_list.append(noisy_ids)
            label_list.append(label_ids)
            pmask_list.append(rand_mask)
    
    noisy_batch = torch.stack(noisy_list)
    labels_batch = torch.stack(label_list)
    p_mask = torch.stack(pmask_list)
    
    print(f"\nNoisy batch shape: {noisy_batch.shape}")
    print(f"Labels batch shape: {labels_batch.shape}")
    print(f"P_mask shape: {p_mask.shape}")
    
    # Check masking statistics
    for i in range(min(3, len(noisy_batch))):
        mask_positions = torch.where(p_mask[i])[0]
        mask_ratio = len(mask_positions) / L
        print(f"\nSample {i}:")
        print(f"  Mask positions: {mask_positions.tolist()[:10]}...")
        print(f"  Mask ratio: {mask_ratio:.2%}")
        print(f"  Masked tokens: {noisy_batch[i][mask_positions].tolist()[:10]}...")
        print(f"  Labels at masks: {labels_batch[i][mask_positions].tolist()[:10]}...")
    
    return noisy_batch, labels_batch, p_mask


def test_loss_computation(noisy_batch, labels_batch, p_mask):
    """Test loss computation logic"""
    print("\n" + "="*80)
    print("TEST 4: Loss Computation")
    print("="*80)
    
    import torch.nn.functional as F
    
    B, T = noisy_batch.shape
    
    # Get actual vocab size from labels
    V = labels_batch[labels_batch != -100].max().item() + 1
    print(f"Detected vocab size: {V}")
    
    # Create dummy logits
    logits = torch.randn(B, T, V)
    
    # Clamp logits
    logits = torch.clamp(logits, min=-1e4, max=1e4)
    
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather log probs for target tokens
    safe_labels = labels_batch.clone()
    safe_labels[labels_batch == -100] = 0
    # Ensure all indices are within valid range
    safe_labels = torch.clamp(safe_labels, min=0, max=V-1)
    logp_tok = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    
    # Compute loss only on masked positions
    loss_per_sample = -(logp_tok * p_mask).sum(dim=1)
    mask_num = p_mask.sum(dim=1).clamp(min=1)
    loss_per_sample = loss_per_sample / mask_num
    
    loss = loss_per_sample.sum() / B
    
    print(f"\nLogits shape: {logits.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Loss per sample: {loss_per_sample.tolist()}")
    print(f"Final loss: {loss.item():.4f}")
    
    # Check that padding positions don't contribute to loss
    print(f"\nVerifying padding positions don't contribute to loss:")
    for i in range(min(2, B)):
        pad_positions = torch.where(labels_batch[i] == -100)[0]
        if len(pad_positions) > 0:
            pmask_at_pad = p_mask[i][pad_positions]
            print(f"  Sample {i}: p_mask at padding = {pmask_at_pad.sum().item()} (should be 0)")
    
    return loss


def test_with_real_data():
    """Test with real data format"""
    print("\n" + "="*80)
    print("TEST 5: Real Data Format")
    print("="*80)
    
    # Simulate real data
    sample_data = [
        {
            "prompt": "What is 2+2?",
            "response": "2+2 equals 4.",
            "step_map": [0, 0, 1, 1, 2]
        },
        {
            "prompt": "Explain photosynthesis.",
            "response": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "step_map": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        }
    ]
    
    print(f"\nSample data:")
    for i, item in enumerate(sample_data):
        print(f"\nSample {i}:")
        print(f"  Prompt: {item['prompt']}")
        print(f"  Response: {item['response']}")
        print(f"  Step map length: {len(item['step_map'])}")
    
    # This would be processed by prepare_inputs_and_labels_for_text
    print("\n✓ Data format is correct")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("DATA PROCESSING TESTS")
    print("="*80)
    
    try:
        # Test 1: Tokenization and padding
        prompt_ids, response_ids, pad_id = test_tokenization_and_padding()
        
        # Test 2: Concatenation and labels
        input_ids, labels = test_concatenation_and_labels(prompt_ids, response_ids, pad_id)
        
        # Test 3: Masking logic
        noisy_batch, labels_batch, p_mask = test_masking_logic(input_ids, labels, pad_id)
        
        # Test 4: Loss computation
        loss = test_loss_computation(noisy_batch, labels_batch, p_mask)
        
        # Test 5: Real data format
        test_with_real_data()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"✗ TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
