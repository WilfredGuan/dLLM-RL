import os
import sys
import argparse
from typing import List, Tuple

import torch
from transformers import AutoTokenizer

"""
This script compares the current tokenization/padding behavior in train/sft_llada.py
with a suggested corrected behavior. It prints token sequences (keeping special tokens)
and token ids to help verify padding, mask tokens, and concatenation logic.

Usage:
  python debug_tokenization_test.py \
    --model <pretrained_model_or_path> \
    --max_prompt_len 64 \
    --max_gen_length 64

Key improvements in this version:
1. Separate tokenizer instances for CURRENT vs SUGGESTED to ensure isolation
2. CURRENT: Replicates exact bugs from sft_llada.py (no prompt masking, template tokens split)
3. SUGGESTED: Applies all fixes (register special tokens, mask PAD+prompt in labels)
4. Detailed comparison output with warnings for unmasked positions

All comments in English as requested.
"""


def pretty_print_tokens(tokenizer: AutoTokenizer, ids: List[int], title: str):
    toks = tokenizer.convert_ids_to_tokens(ids)
    print(f"\n== {title} ==")
    print("ids:   ", ids)
    print("tokens:", toks)
    print("decoded:", tokenizer.decode(ids, skip_special_tokens=False))


def current_prepare(prompt: List[str], response: List[str], tokenizer: AutoTokenizer,
                    max_prompt_len: int, max_gen_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """
    Replicate prepare_inputs_and_labels_for_text() key parts from sft_llada.py
    - Prompt tokenization with left padding (attempted via padding_side in call - IGNORED by HF)
    - Response tokenization with right padding (attempted via padding_side in call - IGNORED by HF)
    - Concatenate and pad using pad_id = tokenizer.encode('<|endoftext|>')[0]
    - Labels: copy of input, only tail padding gets -100 (BUG: prompt not masked!)
    - Template tokens NOT registered, so they get split into multiple tokens
    Returns: input_ids_lm, labels_lm, attention_mask, start_pos, pad_id, mask_id
    """
    # BUG 1: Derived the same way as in sft_llada.py - may produce multi-token IDs
    mask_id = tokenizer.encode('<|mdm_mask|>')[0]
    pad_id = tokenizer.encode('<|endoftext|>')[0]

    # BUG 2: padding_side passed as argument is IGNORED by HF tokenizers
    # 1) prompt ids (left padding requested via argument — IGNORED)
    prompt_outputs = tokenizer(
        prompt,
        padding=True,
        return_tensors="pt",
        padding_side="left",  # IGNORED!
        truncation=True,
        max_length=max_prompt_len
    )
    prompt_ids = prompt_outputs["input_ids"]

    # 2) response ids (right padding requested via argument — IGNORED)
    response_outputs = tokenizer(
        response,
        padding=True,
        return_tensors="pt",
        padding_side="right",  # IGNORED!
        truncation=True,
        max_length=max_gen_length
    )
    response_ids = response_outputs["input_ids"]

    # 3) concatenate + pad
    if response_ids.shape[1] < max_gen_length:
        max_seq_len = prompt_ids.shape[1] + response_ids.shape[1]
    else:
        max_seq_len = prompt_ids.shape[1] + max_gen_length

    sequence_ids = []
    label_ids = []
    attention_masks = []
    
    for p_ids, r_ids in zip(prompt_ids, response_ids):
        p = p_ids.tolist()
        r = r_ids.tolist()
        temp_ids = p + r
        temp_labels = temp_ids.copy()  # BUG 3: No masking for prompt positions!
        temp_attn = [1] * len(temp_ids)
        
        if len(temp_ids) < max_seq_len:
            pad_len = max_seq_len - len(temp_ids)
            temp_ids.extend([pad_id] * pad_len)
            temp_labels.extend([-100] * pad_len)  # Only tail padding gets -100
            temp_attn.extend([0] * pad_len)
        else:
            temp_ids = temp_ids[:max_seq_len]
            temp_labels = temp_labels[:max_seq_len]
            temp_attn = temp_attn[:max_seq_len]
        
        sequence_ids.append(torch.tensor(temp_ids).unsqueeze(0))
        label_ids.append(torch.tensor(temp_labels).unsqueeze(0))
        attention_masks.append(torch.tensor(temp_attn).unsqueeze(0))

    input_ids_lm = torch.cat(sequence_ids, dim=0)
    labels_lm = torch.cat(label_ids, dim=0)
    attention_mask = torch.cat(attention_masks, dim=0)
    start_pos = prompt_ids.shape[1]

    return input_ids_lm, labels_lm, attention_mask, start_pos, pad_id, mask_id


def suggested_prepare(prompt: List[str], response: List[str], tokenizer: AutoTokenizer,
                      max_prompt_len: int, max_gen_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """
    Suggested fixes:
    - FIX 1: Ensure pad token consistency (pad_token_id). If missing, set pad_token = eos_token.
    - FIX 2: Register template tokens (<|user|>, <|assistant|>, etc.) as special tokens.
    - FIX 3: Register <|mdm_mask|> as a single special token.
    - FIX 4: Use tokenizer.padding_side attribute instead of passing as argument.
    - FIX 5: Apply proper label masking: mask ALL PAD positions AND ALL prompt positions.
    Returns: input_ids_lm, labels_lm, attention_mask, start_pos, pad_id, mask_id
    """
    # FIX 1: Ensure pad token consistency
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # FIX 2: Register template tokens as special tokens
    template_tokens = ["<|user|>", "<|assistant|>", "<|system|>"]
    added_vocab = tokenizer.get_added_vocab()
    tokens_to_add = [t for t in template_tokens if t not in added_vocab]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    
    # FIX 3: Register and get mask id as a single token
    if "<|mdm_mask|>" not in tokenizer.get_added_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|mdm_mask|>"]})
    mask_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")

    # FIX 4: Properly control padding sides using attribute
    tokenizer.padding_side = "left"
    prompt_outputs = tokenizer(
        prompt,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_len
    )
    prompt_ids = prompt_outputs["input_ids"]
    prompt_attention_mask = prompt_outputs["attention_mask"]

    tokenizer.padding_side = "right"
    response_outputs = tokenizer(
        response,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_gen_length
    )
    response_ids = response_outputs["input_ids"]
    response_attention_mask = response_outputs["attention_mask"]

    # 5) concatenate + pad with proper label masking
    if response_ids.shape[1] < max_gen_length:
        max_seq_len = prompt_ids.shape[1] + response_ids.shape[1]
    else:
        max_seq_len = prompt_ids.shape[1] + max_gen_length

    sequence_ids = []
    label_ids = []
    attention_masks = []
    prompt_lens = []  # Track per-sample prompt length
    
    for p_ids, r_ids, p_attn, r_attn in zip(prompt_ids, response_ids, prompt_attention_mask, response_attention_mask):
        p = p_ids.tolist()
        r = r_ids.tolist()
        p_attn_list = p_attn.tolist()
        r_attn_list = r_attn.tolist()
        
        # Store full prompt length including padding
        prompt_lens.append(len(p))
        
        temp_ids = p + r
        temp_labels = temp_ids.copy()
        temp_attn = p_attn_list + r_attn_list
        
        if len(temp_ids) < max_seq_len:
            pad_len = max_seq_len - len(temp_ids)
            temp_ids.extend([pad_id] * pad_len)
            temp_labels.extend([-100] * pad_len)
            temp_attn.extend([0] * pad_len)
        else:
            temp_ids = temp_ids[:max_seq_len]
            temp_labels = temp_labels[:max_seq_len]
            temp_attn = temp_attn[:max_seq_len]
        
        sequence_ids.append(torch.tensor(temp_ids).unsqueeze(0))
        label_ids.append(torch.tensor(temp_labels).unsqueeze(0))
        attention_masks.append(torch.tensor(temp_attn).unsqueeze(0))

    input_ids_lm = torch.cat(sequence_ids, dim=0)
    labels_lm = torch.cat(label_ids, dim=0)
    attention_mask = torch.cat(attention_masks, dim=0)
    
    # FIX 5a: Mask all PAD positions in labels
    labels_lm[attention_mask == 0] = -100
    
    # FIX 5b: Mask prompt positions in labels (for SFT)
    for i in range(len(prompt_lens)):
        labels_lm[i, :prompt_lens[i]] = -100
    
    start_pos = prompt_ids.shape[1]

    return input_ids_lm, labels_lm, attention_mask, start_pos, pad_id, mask_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Pretrained model or local path",
        default="GSAI-ML/LLaDA-8B-Instruct",
    )
    parser.add_argument("--max_prompt_len", type=int, default=64)
    parser.add_argument("--max_gen_length", type=int, default=64)
    args = parser.parse_args()

    # Load two separate tokenizers to ensure isolation
    print("Loading tokenizers...")
    tok_current = AutoTokenizer.from_pretrained(args.model)
    tok_suggested = AutoTokenizer.from_pretrained(args.model)
    print(f"Loaded 2 separate tokenizer instances from {args.model}\n")

    # Build synthetic test cases: different prompt/response lengths to trigger padding
    prompts = [
        "<|user|> Short question?<|assistant|>",
        "<|user|> This is a much longer prompt that should demonstrate left padding effect when batched.<|assistant|>"
    ]
    responses = [
        "Short answer.",
        "This is a somewhat longer answer that will be right-padded in the batch."
    ]

    print("\n" + "="*80)
    print("CURRENT METHOD (as in sft_llada.py - with bugs)")
    print("="*80)
    print("[Using separate tokenizer instance: tok_current]")
    cur_inputs, cur_labels, cur_attn_mask, cur_start, cur_pad_id, cur_mask_id = current_prepare(
        prompts, responses, tok_current, args.max_prompt_len, args.max_gen_length
    )
    print(f"\npad_id(from eos encode): {cur_pad_id}")
    print(f"mask_id(encode first piece): {cur_mask_id}")
    print(f"tokenizer.pad_token_id: {tok_current.pad_token_id}")
    print(f"tokenizer.pad_token: {tok_current.pad_token}")
    print(f"start_pos: {cur_start}")
    print(f"Special tokens in vocab: {list(tok_current.get_added_vocab().keys())[:10]}...")

    # Print per-sample analysis
    for i in range(cur_inputs.shape[0]):
        ids = cur_inputs[i].tolist()
        lbl = cur_labels[i].tolist()
        attn = cur_attn_mask[i].tolist()
        
        pretty_print_tokens(tok_current, ids, title=f"CURRENT input_ids sample {i}")
        
        # Show attention mask
        pad_positions = [idx for idx, v in enumerate(attn) if v == 0]
        print(f"attention_mask[0 positions]: first 20 -> {pad_positions[:20]} ... total={len(pad_positions)}")
        
        # Show label mask positions (where -100)
        neg100 = [idx for idx, v in enumerate(lbl) if v == -100]
        print(f"labels[-100 positions]: first 20 -> {neg100[:20]} ... total={len(neg100)}")
        
        # Check if PAD positions are masked in labels
        pad_not_masked = [idx for idx in pad_positions if lbl[idx] != -100]
        if pad_not_masked:
            print(f"⚠️  WARNING: {len(pad_not_masked)} PAD positions NOT masked in labels!")
        
        # Check if prompt positions are masked
        prompt_not_masked = [idx for idx in range(cur_start) if lbl[idx] != -100 and attn[idx] == 1]
        if prompt_not_masked:
            print(f"⚠️  WARNING: {len(prompt_not_masked)} prompt positions NOT masked in labels!")

    print("\n" + "="*80)
    print("SUGGESTED METHOD (with all fixes)")
    print("="*80)
    print("[Using separate tokenizer instance: tok_suggested]")
    sug_inputs, sug_labels, sug_attn_mask, sug_start, sug_pad_id, sug_mask_id = suggested_prepare(
        prompts, responses, tok_suggested, args.max_prompt_len, args.max_gen_length
    )
    print(f"\npad_id(tokenizer.pad_token_id): {sug_pad_id}")
    print(f"mask_id(registered special token): {sug_mask_id}")
    print(f"tokenizer.pad_token: {tok_suggested.pad_token}")
    print(f"start_pos: {sug_start}")
    print(f"Special tokens in vocab: {list(tok_suggested.get_added_vocab().keys())[:10]}...")

    for i in range(sug_inputs.shape[0]):
        ids = sug_inputs[i].tolist()
        lbl = sug_labels[i].tolist()
        attn = sug_attn_mask[i].tolist()
        
        pretty_print_tokens(tok_suggested, ids, title=f"SUGGESTED input_ids sample {i}")
        
        # Show attention mask
        pad_positions = [idx for idx, v in enumerate(attn) if v == 0]
        print(f"attention_mask[0 positions]: first 20 -> {pad_positions[:20]} ... total={len(pad_positions)}")
        
        # Show label mask positions (where -100)
        neg100 = [idx for idx, v in enumerate(lbl) if v == -100]
        print(f"labels[-100 positions]: first 20 -> {neg100[:20]} ... total={len(neg100)}")
        
        # Verify PAD positions are masked
        pad_not_masked = [idx for idx in pad_positions if lbl[idx] != -100]
        if pad_not_masked:
            print(f"⚠️  WARNING: {len(pad_not_masked)} PAD positions NOT masked in labels!")
        else:
            print(f"✅ All {len(pad_positions)} PAD positions correctly masked")
        
        # Verify prompt positions are masked
        prompt_not_masked = [idx for idx in range(sug_start) if lbl[idx] != -100 and attn[idx] == 1]
        if prompt_not_masked:
            print(f"⚠️  WARNING: {len(prompt_not_masked)} prompt positions NOT masked in labels!")
        else:
            print(f"✅ Prompt positions correctly masked")

    # Direct diffs
    print("\n" + "="*80)
    print("DIFF SUMMARY")
    print("="*80)
    print(f"start_pos equal? {cur_start == sug_start}")
    print(f"pad_id equal?   {cur_pad_id == sug_pad_id}")
    print(f"mask_id equal?  {cur_mask_id == sug_mask_id}")
    
    # Check tokenizer differences
    print(f"\nTokenizer differences:")
    print(f"  Current vocab size: {len(tok_current)}")
    print(f"  Suggested vocab size: {len(tok_suggested)}")
    print(f"  Vocab size increased: {len(tok_suggested) - len(tok_current)} tokens")
    
    # Check special token registration
    cur_special = set(tok_current.get_added_vocab().keys())
    sug_special = set(tok_suggested.get_added_vocab().keys())
    new_special = sug_special - cur_special
    if new_special:
        print(f"  New special tokens in suggested: {new_special}")
    
    # Check template token tokenization
    test_tokens = ["<|user|>", "<|assistant|>", "<|mdm_mask|>"]
    print(f"\nTemplate token comparison:")
    for token in test_tokens:
        cur_encoded = tok_current.encode(token, add_special_tokens=False)
        sug_encoded = tok_suggested.encode(token, add_special_tokens=False)
        print(f"  '{token}':")
        print(f"    Current:   {cur_encoded} ({len(cur_encoded)} tokens) -> {tok_current.convert_ids_to_tokens(cur_encoded)}")
        print(f"    Suggested: {sug_encoded} ({len(sug_encoded)} tokens) -> {tok_suggested.convert_ids_to_tokens(sug_encoded)}")
    
    print(f"\nSample-wise comparison:")
    for i in range(cur_inputs.shape[0]):
        cur_ids = cur_inputs[i].tolist()
        sug_ids = sug_inputs[i].tolist()
        cur_lbl = cur_labels[i].tolist()
        sug_lbl = sug_labels[i].tolist()
        
        ids_same = cur_ids == sug_ids
        labels_same = cur_lbl == sug_lbl
        
        print(f"\nsample {i}:")
        print(f"  input_ids equal? {ids_same}")
        print(f"  labels equal? {labels_same}")
        
        if not ids_same:
            mism = [(idx, cur_ids[idx], sug_ids[idx]) for idx in range(min(len(cur_ids), len(sug_ids))) if cur_ids[idx] != sug_ids[idx]]
            print(f"  input_ids mismatches (first 10): {mism[:10]}")
        
        if not labels_same:
            cur_neg100 = sum(1 for v in cur_lbl if v == -100)
            sug_neg100 = sum(1 for v in sug_lbl if v == -100)
            print(f"  labels -100 count: current={cur_neg100}, suggested={sug_neg100}, diff={sug_neg100-cur_neg100}")
            
            # Show where labels differ
            lbl_mism = [(idx, cur_lbl[idx], sug_lbl[idx]) for idx in range(min(len(cur_lbl), len(sug_lbl))) if cur_lbl[idx] != sug_lbl[idx]]
            print(f"  labels mismatches (first 10): {lbl_mism[:10]}")


if __name__ == "__main__":
    main()
