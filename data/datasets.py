# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import os
import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional
import numpy as np
from datasets import load_dataset, load_from_disk
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers import AutoTokenizer, AutoModel
from jinja2 import Template


# ========================
# 模型配置：不同prompt & eos_token
# ========================
MODEL_CONFIGS = {
    "dream": {
        "template": '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
This is the problem:
{{problem}}<|im_end|>
You need to put your final answer in \\boxed{}. 
<|im_start|>assistant
''',
        "eos": "<|im_end|>"
    },
    "diffucoder": {  # diffucoder和dream的prompt
        "template": '''<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
This is the problem:
{{problem}}<|im_end|>
You need to put your final answer in \\boxed{}.
<|im_start|>assistant
''',
        "eos": "<|im_end|>"
    },
    "llada": {
        "template": """<|startoftext|><|start_header_id|>user<|end_header_id|>
This is the problem:
{{problem}}
You need to put your final answer in \\boxed{}.<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>
""",
        "eos": "<|eot_id|>"
    },
    "mmada": {  # mmada和llada的prompt
        "template": """<|startoftext|><|start_header_id|>user<|end_header_id|>
This is the problem:
{{problem}}
You need to put your final answer in \\boxed{}. <|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>
""",
        "eos": "<|eot_id|>"
    },
    "sdar": {  # sdar非CoT模式
        "template": '''<|im_start|>user
This is the problem:
{{problem}}<|im_end|>
You need to put your final answer in \\boxed{}. <|im_start|>assistant
''',
        "eos": "<|im_end|>"
    },
    "sdar_cot": {  # sdar启用CoT模式
        "template": '''<|im_start|>user
{{problem}}
Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>
<|im_start|>assistant
''',
        "eos": "<|im_end|>"
    },
}




# 暂时不用这个，用process_sft + TrainDataset
# def get_dataset(data_name, split, tokenizer, model='llada', max_size=1000000000):

#     cfg = MODEL_CONFIGS[model]
#     template_str = cfg["template"]
#     t = Template(template_str)
#     eos_token = cfg["eos"]

#     def tokenize_sample(sample):

#         input_content = t.render(problem=sample["question"])
        
#         if data_name == 'gsm8k':
#             answer = sample["answer"].split('####')[1].strip()
#             trajectory = sample["answer"].split('####')[0].strip()
#         elif data_name == 'gsm8k-aug':
#             answer = sample["answer"] 
#             trajectory = '\n'.join(sample["steps"]).strip()
#         else:
#             raise ValueError(f"Unsupported dataset type: {data_name}")

#         target_content = f"\n{trajectory}\nThe final answer is \\boxed{{{answer}}}"
#         if not target_content.endswith(eos_token):
#             target_content += eos_token

#         prompt_ids = tokenizer.encode(input_content, add_special_tokens=False)
#         target_ids = tokenizer.encode(target_content, add_special_tokens=False)
        
#         sample_out = {
#             "prompt_ids": prompt_ids,
#             "target_ids": target_ids,
#         }

#         return sample_out
        

#     if data_name == 'gsm8k':
#         ds = load_dataset('openai/gsm8k', "main")
#     elif data_name == 'gsm8k-aug':
#         ds = load_dataset('whyNLP/gsm8k-aug-nl')
#     elif data_name == 'mbpp':
#         ds = load_dataset("mbpp")
#     else:
#         raise ValueError(f"Unsupported dataset type: {data}")

#     dataset = ds[split]
#     if max_size < len(dataset):
#         dataset = dataset.select(range(max_size))

#     data = dataset.to_list()
#     keys = data[0].keys()
#     dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})

#     if torch.cuda.device_count() > 1:
#         if dist.get_rank() == 0:
#             processed_dataset = [
#                 dataset.map(
#                     tokenize_sample, remove_columns=list(dataset.features), num_proc=8
#                 )
#             ]
#         else:
#             processed_dataset = [None]
#         dist.broadcast_object_list(processed_dataset, src=0)
#         dataset = processed_dataset[0]

#     else:
#         dataset = dataset.map(
#             tokenize_sample, remove_columns=list(dataset.features), num_proc=8
#         )

#     return dataset


@torch.no_grad()
def prepare_inputs_and_labels_for_token_ids(input_ids: torch.Tensor, prompt_len: torch.Tensor, mask_id: int, eps=1e-3):

    labels_lm = input_ids.clone()

    b, l = input_ids.shape
    times = (1 - eps) * torch.rand((b, 1), device=input_ids.device) + eps
    p_mask = times.expand(-1, l)

    # Only allow masking after the prompt
    arange = torch.arange(l, device=input_ids.device)[None, :].expand(b, l)
    maskable = arange >= prompt_len[:, None]

    masked_indices = (torch.rand((b, l), device=input_ids.device) < p_mask) & maskable
    noisy_batch = torch.where(masked_indices, mask_id, input_ids)

    answer_lengths = l - prompt_len[:, None]
    answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
    
    return noisy_batch.to(torch.long), labels_lm.to(torch.long), p_mask


def make_collate_fn_pad(pad_id):
    def collate_fn(batch):
        prompt_ids = [torch.tensor(example["prompt_ids"]) for example in batch]
        target_ids = [torch.tensor(example["target_ids"]) for example in batch]

        prompt_ids_padded = torch.nn.utils.rnn.pad_sequence(prompt_ids, batch_first=True, padding_side='left', padding_value=pad_id)
        target_ids_padded = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_side='right', padding_value=pad_id)
        input_ids_padded = torch.cat([prompt_ids_padded, target_ids_padded], dim=1)


        batch_prompt_len = prompt_ids_padded.size(1)

        prompt_lens = [batch_prompt_len for _ in batch]
        prompt_lens_tensor = torch.tensor(prompt_lens, dtype=torch.long)   # 这里是包含了prompt里的pad的，每个batch内部是一样的

        output = {
            "input_ids": input_ids_padded,
            "prompt_len": prompt_lens_tensor,
        }

        return output
    return collate_fn