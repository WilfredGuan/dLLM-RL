import os
import json
import argparse
from jinja2 import Template
from pathlib import Path
from tqdm import tqdm

from typing import List, Dict, Optional
from datasets import load_dataset, Dataset
from accelerate import PartialState


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


def preprocess(data, model_type):
    """根据不同模型生成sft数据"""
    config = MODEL_CONFIGS[model_type]
    template = Template(config["template"])
    eos_token = config["eos"]

    processed = []
    for item in data:
        prompt = template.render(problem=item["prompt"])
        response = item["response"].rstrip() 

        # 防止重复添加eos
        if not response.endswith(eos_token):
            response += eos_token

        processed.append({"prompt": prompt, "response": response})
    return processed


def sft_preprocess_gsm8k_aug_nl(
    split: str,
    model: str = "llada",
    max_size: Optional[int] = None,
    num_proc: Optional[int] = None,
    distributed_shard: bool = True,
) -> List[Dict[str, str]]:
    """
    并行预处理 whyNLP/gsm8k-aug-nl
    - num_proc: 并行进程数；默认使用 os.cpu_count()
    - distributed_shard: 若在 torch.distributed 已初始化的环境下，按 rank 自动分片（适配多卡）
    """
    # 1) 载入数据（直接用 split 字符串，避免多余内存）
    ds: Dataset = load_dataset("whyNLP/gsm8k-aug-nl", split=split)

    # 2) 可选截断
    if max_size is not None:
        ds = ds.select(range(max_size))

    # 3) 分布式场景下自动切分数据（每张卡/每个进程处理一份）
    if distributed_shard:
        try:
            import torch
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world = dist.get_world_size()
                # contiguous=True 保证每个分片是连续块，利于 I/O
                ds = ds.shard(num_shards=world, index=rank, contiguous=True)
        except Exception:
            # 没有 torch 或未初始化分布式时，忽略
            pass

    # 4) 并行 map
    cfg = MODEL_CONFIGS[model]
    template_str = cfg["template"]
    eos_token = cfg["eos"]

    # batched 函数里每个进程各自编译一次 Template，避免 pickling 问题
    def _map_batch(batch):
        t = Template(template_str)
        prompts = []
        responses = []
        for q, steps, ans in zip(batch["question"], batch["steps"], batch["answer"]):
            prompt = t.render(problem=q)
            trajectory = "\n".join(steps).strip()
            resp = f"{trajectory}\nThe final answer is \\boxed{{{ans}}}"
            if not resp.endswith(eos_token):
                resp += eos_token
            prompts.append(prompt)
            responses.append(resp)
        return {"prompt": prompts, "response": responses}

    # 默认吃满 CPU
    if num_proc is None:
        num_proc = max(1, os.cpu_count() or 1)

    ds = ds.map(
        _map_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=ds.column_names,   # 只保留我们需要的两列
        desc="SFT preprocess GSM8K-aug-nl",
        load_from_cache_file=True,        # 复用缓存，加速二次运行
    )

    # 5) 返回与原函数一致的结构
    return [{"prompt": p, "response": r} for p, r in zip(ds["prompt"], ds["response"])]


def main():
    parser = argparse.ArgumentParser(description="Preprocess SFT dataset")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", type=str, default="./", help="Output directory")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model type: dream / diffucoder / llada / mmada / sdar / sdar_cot")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = preprocess(data, args.model)

    input_name = Path(args.input).stem
    output_file = Path(args.output_dir) / f"sft_{input_name}_{args.model}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(processed_data)} samples → {output_file}")


if __name__ == "__main__":
    main()

# python preprocess_sft.py --input demon_openr1math.json --model llada
# python preprocess_sft.py --input demon_openr1math.json --model sdar_cot

