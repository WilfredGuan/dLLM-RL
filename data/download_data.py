import argparse
from huggingface_hub import hf_hub_download
import shutil
import json
from datasets import load_dataset


parser = argparse.ArgumentParser(description="Download a dataset from HF hub")
parser.add_argument(
    "--dataset",
    choices=[
        "PrimeIntellect",
        "MATH_train",
        "demon_openr1math",
        "MATH500",
        "GSM8K",
        "AIME2024",
        "LiveBench",
        "LiveCodeBench",
        "MBPP",
        "HumanEval",
        "whynlp/gsm8k-aug",
        "whynlp/gsm8k-aug-nl",
    ],
    required=True,
    help="Which dataset to download",
)
args = parser.parse_args()
dataset = args.dataset


if dataset == "MATH_train" or dataset == "PrimeIntellect" or dataset == "demon_openr1math" or dataset == "whynlp/gsm8k-aug" or dataset == "whynlp/gsm8k-aug-nl":
    split = "train"
else:
    split = "test"

# Handle datasets with namespace (e.g., whynlp/gsm8k-aug)
if "/" in dataset:
    repo_id = dataset
    dataset_name = dataset.split("/")[-1]
    
    # For custom datasets, use load_dataset and save to JSON
    print(f"Loading dataset {repo_id} (split: {split})...")
    ds = load_dataset(repo_id, split=split)
    
    output_file = f"./{dataset_name}.json"
    print(f"Saving to {output_file}...")
    
    # Convert to list of dicts and save as JSON
    data_list = [item for item in ds]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully saved {len(data_list)} samples to {output_file}")
else:
    repo_id = f"Gen-Verse/{dataset}"
    dataset_name = dataset
    
    # For Gen-Verse datasets, use the original method
    cached_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"{split}/{dataset_name}.json"
    )
    shutil.copy(cached_path, f"./{dataset_name}.json")
    print(f"Successfully downloaded {dataset_name}.json")
