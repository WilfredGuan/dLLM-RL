import pandas as pd
import json

# 固定输入和输出文件
input_file = "train-00000-of-00001.parquet"
output_file = "GSM8K-aug-train.json"

print(f"Loading {input_file} ...")
df = pd.read_parquet(input_file)

# 转为 list of dict
records = df.to_dict(orient="records")

result = []
for ex in records:
    result.append({
        "question":ex["question"],
        "ground_truth_answer":ex["answer"]
    })

print(f"Saving {output_file} ...")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"✅ Done! Saved {len(records)} records to {output_file}")