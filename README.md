10.28
修复 `prepare_inputs_and_labels_for_token_ids` and loss in `process_forward`

1. 每个数据至少有8个<|endoftext|>，补在 collate_fn 中，因为考虑到最长的那个sample没有<|endoftext|>，只用right pad (as in d1 and mmada)
2. 每个数据的mask最多存在在尾部的前8个<|endoftext|>中，后面就没必要算了，不然大量的mask都在尾部
3. 部分样本很浪费，rand_mask全是false，选择对这种样本反转（即回答全部mask）



```bash
accelerate launch \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 8888 \
  --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero3.yaml \
  train/sft_llada.py \
  config=configs/sft_llada_recursive.yaml > output.txt 2>&1
```

```bash
python eval.py config=configs/llada_eval_recursive.yaml
```