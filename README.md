# Usage 
## Training
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
## Evaluation
```bash
python eval.py config=configs/llada_eval_recursive.yaml > eval.txt 2>&1    
```

# Experiment Log
## 10.29 - 10.30
### Model Setting

- f_L = block[:16], f_H = block[16:]
- Introduced halt_max_steps (from TRM). 但是目前为1,并没有加入异步的停止策略
- Parameters:
  -- H_cycles = 2
  -- L_cycles = 4
  -- halt_max_steps = 1
- During inference, each step decodes a fixed number of tokens (unmask_token_number_per_step).
- Trained for 25 epochs.

### Version 1 Pseudo Code

```python
for batch in train_loader:

    # === Data preparation ===
    x, label = prepare(batch)                           # Add noise, masking, etc.
    z, y = init_from_normalization_dist()               # Initialize latent and hierarchical states

    # === Recursive halting steps ===
    for t in range(halt_max_steps):

        # (1) Hierarchical reasoning with no grad for first H_cycles - 1
        with torch.no_grad():
            for h in range(H_cycles - 1):               # Outer loop (high-level)
                for l in range(L_cycles):               # Inner loop (low-level recursion)
                    z = f_L(x, y + z)                   # Latent update
                y = f_H(y, z)                           # Hierarchical update

        # (2) Final step with gradient
        for l in range(L_cycles):
            z = f_L(x, y + z)                           # Optimize z
        y = f_H(y, z)                                   # Optimize y

        # (3) Compute loss
        y_hat = LM_head(y)
        loss = dLLM_loss(y_hat, label)

        # (4) Backpropagation and optimization
        loss.backward()
        opt.step()
        opt.zero_grad()

        # (5) Detach hidden states for next halting step
        z, y = z.detach(), y.detach()
```
### Note 

但实际上后面我想把 `halt_max_steps` 的循环和 `H_cycles` 合并，因为我觉得小模型设置这个是为了更深的深度，而且去监督中间步骤，而大模型本来就已经有这样的深度，recursive 应该只用来 refine latent reasoning z

### Version 1 Results on GSM8K (Full Test Set)

修改了reward.py，加入 `pass@k` 的计算

#umask/step: unmask_token_number_per_step

| Model      | #Unmask/step | pass@1 | pass@5 | pass@10 |
|-------------|--------------|--------|--------|---------|
| Version 1   | 1            |        |        |         |
| Version 1   | 2            |        |        |         |
| Version 1   | 4            |        |        |         |
| Version 1   | 8            |        |        |         |

### 
Todo list:
- [ ] version 1 的一些其他设置variants
- [ ] version 1 RL
- [ ] version 试一个更难的数据集，ARC? not sure，但这才应该是终极目标
- [ ] inference 确实得加速，不然测试和 RL 都会好慢


## 10.28

修复 `prepare_inputs_and_labels_for_token_ids` and loss in `process_forward`

1. 每个数据至少有 8 个 <|endoftext|>，补在 collate_fn 中，因为考虑到最长的那个sample没有<|endoftext|>，只用right pad (as in d1 and mmada)
2. 每个数据的mask最多存在在尾部的前 16 个<|endoftext|>中，后面就没必要算了，不然大量的mask都在尾部
3. 部分样本很浪费，rand_mask全是false，选择对这种样本反转（即回答全部mask）