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