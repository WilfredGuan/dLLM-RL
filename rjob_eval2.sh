#!/bin/bash                                                                                                         
# cd /mnt/shared-storage-user/gaoxin1                                                                               
# source setup_env.sh                                                                                               
# conda activate contok                                                                                             

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# -------- Conda 初始化 --------
__conda_setup="$('/mnt/shared-storage-user/gaoxin1/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/shared-storage-user/gaoxin1/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/shared-storage-user/gaoxin1/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/shared-storage-user/gaoxin1/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup


# -------- huggingface 模型 --------
export HF_HOME=/mnt/shared-storage-user/gaoxin1/.cache
export MODELSCOPE_CACHE=/mnt/shared-storage-user/gaoxin1/.cache/modelscope
export WANDB_CACHE_DIR=/mnt/shared-storage-user/gaoxin1/.cache
export TORCH_EXTENSION_DIR=/mnt/shared-storage-user/gaoxin1/.cache

conda activate dlm
cd /mnt/shared-storage-user/gaoxin1/project/dLLM-RL


python eval.py config=configs/llada_eval_recursive.yaml rollout.unmask_token_number_per_step=2