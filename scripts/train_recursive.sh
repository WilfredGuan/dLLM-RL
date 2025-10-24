#!/bin/bash

# ============================================
# Training Script for Recursive LLaDA
# ============================================

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/geminicephfs/game-center-recmd/wilfredguan/.cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export TORCH_DISTRIBUTED_TIMEOUT=36000
export NCCL_TIMEOUT=36000
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=ERROR
export TOKENIZERS_PARALLELISM=true


# ============================================
# Configuration
# ============================================

# Model configuration
MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"  # Change this to your model

# Project name (will create directory for checkpoints and logs)
PROJECT_NAME="sft_llada_recursive"

# Checkpoint save directory (relative to project root)
# Final path will be: ./${PROJECT_NAME}/ckpt/
CKPT_DIR="/geminicephfs/game-center-recmd/wilfredguan/ckpt/${PROJECT_NAME}/ckpt"
# Config file
CONFIG_FILE="configs/sft_llada_recursive.yaml"

# ============================================
# Training Parameters (Optional - can also set in yaml)
# ============================================

# Number of GPUs to use
NUM_GPUS=8

# ============================================
# Create necessary directories
# ============================================

mkdir -p ${PROJECT_NAME}
mkdir -p ${PROJECT_NAME}/ckpt
mkdir -p ${PROJECT_NAME}/logs
mkdir -p ${PROJECT_NAME}/ckpt/tensorboard_logs

echo "============================================"
echo "Training Configuration"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Project: ${PROJECT_NAME}"
echo "Checkpoint Dir: ${CKPT_DIR}"
echo "Config: ${CONFIG_FILE}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "============================================"

# ============================================
# Update config file with actual paths
# ============================================

# Create a temporary config with updated paths
TMP_CONFIG="${PROJECT_NAME}/config_runtime.yaml"
cp ${CONFIG_FILE} ${TMP_CONFIG}

# Use sed to update the config (works on Linux)

echo "Updated config saved to: ${TMP_CONFIG}"
echo ""

# ============================================
# Launch Training
# ============================================

# ACCELERATE_CONFIG="accelerate_configs/1_node_8_gpus_deepspeed_zero3.yaml"

if [ ${NUM_GPUS} -gt 1 ]; then
    echo "Launching multi-GPU training with ${NUM_GPUS} GPUs..."
    # accelerate launch --config_file ${ACCELERATE_CONFIG} \
    accelerate launch --num_processes=${NUM_GPUS} \
        --mixed_precision="bf16" \
        train/sft_llada.py \
        --config ${TMP_CONFIG}
else
    echo "Launching single-GPU training..."
    python train/sft_llada.py \
        --config ${TMP_CONFIG}
fi

echo ""
echo "============================================"
echo "Training completed!"
echo "Checkpoints saved to: ${CKPT_DIR}"
echo "TensorBoard logs: ${PROJECT_NAME}/ckpt/tensorboard_logs"
echo ""
echo "To view TensorBoard:"
echo "  tensorboard --logdir=${PROJECT_NAME}/ckpt/tensorboard_logs"
echo "============================================"
