#!/bin/bash

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


# ============================================================================
# CONFIGURATION SECTION
# ============================================================================
# MASTER_ADDR=${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}
# MASTER_PORT=${MASTER_PORT:-29700}
NODE_RANK=${RANK:-0}
# echo "[INFO] NODE_RANK=$NODE_RANK pinging MASTER_ADDR=$MASTER_ADDR..."

# Check network
# if ping -c 1 -W 2 "$MASTER_ADDR" > /dev/null 2>&1; then
#   echo "[NODE_RANK=$NODE_RANK ✅] Successfully reached master node: $MASTER_ADDR"
# else
#   echo "[NODE_RANK=$NODE_RANK ❌] Failed to reach master node: $MASTER_ADDR"
#   exit 1
# fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=ERROR
export TOKENIZERS_PARALLELISM=true

export DATASET_NAME="mbpp"

export TRAIN_DATA_PATH="./data/mbpp"                       # Path to MBPP dataset directory (HuggingFace format)
export VAL_DATA_PATH="./data/mbpp"                         # Path to MBPP dataset directory (same as train, split handled by code)

# Logging and Monitoring
# export WANDB_PROJECT="unigrpo-mbpp-diffucoder"
# export OUTPUT_DIR="/nas/shared/sys2/kyguan/$(date +%Y%m%d_%H%M%S)_diffucoder_SFT_1024_step_1000_3_inner_updates"
export OUTPUT_DIR="/geminicephfs/game-center-recmd/wilfredguan/ckpt/$(date +%Y%m%d_%H%M%S)_llada_mbpp"
# export OUTPUT_DIR="/nas/shared/sys2/kyguan/$(date +%Y%m%d_%H%M%S)_diffucoder_25k_reproduce_0810"
export LOG_INTERVAL=1           # Log training metrics every N steps
export EVAL_INTERVAL=10          # Run validation every N steps
export PLOT_INTERVAL=1           # Generate plots every N epochs
export SAVE_INTERVAL=10         # Save checkpoint every N steps  
export MAX_CHECKPOINTS=10         # Maximum number of regular checkpoints to keep

# Model Configuration
# export MODEL_NAME="/nas/shared/sys2/yanjian/dafny_diffusion/MMaDA/mmada-training-300M_2048/hf_model_sft_v3"
# export MODEL_NAME="/nas/shared/sys2/gaoxin/project/diffucoder/saved_models/diffucoder-training-7B_512_tab-step-25000"
export MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"
export MODEL_DTYPE="bfloat16"

# Training Hyperparameters
# DCoLT settings:
export NUM_EPOCHS=2
export BATCH_SIZE=16 # 64 -- which means the batch_size per GPU is BATCH_SIZE // Num_Processes
export LEARNING_RATE=1e-5
export GROUP_SIZE=8 # 16             # Number of completions per group for GRPO
export DIFFUSION_STEPS=256        # DCoLT:256, Total diffusion timesteps (T); 
export USE_UPM=false

export WEIGHT_DECAY=0.01
export MAX_GRAD_NORM=1.0
export WARMUP_RATIO=0.1
# MmaDa adopted this inner updates
export INNER_UPDATES=3 # MmaDa

# Generation Configuration
export GENERATION_STEPS=256       # DCoLT:256, Sampling steps for generation NOTE: this is only for the N_sampling configuration, which is different from DIFFUSION_STEP
export GEN_LENGTH=256           # DCoLT:256, Generated sequence length

# Note that DCoLT jointly trained a new unmask scheduler
export BLOCK_LENGTH=32          # DCoLT:8, Block length for semi-autoregressive generation
export TEMPERATURE=1.0          # Sampling temperature for diversity
export CFG_SCALE=0            # Classifier-free guidance scale
export MASK_ID=126336           # 4 for ours, 126336 for llada, 151666 for diffucoder

# Reward Configuration - MBPP specific settings
export PARSE_REWARD=0.2         # Reward for successful code parsing
export RUN_REWARD=0.8           # Reward for successful test execution (parse_reward + run_reward should = 1.0)
export DAFNY_EXECUTABLE="/cpfs04/user/gaoxin1/dafny/dafny"  # Not used for MBPP but kept for compatibility
export DAFNY_TIMEOUT=120        # Not used for MBPP but kept for compatibility
export MAX_WORKERS=16

# Resume Training (optional)
# export RESUME_FROM="/nas/shared/sys2/kyguan/20250713_114618_diffucoder_dafny_machine_1/global_step_30"            # Path to checkpoint file, leave empty for fresh start
# export RESUME_FROM="/geminicephfs/game-center-recmd/wilfredguan/ckpt/20250919_111231_llada_gsm8k_11_33/checkpoints/global_step_30"            # Path to checkpoint file, leave empty for fresh start
# export RESUME_FROM="/geminicephfs/game-center-recmd/wilfredguan/ckpt/20250922_124914_llada_gsm8k/checkpoints/global_step_80/"            # Path to checkpoint file, leave empty for fresh start

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "Setting up MBPP training environment..."
echo "Starting cluster training with NODE_RANK=$NODE_RANK"

# Create output directory
if [ "$NODE_RANK" = "0" ]; then
    mkdir -p "${OUTPUT_DIR}"
    echo "[MASTER] Created output directory: ${OUTPUT_DIR}"
fi

# Set optimal CUDA settings
# export CUDA_LAUNCH_BLOCKING=0
export TORCH_BACKENDS_CUDNN_BENCHMARK=true

# Python path setup
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# ============================================================================
# TRAINING EXECUTION
# ============================================================================

echo "Starting UniGRPO training with the following configuration:"
echo "  Dataset: ${DATASET_NAME}"
echo "  Model: ${MODEL_NAME}"
echo "  Group Size: ${GROUP_SIZE}"
echo "  Inner Updates: ${INNER_UPDATES}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Parse Reward: ${PARSE_REWARD}"
echo "  Run Reward: ${RUN_REWARD}"
echo "  Output Directory: ${OUTPUT_DIR}"


# Run training
accelerate launch --config_file accelerate_configs/dlc_multinode.yaml \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision bf16 \
    main.py \
    --train_data_path "${TRAIN_DATA_PATH}" \
    --val_data_path "${VAL_DATA_PATH}" \
    --model_name "${MODEL_NAME}" \
    --model_dtype "${MODEL_DTYPE}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --group_size "${GROUP_SIZE}" \
    --inner_updates "${INNER_UPDATES}" \
    --diffusion_steps "${DIFFUSION_STEPS}" \
    --generation_steps "${GENERATION_STEPS}" \
    --gen_length "${GEN_LENGTH}" \
    --block_length "${BLOCK_LENGTH}" \
    --temperature "${TEMPERATURE}" \
    --cfg_scale "${CFG_SCALE}" \
    --parse_reward "${PARSE_REWARD}" \
    --compile_reward "${COMPILE_REWARD:-0.0}" \
    --ensures_match_reward "${ENSURES_MATCH_REWARD:-0.0}" \
    --dafny_executable "${DAFNY_EXECUTABLE}" \
    --dafny_timeout "${DAFNY_TIMEOUT}" \
    --max_workers "${MAX_WORKERS}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_interval "${LOG_INTERVAL}" \
    --eval_interval "${EVAL_INTERVAL}" \
    --plot_interval "${PLOT_INTERVAL}" \
    --save_interval "${SAVE_INTERVAL}" \
    --use_upm "${USE_UPM}" \
    --dataset_name "${DATASET_NAME}" \
    --mask_id "${MASK_ID}" \
    ${RESUME_FROM:+--resume_from "${RESUME_FROM}"}

echo ""
echo "MBPP training completed successfully!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Checkpoints: ${OUTPUT_DIR}/checkpoints/"
echo "Plots: ${OUTPUT_DIR}/plots/"

# ============================================================================
# POST-TRAINING TASKS
# ============================================================================

# Copy training script to output directory for reproducibility
cp "$0" "${OUTPUT_DIR}/train_script.sh"