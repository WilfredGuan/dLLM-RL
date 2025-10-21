#!/bin/bash
# wandb_sync.sh
# 每隔5分钟自动同步wandb offline runs

# 离线run目录（根据实际路径修改，比如wandb目录下）
WANDB_DIR="/mnt/shared-storage-user/gaoxin1/project/dLLM-RL/wandb"

while true; do
    echo "[$(date)] 开始同步..."
    wandb sync "$WANDB_DIR"/offline-run-*
    echo "[$(date)] 同步完成，休眠5分钟..."
    sleep 300   # 300秒 = 5分钟
done