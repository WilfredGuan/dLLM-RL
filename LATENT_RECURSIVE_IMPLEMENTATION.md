# Latent Recursive Training 实现总结

## 修改概览

已完成基于 `modeling_recursive_llada.py` 的 latent recursive 训练功能实现，支持在训练和推理时进行多步 hidden space 递归处理。

---

## 1. 配置文件修改

### 1.1 新增配置参数 (`configuration_llada.py`)

在 `ModelConfig` 类中添加：
- `use_latent_recursive: bool = False` - 是否启用 latent recursive
- `max_latent_recursive_steps: int = 32` - 支持的最大递归步数

### 1.2 新建训练配置 (`configs/sft_llada_recursive.yaml`)

新增参数：
```yaml
training:
  use_latent_recursive: True
  latent_recursive_steps: 16
  recursive_in_training: True
  block_size: 32

generation:
  diffusion_time_steps: 128
  use_latent_recursive: True
  latent_recursive_steps: 16

logging:
  use_tensorboard: True
  log_dir: "tensorboard_logs"
  log_interval: 10
```

---

## 2. 模型架构修改 (`modeling_recursive_llada.py`)

### 2.1 添加 Latent Step Embedding

在 `LLaDAModel.__init__()` 中：
```python
if config.use_latent_recursive:
    self.latent_step_embedding = nn.Embedding(
        config.max_latent_recursive_steps,
        config.d_model
    )
```

### 2.2 新增 `forward_latent_only()` 方法

功能：在 hidden space 进行 forward，不投射到 vocab
- 输入：hidden_states, latent_step, attention_bias
- 处理：添加 step embedding → 通过所有 blocks → layer norm
- 输出：更新后的 hidden_states

---

## 3. 训练逻辑修改 (`train/sft_llada.py`)

### 3.1 命令行参数支持

```python
parser.add_argument('--config', type=str, 
                   default='configs/sft_llada.yaml')
```

### 3.2 TensorBoard 日志

- 移除 wandb 依赖
- 使用 `SummaryWriter` 记录到 `{project}/ckpt/tensorboard_logs/`
- 记录指标：loss, lr, latent_recursive_steps

### 3.3 Latent Recursive 训练流程

```python
def forward_process(input_ids, labels, p_mask_lm):
    if use_latent_recursive and recursive_in_training:
        # 1. 获取初始 hidden states
        hidden_states = model.model.transformer.wte(input_ids)
        
        # 2. Latent recursive 阶段（N步）
        for step in range(latent_recursive_steps):
            hidden_states = model.model.forward_latent_only(
                hidden_states, step, attention_bias
            )
        
        # 3. 投射到 vocab
        logits = F.linear(hidden_states, wte.weight)
    
    # 4. 计算 loss（对所有 mask 位置并行）
    return loss
```

---

## 4. 工具函数修改 (`train/utils.py`)

修改 `get_config()` 支持自定义配置路径：
```python
def get_config(config_path=None):
    if config_path is not None:
        yaml_conf = OmegaConf.load(config_path)
    else:
        yaml_conf = OmegaConf.load(cli_conf.config)
```

---

## 5. 核心设计逻辑

### 5.1 训练阶段

```
输入序列（带mask）
    ↓
初始 embedding
    ↓
Latent Recursive（N步，hidden space）
    ├─ Step 0: hidden + step_emb[0] → blocks → norm
    ├─ Step 1: hidden + step_emb[1] → blocks → norm
    └─ Step N-1: hidden + step_emb[N-1] → blocks → norm
    ↓
投射到 vocab
    ↓
并行计算所有 masked 位置的 loss
```

### 5.2 推理阶段（Generate）

解码效率配平公式：
```python
original_efficiency = max_gen_length / diffusion_time_steps
steps_per_block_original = block_size / original_efficiency
remaining_steps = steps_per_block_original - latent_recursive_steps
unmask_speed = block_size / remaining_steps  # tokens per step
```

示例（你的场景）：
- diffusion_time_steps = 128, max_gen_length = 128
- block_size = 32, latent_recursive_steps = 16
- 原始效率：1 token/step
- 每个 block：32 steps
- 扣除 latent：剩余 16 steps
- unmask 速度：2 tokens/step ✓

---

## 6. 使用方法

### 6.1 训练

```bash
# 使用新配置训练
python train/sft_llada.py --config configs/sft_llada_recursive.yaml

# 使用原始配置（向后兼容）
python train/sft_llada.py --config configs/sft_llada.yaml
```

### 6.2 查看日志

```bash
tensorboard --logdir {project}/ckpt/tensorboard_logs/
```

### 6.3 测试配置

```bash
python test_recursive_config.py
```

---

## 7. 关键优势

1. **向后兼容**：不影响原始 `modeling_llada.py`
2. **灵活控制**：通过配置文件开关 latent recursive
3. **解码效率配平**：自动计算 unmask 速度保持总效率一致
4. **本地日志**：TensorBoard 替代 wandb，适合内网环境
5. **最小侵入**：核心逻辑集中在 `forward_latent_only()` 方法

---

## 8. 待完成事项

### 8.1 必需配置

修改 `configs/sft_llada_recursive.yaml`：
- `model.pretrained_model`: 你的模型路径
- `dataset.optimization_data`: 你的数据集名称

### 8.2 Generate 方法实现（可选）

如需推理时使用 latent recursive，需在 `LLaDAModelLM` 中实现 `generate()` 方法，参考之前方案中的 `_generate_with_latent_recursive()`。

### 8.3 模型初始化

首次使用时需要初始化 `latent_step_embedding`：
```python
model_config.use_latent_recursive = True
model = LLaDAModel(model_config, init_params=True)
```

---

## 9. 文件清单

修改的文件：
- ✓ `models/llada/configuration_llada.py` - 添加配置参数
- ✓ `models/llada/modeling_recursive_llada.py` - 核心功能实现
- ✓ `train/sft_llada.py` - 训练逻辑和日志
- ✓ `train/utils.py` - 配置加载函数

新增的文件：
- ✓ `configs/sft_llada_recursive.yaml` - 新配置文件
- ✓ `test_recursive_config.py` - 测试脚本

未修改的文件：
- ✓ `models/llada/modeling_llada.py` - 保持原样

---

## 10. 注意事项

1. **Step Embedding 初始化**：使用小标准差（0.02）初始化，避免初期扰动过大
2. **Attention Bias**：训练时可能需要根据具体需求准备 attention_bias
3. **显存优化**：如遇显存不足，可启用 `gradient_checkpointing_enable: True`
4. **Batch 处理**：不同 recursive_steps 的样本可能需要分开处理

---

## 11. 调试建议

1. 先用小数据集测试（如 100 条）
2. 设置较小的 `latent_recursive_steps`（如 4）验证流程
3. 检查 TensorBoard 中 loss 曲线是否正常
4. 对比 `use_latent_recursive=False` 和 `True` 的训练效果
