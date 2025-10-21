import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import argparse
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.optim import AdamW

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed


from train.utils import get_config, flatten_omega_conf, AverageMeter

from models import LLaDAModelLM, LLaDAModelLMRecursive, InnerCarry
from train.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error
from data.preprocess_sft import sft_preprocess_gsm8k_aug_nl

from torch.utils.data import Dataset, DataLoader


try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

os.environ["WANDB_MODE"] = "offline"


logger = get_logger(__name__, log_level="INFO")


class TrainDataset(Dataset):
    def __init__(self, inputs, labels, pmasks, original_texts=None):
        self.inputs = inputs
        self.labels = labels
        self.pmasks = pmasks
        self.original_texts = original_texts  # List of (prompt, response) tuples

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.labels[idx],
            self.pmasks[idx]
        )
    
    def get_original_text(self, idx):
        """Get original prompt/response for debugging"""
        if self.original_texts is not None and idx < len(self.original_texts):
            return self.original_texts[idx]
        return None, None




def main():
    # #########################
    # # Parse arguments       #
    # #########################
    # parser = argparse.ArgumentParser(description="SFT training for LLaDA")
    # parser.add_argument(
    #     '--config',
    #     type=str,
    #     default='configs/sft_llada.yaml',
    #     help='Path to config yaml file'
    # )
    # args = parser.parse_args()

    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    project_name = config.experiment.project
    pretrained_model = config.model.pretrained_model

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.project) / "logs")

    # ---- New: choose logging backend from config ----
    log_cfg = config.get('logging', {}) or {}
    backend = (log_cfg.get('backend') or 'none').lower()
    assert backend in {'wandb', 'tensorboard', 'none'}, f"logging.backend must be one of ['wandb','tensorboard','none'], got {backend}"

    # accelerate will manage the chosen tracker(s)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=None if backend == 'none' else backend,
        project_dir=str(Path(config.experiment.project) / "logs"),
        split_batches=True,
    )

    # init trackers (wandb/tensorboard). Config会被记录到后端
    # 这里把 OmegaConf 展平，方便在 UI 里查看
    flat_cfg = OmegaConf.to_container(config, resolve=True)

    if backend != 'none':
        accelerator.init_trackers(
            project_name=config.experiment.project,  # 统一使用这个
            config=flat_cfg
        )
        if backend == 'tensorboard':
            accelerator.print(f"[Logging] Using TensorBoard. Logs under: {accelerator.project_dir}")
        else:
            accelerator.print(f"[Logging] Using Weights & Biases project: {config.experiment.project}")
    else:
        accelerator.print("[Logging] No online/offline logging backend (none)")
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # Skip wandb initialization - using tensorboard instead

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = Path(config.experiment.project) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len,
                                       max_gen_length=config.training.max_gen_length,
                                       ignore_id=-100)

    # Choose model class based on config
    use_latent_recursive = config.training.get('use_latent_recursive', False)
    logger.info(f"use_latent_recursive: {use_latent_recursive}")

    if use_latent_recursive:
        logger.info("Loading LLaDAModelLMRecursive (with latent recursive support)")
        # Load config first and modify it before loading model
        from models.llada.configuration_llada import LLaDAConfig
        model_config = LLaDAConfig.from_pretrained(pretrained_model)
        R = config.training.get('R', 1)
        model_config.use_latent_recursive = True
        model_config.max_latent_recursive_steps = R

        # Load model with modified config
        model = LLaDAModelLMRecursive.from_pretrained(
            pretrained_model, 
            config=model_config,
            torch_dtype=torch.bfloat16
        )

        logger.info(f"Enabled latent recursive with max_steps={R}")
    else:
        logger.info("Loading LLaDAModelLM (original)")
        model = LLaDAModelLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16)

    ## init Low_level, High_level
    model.model.low_level_end_idx = config.training.low_level_end_idx

    model = model.to(accelerator.device)

    # Enable gradient checkpointing if configured
    if config.training.get('gradient_checkpointing_enable', False):
        logger.info("Enabling gradient checkpointing...")
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        elif hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        logger.info("Gradient checkpointing enabled")


    # GPU Memory Check after model loading
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"[After Model Load] GPU {accelerator.device} Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")

    mask_id = tokenizer.encode('<|mdm_mask|>')[0]
    pad_id = tokenizer.encode('<|endoftext|>')[0]

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    def collapse_k_unique(lst, k: int):
        if k <= 0:
            raise ValueError("k must be > 0")
        uniq = sorted(set(lst))

        mapping = {}
        n = len(uniq)
        for idx, val in enumerate(uniq):
            group = idx // k
            end_idx = min((group + 1) * k - 1, n - 1)
            rep = uniq[end_idx]
            mapping[val] = rep
        return [mapping[x] for x in lst]

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    @torch.no_grad()
    def prepare_inputs_and_labels_for_text(
        prompt, response, step_map, eps=1e-3, mask_id=mask_id
    ):
        input_ids_lm, labels_lm, start_pos, drop_num = uni_prompting((prompt, response))

        B, L = input_ids_lm.shape
        max_gen_len = config.training.max_gen_length
        if max_gen_len + start_pos < L:
            L_after = start_pos + max_gen_len
        else:
            L_after = L
        input_ids_lm = input_ids_lm[:, :L_after]
        labels_lm = labels_lm[:, :L_after]

        lower = config.training.lower_p
        upper = config.training.upper_p

        if config.training.method == "semi-ar":

            noisy_list, label_list, pmask_list = [], [], []

            device = input_ids_lm.device
            B, L   = input_ids_lm.shape

            for b in range(B):
                # 1) transform step_map
                order_list = list(step_map[b])
                order_list = collapse_k_unique(order_list, config.training.block_size)
                order = torch.as_tensor(order_list, device=device)
                order_full = torch.full((L_after,), -1, device=device)
                order_full[start_pos:] = order[: L_after - start_pos]

                uniq_steps = torch.unique(order_full[start_pos:], sorted=True)

                base_ids = input_ids_lm[b]  # (L,)

                if config.training.post_num is not None:
                    pad_mask_b = (base_ids == pad_id)
                    pad_mask_b[:start_pos] = False
                    keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
                    tail_pad_b       = pad_mask_b & ~keep_first_pad_b
                else:
                    keep_first_pad_b = torch.zeros(L, dtype=torch.bool, device=device)
                    tail_pad_b       = torch.zeros(L, dtype=torch.bool, device=device)

                for i in range(0, len(uniq_steps)):

                    block_mask = (order_full == uniq_steps[i])
                    p = torch.empty(L, device=device).uniform_(lower, upper)
                    block_mask = (torch.rand(L, device=device) < p) & block_mask

                    noisy_ids = base_ids.clone()
                    mask_pos  = (order_full > uniq_steps[i]) | block_mask
                    noisy_ids[mask_pos] = mask_id

                    pmask_this = block_mask & ~tail_pad_b

                    if not pmask_this.any():
                        continue

                    noisy_list.append(noisy_ids)
                    label_list.append(labels_lm[b])
                    pmask_list.append(pmask_this)

                del order, order_full, uniq_steps

            noisy_batch = torch.stack(noisy_list)
            labels_lm   = torch.stack(label_list)
            p_mask      = torch.stack(pmask_list)

        elif config.training.method == "random_masking":
            m = config.training.mask_times_per_sample
            B, L = input_ids_lm.shape
            device = input_ids_lm.device

            noisy_list, label_list, pmask_list = [], [], []
            for b in range(B):
                base_ids  = input_ids_lm[b]
                label_ids = labels_lm[b]

                if config.training.post_num is not None:
                    pad_mask_b = (base_ids == pad_id)
                    pad_mask_b[:start_pos] = False
                    keep_first_pad_b = pad_mask_b & (torch.cumsum(pad_mask_b.int(), dim=0) <= config.training.post_num)
                    tail_pad_b       = pad_mask_b & ~keep_first_pad_b
                else:
                    keep_first_pad_b = torch.zeros(L, dtype=torch.bool, device=device)
                    tail_pad_b       = torch.zeros(L, dtype=torch.bool, device=device)

                for _ in range(m):
                    t = (upper - lower) * torch.rand(1, device=device) + lower
                    rand_mask = torch.rand(L, device=device) < t
                    rand_mask[:start_pos] = False
                    rand_mask = rand_mask & ~tail_pad_b

                    if not rand_mask.any():
                        continue

                    noisy_ids = base_ids.clone()
                    noisy_ids[rand_mask]   = mask_id
                    noisy_ids[tail_pad_b]  = mask_id

                    noisy_list.append(noisy_ids)
                    label_list.append(label_ids)
                    pmask_list.append(rand_mask)

            noisy_batch = torch.stack(noisy_list)    # (B*m, L)
            labels_lm   = torch.stack(label_list)
            p_mask      = torch.stack(pmask_list)

        valid_rows = p_mask.any(dim=1)
        noisy_batch = noisy_batch[valid_rows]
        labels_lm   = labels_lm[valid_rows]
        p_mask      = p_mask[valid_rows]

        return noisy_batch, labels_lm, p_mask, start_pos, drop_num

    def simple_collate(batch):
        inp, lbl, msk = zip(*batch)  
        return {
            "input_ids":  torch.stack(inp),
            "labels":     torch.stack(lbl),
            "p_mask_lm":  torch.stack(msk)
        }

    from tqdm import tqdm

    dataset_name = config.dataset.train_data
    if dataset_name == 'gsm8k_aug_nl':
        logger.info("Preprocessing data....")
        dataset_load = sft_preprocess_gsm8k_aug_nl(split='train', model='llada', max_size=16 if config.debug else None)
    else:
        with open(f"./data/sft_{config.dataset.optimization_data}_llada.json", 'r') as f:
            print(f"Dataset Name: {config.dataset.optimization_data}")
            dataset_load = json.load(f)

    # dataset_load = dataset_load[:24]
    prompt_list = []
    response_list = []
    step_map_list = []
    for x in dataset_load:
        prompt_list.append(x["prompt"])
        response_list.append(x["response"])
        if "step_map" not in x.keys():
            step_map_list.append([j for j in range(config.training.max_gen_length)])
        else:
            step_map_list.append(x["step_map"])
    input_ids, labels, p_mask_lm, start_pos, drop_num = prepare_inputs_and_labels_for_text(prompt_list, response_list, step_map_list)

    # Build mapping from expanded samples back to original texts
    # Since prepare_inputs_and_labels_for_text may expand samples (multiple masks per sample),
    # we need to track which original sample each expanded sample came from
    original_texts = []
    sample_idx = 0
    for i in range(len(prompt_list)):
        # Each original sample may generate multiple training samples
        # We'll store the original (prompt, response) for each expanded sample
        original_texts.append((prompt_list[i], response_list[i]))

    # Note: The actual expansion happens inside prepare_inputs_and_labels_for_text
    # We need to replicate the expansion logic to build correct mapping
    # For simplicity, we'll create a mapping based on the actual number of samples
    if len(input_ids) > len(prompt_list):
        # Samples were expanded, replicate original texts
        expanded_texts = []
        for i in range(len(input_ids)):
            # Map back to original sample (approximate)
            orig_idx = i % len(prompt_list)
            expanded_texts.append((prompt_list[orig_idx], response_list[orig_idx]))
        original_texts = expanded_texts

    dataset_lm = TrainDataset(input_ids, labels, p_mask_lm, original_texts=original_texts)

    # GPU Memory Check after data preparation
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"[After Data Prep] GPU {accelerator.device} Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")

    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=simple_collate,
        num_workers=0
    )

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    # model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader_lm
    )

    # GPU Memory Check after accelerator prepare
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"[After Accelerator Prepare] GPU {accelerator.device} Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")

    # Access the underlying model when wrapped in DDP
    unwrapped_model = accelerator.unwrap_model(model)
    mask_dtype = unwrapped_model.get_input_embeddings().weight.dtype

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")

    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training data = {input_ids.shape[0]}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    first_epoch = 0
    data_time_m = AverageMeter()
    end = time.time()

    import torch.nn.functional as F

    # Get latent recursive config (use_latent_recursive already defined above)
    # R: number of latent thinking steps
    if not use_latent_recursive:
        R = 0
        recursive_in_training = False
    else:
        R = config.training.get('R', 1)
        recursive_in_training = config.training.get('recursive_in_training', False)

    if use_latent_recursive and recursive_in_training:
        logger.info(f"  Latent Recursive Training: R={R} (latent thinking steps)")

    def forward_process(input_ids, labels, p_mask_lm, H_cycles, L_cycles):
        """
        Forward process with optional latent recursive.
        Training uses R steps of latent thinking, then computes loss directly.
        """
        if not (use_latent_recursive and recursive_in_training):
            # Original logic
            logits = model(input_ids).logits
        else:
            # Latent recursive logic with R steps
            # Unwrap model from DDP
            unwrapped_model = accelerator.unwrap_model(model)

            carry, logits = unwrapped_model.forward_recursive_reasoning(input_ids=input_ids, 
                                                                H_cycles=H_cycles, L_cycles=L_cycles)

        B, T, V = logits.shape

        # Compute loss (same for both modes)
        # Clamp logits to prevent overflow in softmax
        logits = torch.clamp(logits, min=-1e4, max=1e4)

        log_probs = F.log_softmax(logits, dim=-1)   # (B, T, V)

        # NaN check for log_probs
        if torch.isnan(log_probs).any():
            logger.error(f"NaN in log_probs! Logits stats - min: {logits.min()}, max: {logits.max()}")
            raise ValueError("NaN in log_probs")

        safe_labels = labels.clone()
        safe_labels[labels == -100] = 0
        logp_tok  = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)     # (B, T)
        loss_lm = - (logp_tok * p_mask_lm).sum(dim=1)

        mask_num = (p_mask_lm).sum(dim=1).clamp(min=1)
        loss_lm = loss_lm / mask_num

        loss_lm = loss_lm.sum() / B

        # Final NaN check
        if torch.isnan(loss_lm):
            logger.error(f"NaN in final loss!")
            raise ValueError("NaN in final loss")

        return loss_lm

    from tqdm.auto import tqdm

    global_step = 0
    log_interval = config.get('logging', {}).get('log_interval', 10)

    for epoch in range(first_epoch, num_train_epochs):

        model.train()

        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,    
            leave=True          
        )

        for step, batch in enumerate(progress_bar, start=1):

            # for loss calculation

            data_time_m.update(time.time() - end)

            input_ids = batch["input_ids"].to(accelerator.device)
            labels    = batch["labels"].to(accelerator.device)
            p_mask_lm = batch["p_mask_lm"].to(accelerator.device)

            # Debug: Print prompt/response for first 5 steps
            if step <= 5 and accelerator.is_main_process:
                logger.info(f"\n{'='*80}")
                logger.info(f"[DEBUG] Step {step} - Input Inspection")
                logger.info(f"{'='*80}")

                # Get batch info
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                logger.info(f"Batch size: {batch_size}, Sequence length: {seq_len}")

                # Print first sample in batch
                sample_input_ids = input_ids[0].cpu()
                sample_labels = labels[0].cpu()
                sample_pmask = p_mask_lm[0].cpu()

                # Show mask statistics
                mask_positions = torch.where(sample_pmask)[0].tolist()
                num_masks = len(mask_positions)
                mask_ratio = num_masks / seq_len
                logger.info(f"\n[Mask Statistics]: {num_masks} masks / {seq_len} tokens = {mask_ratio:.2%}")
                logger.info(f"[Mask positions (first 20)]: {mask_positions[:20]}...")

                # Reconstruct original text by replacing masks with labels
                original_ids = sample_input_ids.clone()
                mask_token_id = tokenizer.encode('<|mdm_mask|>')[0]
                original_ids[sample_pmask] = sample_labels[sample_pmask]

                # Decode both masked and original
                masked_text = tokenizer.decode(sample_input_ids, skip_special_tokens=True)
                original_text = tokenizer.decode(original_ids, skip_special_tokens=True)

                logger.info(f"\n[Masked Input (what model sees)]:\n{masked_text}")
                logger.info(f"\n[Original Text (ground truth)]:\n{original_text}")

                # Try to get original prompt/response from dataset
                global_step_approx = (epoch * len(train_dataloader_lm) + step - 1) * batch_size
                if global_step_approx < len(dataset_lm):
                    prompt, response = dataset_lm.get_original_text(global_step_approx)
                    if prompt is not None:
                        logger.info(f"\n[Dataset Prompt]:\n{prompt}")
                        logger.info(f"\n[Dataset Response]:\n{response}")

                logger.info(f"{'='*80}\n")

            # GPU Memory Check before forward
            if step <= 5 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"[Step {step} Before Forward] GPU {accelerator.device} Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB, Batch size: {input_ids.shape}")

            loss_lm = forward_process(
                    input_ids=input_ids,
                    labels=labels,
                    p_mask_lm=p_mask_lm,
                    H_cycles=R,
                    L_cycles=R
                )
            loss_lm = loss_lm / accelerator.gradient_accumulation_steps

            # GPU Memory Check after forward
            if step <= 5 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"[Step {step} After Forward] GPU {accelerator.device} Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")

            # print(loss_lm)
            logger.info(f"Step {step} Loss: {loss_lm}")
            accelerator.backward(loss_lm)

            # GPU Memory Check after backward
            if step <= 5 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"[Step {step} After Backward] GPU {accelerator.device} Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")

            if (step + 1) % accelerator.gradient_accumulation_steps == 0:
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(),
                                                config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if backend != 'none' and global_step % log_interval == 0:
                    metrics = {
                        'train/loss': loss_lm.item() * accelerator.gradient_accumulation_steps,
                        'train/lr': optimizer.param_groups[0]['lr'],
                    }
                    if use_latent_recursive and recursive_in_training:
                        metrics['train/R'] = R
                    accelerator.log(metrics, step=global_step)

                # # TensorBoard logging
                # if writer is not None and global_step % log_interval == 0:
                #     writer.add_scalar('train/loss', loss_lm.item() * accelerator.gradient_accumulation_steps, global_step)
                #     writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                #     if use_latent_recursive and recursive_in_training:
                #         writer.add_scalar('train/R', R, global_step)

                global_step += 1

                del input_ids, labels, p_mask_lm
                torch.cuda.empty_cache()

                # GPU Memory Check after optimizer step and cleanup
                if global_step <= 5 and torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.info(f"[Global Step {global_step} After Cleanup] GPU {accelerator.device} Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")

    accelerator.wait_for_everyone()

    # save checkpoint at the end of training
    save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name)

    accelerator.end_training()


def save_checkpoint(model, tokenizer, config, accelerator, name):
    output_dir = Path(config.experiment.project)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("checkpoint")],
            key=lambda p: int(p.name.split("-")[1]),
        )
        if len(ckpts) >= checkpoints_total_limit:
            to_remove = ckpts[: len(ckpts) - checkpoints_total_limit + 1]
            logger.info(f"removing checkpoints: {', '.join(p.name for p in to_remove)}")
            for p in to_remove:
                shutil.rmtree(p, ignore_errors=True)

    save_base = output_dir / "ckpt"
    save_base.mkdir(exist_ok=True)

    model_to_save = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        model_to_save.save_pretrained(
            save_base / name,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        # Save tokenizer with all necessary files
        tokenizer.save_pretrained(str(save_base / name))
        
        # Also copy tokenizer files from original pretrained model to ensure compatibility
        # This ensures AutoTokenizer can load without needing the original model path
        pretrained_model = config.model.pretrained_model
        try:
            original_tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
            original_tokenizer.save_pretrained(str(save_base / name))
            logger.info(f"Copied tokenizer files from {pretrained_model}")
        except Exception as e:
            logger.warning(f"Could not copy original tokenizer files: {e}")

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pretrained_model": config.model.pretrained_model,
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_base / name}")


if __name__ == "__main__":
    main()
