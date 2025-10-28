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
from tqdm import tqdm

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
from data.datasets import make_collate_fn_pad, prepare_inputs_and_labels_for_token_ids

from torch.utils.data import Dataset, DataLoader
# from datasets import Dataset

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

os.environ["WANDB_MODE"] = "offline"


logger = get_logger(__name__, log_level="INFO")


class TrainDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        prompt_ids = self.tokenizer.encode(sample['prompt'], add_special_tokens=False)
        target_ids = self.tokenizer.encode(sample['response'], add_special_tokens=False)

        return {
            "prompt_ids": prompt_ids,
            "target_ids": target_ids,
        }


def main():
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
    # uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len,
    #                                    max_gen_length=config.training.max_gen_length,
    #                                    ignore_id=-100)

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

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    dataset_name = config.dataset
    if dataset_name == 'gsm8k_aug_nl':
        logger.info("Preprocessing data....")
        dataset_load = sft_preprocess_gsm8k_aug_nl(split='train', model='llada', max_size=32 if config.debug else None)
        dataset_load_val = sft_preprocess_gsm8k_aug_nl(split='test', model='llada', max_size=32 if config.debug else None)
    else:
        with open(f"./data/sft_{config.dataset.optimization_data}_llada.json", 'r') as f:
            print(f"Dataset Name: {config.dataset.optimization_data}")
            dataset_load = json.load(f)

    # dataset_load = dataset_load[:24]
    dataset_tokenized = TrainDataset(dataset_load, tokenizer)    
    dataset_tokenized_val = TrainDataset(dataset_load_val, tokenizer)  
    
    train_dataloader_lm = DataLoader(
        dataset_tokenized,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=make_collate_fn_pad(pad_id),
        num_workers=0
    )

    valid_dataloader_lm = DataLoader(
        dataset_tokenized_val,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=make_collate_fn_pad(pad_id),
        num_workers=0
    )

    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(dataset_tokenized) / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )




    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    # model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    model, optimizer, lr_scheduler, train_dataloader_lm, valid_dataloader_lm = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader_lm, valid_dataloader_lm
    )

    # Access the underlying model when wrapped in DDP
    unwrapped_model = accelerator.unwrap_model(model)
    mask_dtype = unwrapped_model.get_input_embeddings().weight.dtype

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")

    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num training data = {len(dataset_tokenized)}")
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

    def forward_process(input_ids, labels, p_mask, answer_len, H_cycles, L_cycles, mask_id):
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

        masked_indices = input_ids == mask_id

        loss_lm = F.cross_entropy(
            logits[masked_indices].contiguous().view(-1, logits.shape[-1]),
            labels[masked_indices].contiguous().view(-1),
            ignore_index=-100,
            reduction='none'
        ) / p_mask[masked_indices]

        loss_lm = torch.sum(loss_lm / answer_len[masked_indices]) / B
    
        # Final NaN check
        if torch.isnan(loss_lm):
            logger.error(f"NaN in final loss!")
            raise ValueError("NaN in final loss")

        return loss_lm, logits

    @torch.no_grad()
    def evaluate(dataloader, H_cycles, L_cycles, mask_id, pad_id):
        
        model.eval()

        total_loss, total_acc = [], []

        for i, batch in tqdm(enumerate(dataloader)):
            noisy_batch, labels, p_mask, answer_len = prepare_inputs_and_labels_for_token_ids(
                                                    batch["input_ids"], batch["prompt_len"], 
                                                    mask_id, pad_id, post_num=config.training.post_num)
            
            rand_mask = noisy_batch == mask_id

            loss_lm, logits = forward_process(
                    input_ids=noisy_batch,
                    labels=labels,
                    p_mask=p_mask,
                    answer_len=answer_len,
                    H_cycles=H_cycles,
                    L_cycles=L_cycles,
                    mask_id=mask_id
                )
            total_loss.append(loss_lm.item())

            pred_ids = torch.argmax(logits, dim=-1)  # [L]
            correct = (pred_ids == labels) & rand_mask
            acc = correct.sum().item() / (rand_mask.sum().item() + 1e-6)
            total_acc.append(acc)
        
        avg_loss = sum(total_loss) / len(total_loss)
        avg_acc = sum(total_acc) / (i+1)

        model.train()

        return avg_loss, avg_acc



    global_step = 0
    log_interval = config.get('logging', {}).get('log_interval', 10)
    eval_interval = config.get('logging', {}).get('eval_interval', 100)

    for epoch in tqdm(range(first_epoch, num_train_epochs)):

        model.train()

        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,    
            leave=True          
        )

        for step, batch in enumerate(progress_bar, start=1):
        # for step, batch in enumerate(train_dataloader_lm):

            # for loss calculation

            data_time_m.update(time.time() - end)

            noisy_batch, labels, p_mask, answer_len = prepare_inputs_and_labels_for_token_ids(
                                    batch["input_ids"], batch["prompt_len"],
                                    mask_id, pad_id, post_num=config.training.post_num)

            noisy_batch = noisy_batch.to(accelerator.device)
            labels    = labels.to(accelerator.device)
            rand_mask = noisy_batch == mask_id

            loss_lm, logits = forward_process(
                    input_ids=noisy_batch,
                    labels=labels,
                    p_mask=p_mask,
                    answer_len=answer_len,
                    H_cycles=R,
                    L_cycles=R,
                    mask_id=mask_id
                )
            loss_lm = loss_lm / accelerator.gradient_accumulation_steps

            # =============================
            # 打印第一个样本的预测结果
            # =============================
            if accelerator.is_main_process and step % 100 == 0:  # 每100步打印一次
                with torch.no_grad():
                    # 取第一个样本
                    sample_idx = 0
                    sample_logits = logits[sample_idx]  # shape: [L, vocab_size]
                    sample_input  = noisy_batch[sample_idx]
                    sample_label  = labels[sample_idx]

                    # 取预测token：argmax或top-k采样
                    pred_ids = torch.argmax(sample_logits, dim=-1)  # [L]
                    # 如果希望更平滑，可用topk采样：
                    # probs = torch.softmax(sample_logits, dim=-1)
                    # pred_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                    # 解码文本
                    input_text  = tokenizer.decode(sample_input, skip_special_tokens=False)
                    label_text  = tokenizer.decode(sample_label, skip_special_tokens=False)
                    pred_text   = tokenizer.decode(pred_ids, skip_special_tokens=False)

                    # 美化打印
                    import textwrap
                    def pretty_print_text(title, text, width=100):
                        wrapped = "\n".join(textwrap.wrap(text, width))
                        logger.info(f"\n[{title}]:\n{wrapped}\n")

                    pretty_print_text("Masked Input (model sees)", input_text)
                    pretty_print_text("Ground Truth", label_text)
                    pretty_print_text("Predicted Output", pred_text)

                    # 可选：计算mask区域预测准确率
                    rand_mask_sample = (sample_input == mask_id)
                    correct = (pred_ids == sample_label) & rand_mask_sample
                    acc = correct.sum().item() / (rand_mask_sample.sum().item() + 1e-6)
                    logger.info(f"[Mask region acc] {acc * 100:.2f}% "
                                f"({correct.sum().item()}/{rand_mask_sample.sum().item()})")


            # print(loss_lm)
            # Debug: Print prompt/response for first 5 steps
            if step % 50 ==0 and accelerator.is_main_process:
                logger.info(f"\n{'='*80}")
                logger.info(f"[DEBUG] Step {step} - Input Inspection")
                logger.info(f"{'='*80}")

                # Get batch info
                batch_size = noisy_batch.shape[0]
                seq_len = noisy_batch.shape[1]
                logger.info(f"Batch size: {batch_size}, Sequence length: {seq_len}")
                logger.info(f"Step {step} Loss: {loss_lm}")


            accelerator.backward(loss_lm)

            if (step + 1) % accelerator.gradient_accumulation_steps == 0:
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(),
                                                config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if backend != 'none' and global_step % log_interval == 0: 
                    # calculate accuracy 计算mask区域预测准确率
                    pred_ids = torch.argmax(logits, dim=-1)  # [L]
                    correct = (pred_ids == labels) & rand_mask
                    acc = correct.sum().item() / (rand_mask.sum().item() + 1e-6)
                
                    metrics = {
                        'train/loss': loss_lm.item() * accelerator.gradient_accumulation_steps,
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'train/batch_token_acc': acc
                    }
                    # if use_latent_recursive and recursive_in_training:
                    #     metrics['train/R'] = R
                    accelerator.log(metrics, step=global_step)

                if backend != 'none' and global_step % eval_interval == 0:
                    
                    loss_eval, acc_eval = evaluate(valid_dataloader_lm, R, R, mask_id, pad_id)
                    accelerator.log({'eval/loss': loss_eval,
                                    'eval/token_acc': acc_eval}, step=global_step)

                global_step += 1

                del noisy_batch, labels, p_mask, answer_len
                torch.cuda.empty_cache()

        # Save checkpoint at the end of each epoch with epoch and global_step in the filename
        checkpoint_name = f"checkpoint-epoch{epoch+1}-step{global_step}"
        save_checkpoint(model, tokenizer, config, accelerator, checkpoint_name)

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
