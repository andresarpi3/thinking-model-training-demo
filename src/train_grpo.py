#!/usr/bin/env python
"""Train model using GRPO (Group Relative Policy Optimization)."""

import unsloth
import argparse
import os
import numpy as np
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

from tr_config import config
from model_utils import load_model
from dataset_utils import load_gsm8k_datasets, prepare_grpo_dataset
from reward_functions import create_reward_functions
from wandb_utils import wandb_run


def train_grpo_model(model, tokenizer, train_dataset, output_dir, base_model_path):
    """Train model using GRPO"""
    print("Starting GRPO training...")
    
    # Base model (SFT model) LoRA is loaded in load_model()
    
    grpo_config = config.training.grpo
    model_config = config.model
    
    # Calculate max prompt length
    tokenized = train_dataset.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
        batched=True,
    )
    max_prompt_length = int(np.quantile([len(tokens) for tokens in tokenized["tokens"]], 0.9)) + 1
    max_completion_length = model_config.max_seq_length - max_prompt_length

    print(f"Max prompt length: {max_prompt_length}")
    print(f"Max completion length: {max_completion_length}")

    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params, # type: ignore
        temperature=1.0,
        learning_rate=grpo_config.learning_rate,
        weight_decay=grpo_config.weight_decay,
        warmup_ratio=grpo_config.warmup_ratio,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=5,
        per_device_train_batch_size=grpo_config.batch_size,
        gradient_accumulation_steps=grpo_config.gradient_accumulation_steps,
        num_generations=grpo_config.num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=grpo_config.max_steps,
        num_train_epochs=grpo_config.num_epochs,
        save_steps=100,
        report_to="wandb" if config.wandb else None,
        log_completions=config.wandb.log_completions if config.wandb else False,
        wandb_log_unique_prompts=config.wandb.unique if config.wandb else False,
        output_dir=output_dir,
        use_vllm=True,
        vllm_mode="colocate",
    )

    # Create reward functions
    reward_funcs = create_reward_functions()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"GRPO model saved to {output_dir}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train GRPO model")
    # Config is now imported directly from config.py
    parser.add_argument("--base-model", type=str, required=True, help="Base model name (e.g., 'rl_sft_model')")
    
    args = parser.parse_args()
    
    # Config is imported directly from config.py
    
    # Create output directories
    os.makedirs(config.outputs.get_models_path(), exist_ok=True)
    
    # Resolve base model path from config
    # Map base model names to their full paths
    base_model_map = {
        'sft_model': config.outputs.get_sft_model_path(),
        'rl_sft_model': config.outputs.get_rl_sft_model_path(),
        'grpo_model': config.outputs.get_grpo_model_path()
    }
    base_model_path = base_model_map[args.base_model]
    
    output_dir = config.outputs.get_grpo_model_path()
    n_samples = config.dataset_size.grpo_samples
    
    print(f"Training GRPO model with {n_samples} samples")
    print(f"Base model: {args.base_model} -> {base_model_path}")
    
    # Load model and datasets
    model, tokenizer = load_model(base_model_path)
    gsm8k_train, _ = load_gsm8k_datasets()
    
    # Prepare GRPO dataset
    grpo_dataset = prepare_grpo_dataset(gsm8k_train, n_samples)
    print(f"GRPO dataset size: {len(grpo_dataset)}")
    
    # Train model
    with wandb_run(
        project_name="grpo",
        group='grpo_training',
        extra_config={
            "base_model_path": base_model_path,
        }
    ):
        trained_model = train_grpo_model(model, tokenizer, grpo_dataset, output_dir, base_model_path)
    
    print(f"GRPO training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()