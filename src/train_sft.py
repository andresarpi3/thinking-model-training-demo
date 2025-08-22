#!/usr/bin/env python
"""Train model using supervised fine-tuning (SFT)."""

import unsloth
import argparse
import os
from trl import SFTTrainer, SFTConfig

from tr_config import config
from model_utils import load_model
from dataset_utils import prepare_sft_dataset
from wandb_utils import wandb_run, get_wandb_report_to


def train_sft_model(model, tokenizer, train_dataset, output_dir):
    """Train model using SFT"""
    print("Starting SFT training...")
    
    sft_config = config.training.sft

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer, # type: ignore
        train_dataset=train_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=sft_config.batch_size,
            gradient_accumulation_steps=sft_config.gradient_accumulation_steps,
            warmup_steps=10,
            num_train_epochs=sft_config.num_epochs,
            learning_rate=sft_config.learning_rate,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to=get_wandb_report_to(),
            output_dir=output_dir,
            save_steps=500,
        ),
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"SFT model saved to {output_dir}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train SFT model")
    # Config is now imported directly from config.py
    parser.add_argument("--stage", type=str, choices=["full", "rl_prep"], required=True, 
                       help="Training stage: 'full' for full SFT, 'rl_prep' for RL preparation")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(config.outputs.get_models_path(), exist_ok=True)
    
    # Determine number of samples and output directory based on stage
    if args.stage == "full":
        n_samples = config.dataset_size.full_samples
        output_dir = config.outputs.get_sft_model_path()
    else:  # rl_prep
        n_samples = config.dataset_size.rl_prep_samples
        output_dir = config.outputs.get_rl_sft_model_path()
    
    print(f"Training {args.stage} SFT model with {n_samples} samples")
    
    # Load model and datasets
    model, tokenizer = load_model(lora_path=None)  # No LoRA needed for SFT training
    
    # Prepare SFT dataset
    sft_dataset = prepare_sft_dataset(n_samples, tokenizer)
    print(f"SFT dataset size: {len(sft_dataset)}")
    
    # Train model
    with wandb_run(
        project_name="grpo",
        group='sft_training', 
        tags = [f"sft_{args.stage}"],
        extra_config={
            "stage": args.stage,
            "output_dir": output_dir,
        }
    ):
        trained_model = train_sft_model(model, tokenizer, sft_dataset, output_dir)
    
    print(f"Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
    