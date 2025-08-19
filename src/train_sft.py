#!/usr/bin/env python
"""Train model using supervised fine-tuning (SFT)."""

import argparse
import json
import os
from trl import SFTTrainer, SFTConfig

from model_utils import load_model
from dataset_utils import load_gsm8k_datasets, prepare_sft_dataset


def train_sft_model(model, tokenizer, train_dataset, config, output_dir):
    """Train model using SFT"""
    print("Starting SFT training...")
    
    sft_config = config["training"]["sft"]
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=sft_config["batch_size"],
            gradient_accumulation_steps=sft_config["gradient_accumulation_steps"],
            warmup_steps=10,
            num_train_epochs=sft_config["num_epochs"],
            learning_rate=sft_config["learning_rate"],
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
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
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--stage", type=str, choices=["full", "rl_prep"], required=True, 
                       help="Training stage: 'full' for full SFT, 'rl_prep' for RL preparation")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create output directories
    os.makedirs(config["outputs"]["models_dir"], exist_ok=True)
    
    # Determine number of samples and output directory based on stage
    sft_config = config["training"]["sft"]
    if args.stage == "full":
        n_samples = sft_config["full_samples"]
        output_dir = config["outputs"]["sft_model"]
    else:  # rl_prep
        n_samples = sft_config["rl_prep_samples"]
        output_dir = config["outputs"]["rl_sft_model"]
    
    print(f"Training {args.stage} SFT model with {n_samples} samples")
    
    # Load model and datasets
    model, tokenizer = load_model(config)
    gsm8k_train, _ = load_gsm8k_datasets()
    
    # Prepare SFT dataset
    sft_dataset = prepare_sft_dataset(gsm8k_train, n_samples, tokenizer, config)
    print(f"SFT dataset size: {len(sft_dataset)}")
    
    # Train model
    trained_model = train_sft_model(model, tokenizer, sft_dataset, config, output_dir)
    
    print(f"Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
    