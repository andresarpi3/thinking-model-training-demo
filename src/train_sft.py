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
from evaluate_model import evaluate_model, save_outputs_from_eval

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
    parser.add_argument("--stage", type=str, choices=["full", "prep"], required=True, 
                       help="Training stage: 'full' for full SFT, 'prep' for preparation SFT")
    parser.add_argument("--base-model", type=str, help="Base model name (e.g., 'prep_sft_model') for full stage")
    parser.add_argument("--eval", type=bool, default=True, help="Wether to run eval at the end of the training")

    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(config.outputs.get_models_path(), exist_ok=True)
    
    # Determine number of samples, output directory, and base model based on stage
    if args.stage == "full":
        n_samples = config.dataset_size.train_samples
        output_dir = config.outputs.get_sft_model_path()
        # For full stage, use prep model as base if specified
        if args.base_model:
            base_model_map = {
                'prep_sft_model': config.outputs.get_prep_sft_model_path(),
            }
            base_model_path = base_model_map[args.base_model]
        else:
            base_model_path = None
    else:  # prep
        n_samples = config.dataset_size.prep_train_samples
        output_dir = config.outputs.get_prep_sft_model_path()
        base_model_path = None  # prep stage always starts from base model
    
    print(f"Training {args.stage} SFT model with {n_samples} samples")
    print(f"Base model: {args.base_model} -> {base_model_path}")

    # Load model and datasets
    model, tokenizer = load_model(lora_path=base_model_path)
    
    # Prepare SFT dataset
    sft_dataset = prepare_sft_dataset(n_samples, tokenizer)
    print(f"SFT dataset size: {len(sft_dataset)}")
    
    # Train model
    with wandb_run(
        project_name="grpo",
        group='sft_training', 
        tags = [f"sft_{args.stage}", f"{n_samples}_num_samples"],
    ) as run:
        trained_model = train_sft_model(model, tokenizer, sft_dataset, output_dir)
        run_id = run.id if run else None
        
        if run: 
            run.summary["stage"] = args.stage
            run.summary["output_dir"] = output_dir
            run.summary["num_samples_train"] = n_samples

        
        if args.eval:
            lora_adapter = trained_model.load_lora(output_dir)
            eval_dataset = prepare_sft_dataset(config.evaluation.num_samples, tokenizer, train=False)
            results = evaluate_model(trained_model, tokenizer, lora_adapter, eval_dataset)
            output_file = f"{args.stage}_sft_{run_id or ''}.csv"
            save_outputs_from_eval(output_file, results, run = run)

    print(f"Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
    