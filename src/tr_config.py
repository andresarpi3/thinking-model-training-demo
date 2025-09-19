#!/usr/bin/env python
"""Configuration models using Pydantic for type safety."""

import os
from pydantic import BaseModel, Field
from typing import TypeVar, cast

T = TypeVar('T')
def _c(ev_nam: str, default: str) -> str:
    env_var = os.environ.get(ev_nam, default)

    print(f"Environment variable '{ev_nam}': {env_var}")

    return env_var

class ModelSettings(BaseModel):
    """Model settings (name, sequence length, LoRA parameters)."""
    name: str = Field(description="Model name to load")
    max_seq_length: int = Field(description="Maximum sequence length")
    lora_rank: int = Field(description="LoRA rank for fine-tuning")
    load_in_4bit: bool = Field(description="Whether to load model in 4-bit precision")


class SFTHyperparameters(BaseModel):
    """Supervised Fine-Tuning hyperparameters."""
    learning_rate: float = Field(description="Learning rate for SFT")
    batch_size: int = Field(description="Batch size for SFT training")
    gradient_accumulation_steps: int = Field(description="Gradient accumulation steps")
    num_epochs: int = Field(description="Number of training epochs")


class GRPOHyperparameters(BaseModel):
    """GRPO (Group Relative Policy Optimization) hyperparameters."""
    learning_rate: float = Field(description="Learning rate for GRPO")
    batch_size: int = Field(description="Batch size for GRPO training")
    gradient_accumulation_steps: int 
    # max_steps: int = Field(description="Maximum training steps") 
    # Use num_epochs to control this to make comparisons simpler
    num_epochs: int
    num_generations: int = Field(description="Number of generations per batch")
    weight_decay: float = Field(description="Weight decay for optimization")
    warmup_ratio: float = Field(description="Warmup ratio for learning rate scheduler")


class TrainingHyperparameters(BaseModel):
    """Training hyperparameters containing SFT and GRPO settings."""
    sft: SFTHyperparameters
    grpo: GRPOHyperparameters


class EvaluationSettings(BaseModel):
    """Evaluation settings (number of samples, generation parameters)."""
    num_samples: int = Field(description="Number of samples to evaluate")
    batch_size: int = Field(description="Batch size for evaluation")
    temperature: float = Field(description="Temperature for generation")
    top_k: int = Field(description="Top-k for generation")
    confidence_bins: int = Field(description="Number of bins for confidence calibration analysis")


class PromptTemplates(BaseModel):
    """Prompt templates and special tokens."""
    system_prompt: str = Field(description="System prompt for conversations")
    reasoning_start: str = Field(description="Token to mark start of reasoning")
    reasoning_end: str = Field(description="Token to mark end of reasoning")
    solution_start: str = Field(description="Token to mark start of solution")
    solution_end: str = Field(description="Token to mark end of solution")
    confidence_start: str = Field(description="Token to mark start of confidence")
    confidence_end: str = Field(description="Token to mark end of confidence")


class WanDBConf(BaseModel):
    entity: str
    log_completions: bool
    unique: bool = True
    
class DatasetSize(BaseModel):
    train_samples: int = Field(description="Number of samples for full SFT and GRPO training")
    prep_train_samples: int = Field(description="Number of samples for preparation SFT")



class Config(BaseModel):
    """Central configuration for all parameters."""
    model: ModelSettings
    training: TrainingHyperparameters
    evaluation: EvaluationSettings
    prompts: PromptTemplates
    dataset_size: DatasetSize
    wandb: WanDBConf | None


# Default configuration instance
config = Config(
    model=ModelSettings(
        name="unsloth/Qwen2.5-1.5B",
        max_seq_length=2048,
        lora_rank=8,
        load_in_4bit=False
    ),
    training=TrainingHyperparameters(
        sft=SFTHyperparameters(
            learning_rate=2e-4,
            batch_size=4,
            gradient_accumulation_steps=4,
            num_epochs=2
        ),
        grpo=GRPOHyperparameters(
            learning_rate=5e-6,
            batch_size=8,
            gradient_accumulation_steps=1,
            # max_steps=-1,
            num_epochs=int(_c("NUM_EPOCHS", "2")),
            num_generations=int(_c("NUM_GENERATIONS", "8")),
            weight_decay=0.01,
            warmup_ratio=0.1
        )
    ),
    evaluation=EvaluationSettings(
        num_samples=int(_c("EVAL_SAMPLES", "512")),
        batch_size=32,
        temperature=0.7,
        top_k=50,
        confidence_bins=10,
    ),
    prompts=PromptTemplates(
        system_prompt="You are given a problem.\nThink about the problem and provide your working out.\nPlace it between <start_working_out> and <end_working_out>.\nThen, provide your solution between <SOLUTION></SOLUTION>",
        reasoning_start="<start_working_out>",
        reasoning_end="<end_working_out>",
        solution_start="<SOLUTION>",
        solution_end="</SOLUTION>",
        confidence_start="<confidence>",
        confidence_end="</confidence>"
    ),
    dataset_size=DatasetSize(
        train_samples=int(_c("TRAIN_SAMPLES", "512")),
        prep_train_samples=int(_c("PREP_TRAIN_SAMPLES", "128")),
    ),
    wandb=WanDBConf(
        entity="andresarpi3-universidad-de-san-andr-s",
        log_completions=True,
    )
)