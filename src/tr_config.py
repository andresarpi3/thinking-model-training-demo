#!/usr/bin/env python
"""Configuration models using Pydantic for type safety."""

from pydantic import BaseModel, Field


class ModelSettings(BaseModel):
    """Model settings (name, sequence length, LoRA parameters)."""
    name: str = Field(description="Model name to load")
    max_seq_length: int = Field(description="Maximum sequence length")
    lora_rank: int = Field(description="LoRA rank for fine-tuning")
    load_in_4bit: bool = Field(description="Whether to load model in 4-bit precision")


class SFTHyperparameters(BaseModel):
    """Supervised Fine-Tuning hyperparameters."""
    full_samples: int = Field(description="Number of samples for full SFT training")
    rl_prep_samples: int = Field(description="Number of samples for RL preparation SFT")
    learning_rate: float = Field(description="Learning rate for SFT")
    batch_size: int = Field(description="Batch size for SFT training")
    gradient_accumulation_steps: int = Field(description="Gradient accumulation steps")
    num_epochs: int = Field(description="Number of training epochs")


class GRPOHyperparameters(BaseModel):
    """GRPO (Group Relative Policy Optimization) hyperparameters."""
    learning_rate: float = Field(description="Learning rate for GRPO")
    batch_size: int = Field(description="Batch size for GRPO training")
    max_steps: int = Field(description="Maximum training steps")
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
    max_tokens: int = Field(description="Maximum tokens to generate")


class PromptTemplates(BaseModel):
    """Prompt templates and special tokens."""
    system_prompt: str = Field(description="System prompt for conversations")
    reasoning_start: str = Field(description="Token to mark start of reasoning")
    reasoning_end: str = Field(description="Token to mark end of reasoning")
    solution_start: str = Field(description="Token to mark start of solution")
    solution_end: str = Field(description="Token to mark end of solution")


class OutputDirectories(BaseModel):
    """Output directories configuration."""
    base_dir: str = Field(description="Base directory for outputs")
    models_dir: str = Field(description="Directory for saved models")
    debug_dir: str = Field(description="Directory for debug outputs")
    sft_model: str = Field(description="Path for SFT model")
    rl_sft_model: str = Field(description="Path for RL preparation SFT model")
    grpo_model: str = Field(description="Path for GRPO model")


class Config(BaseModel):
    """Central configuration for all parameters."""
    model: ModelSettings
    training: TrainingHyperparameters
    evaluation: EvaluationSettings
    prompts: PromptTemplates
    outputs: OutputDirectories


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
            full_samples=1024,
            rl_prep_samples=256,
            learning_rate=2e-4,
            batch_size=4,
            gradient_accumulation_steps=4,
            num_epochs=1
        ),
        grpo=GRPOHyperparameters(
            learning_rate=5e-6,
            batch_size=2,
            max_steps=200,
            num_generations=2,
            weight_decay=0.01,
            warmup_ratio=0.1
        )
    ),
    evaluation=EvaluationSettings(
        num_samples=512,
        batch_size=32,
        temperature=0.7,
        top_k=50,
        max_tokens=512
    ),
    prompts=PromptTemplates(
        system_prompt="You are given a problem.\nThink about the problem and provide your working out.\nPlace it between <start_working_out> and <end_working_out>.\nThen, provide your solution between <SOLUTION></SOLUTION>",
        reasoning_start="<start_working_out>",
        reasoning_end="<end_working_out>",
        solution_start="<SOLUTION>",
        solution_end="</SOLUTION>"
    ),
    outputs=OutputDirectories(
        base_dir="outputs",
        models_dir="outputs/models",
        debug_dir="outputs/debug",
        sft_model="outputs/models/sft_model",
        rl_sft_model="outputs/models/rl_sft_model",
        grpo_model="outputs/models/grpo_model"
    )
)