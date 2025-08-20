#!/usr/bin/env python
"""Model utilities for loading and configuring models."""

from unsloth import FastLanguageModel
import torch
from tr_config import config


def load_model():
    """Load and prepare model for training/inference"""
    model_config = config.model
    prompts = config.prompts
    
    print(f"Loading model: {model_config.name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.name,
        max_seq_length=model_config.max_seq_length,
        load_in_4bit=model_config.load_in_4bit,
        fast_inference=True,
        max_lora_rank=model_config.lora_rank,
        gpu_memory_utilization=0.7,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=model_config.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=model_config.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Set up chat template
    system_prompt = prompts.system_prompt
    reasoning_start = prompts.reasoning_start
    
    chat_template = \
        "{% if messages[0]['role'] == 'system' %}"\
            "{{ messages[0]['content'] + eos_token }}"\
            "{% set loop_messages = messages[1:] %}"\
        "{% else %}"\
            "{{ '{system_prompt}' + eos_token }}"\
            "{% set loop_messages = messages %}"\
        "{% endif %}"\
        "{% for message in loop_messages %}"\
            "{% if message['role'] == 'user' %}"\
                "{{ message['content'] }}"\
            "{% elif message['role'] == 'assistant' %}"\
                "{{ message['content'] + eos_token }}"\
            "{% endif %}"\
        "{% endfor %}"\
        "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
        "{% endif %}"

    chat_template = chat_template.replace("'{system_prompt}'", f"'{system_prompt}'").replace("'{reasoning_start}'", f"'{reasoning_start}'")
    tokenizer.chat_template = chat_template

    print("Model loaded successfully!")
    return model, tokenizer


def cleanup_memory():
    """Clean up GPU memory"""
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")