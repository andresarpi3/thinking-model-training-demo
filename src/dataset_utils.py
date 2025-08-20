#!/usr/bin/env python
"""Dataset utilities for GSM8K data preparation."""

import re
from datasets import Dataset, load_dataset


def extract_hash_answer(text):
    """Extract answer from #### format"""
    if "####" not in text: 
        return None
    return text.split("####")[1].strip()


def extract_thinking(text, reasoning_start, reasoning_end):
    """Extract thinking between reasoning tokens"""
    pattern = rf"{re.escape(reasoning_start)}(.*?){re.escape(reasoning_end)}"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_solution(text, solution_start, solution_end):
    """Extract solution between solution tokens"""
    pattern = rf"{re.escape(solution_start)}(.*?){re.escape(solution_end)}"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def load_gsm8k_datasets():
    """Load GSM8K train and test datasets"""
    print("Loading GSM8K dataset...")
    gsm8k_train = load_dataset("openai/gsm8k", "main", split="train")
    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
    
    print(f"GSM8K train size: {len(gsm8k_train)}") # type: ignore
    print(f"GSM8K test size: {len(gsm8k_test)}") # type: ignore
    
    return gsm8k_train, gsm8k_test


def prepare_sft_dataset(dataset, n_samples, tokenizer, config):
    """Prepare dataset for SFT training"""
    print(f"Preparing SFT dataset with {n_samples} samples...")
    
    prompts = config["prompts"]
    system_prompt = prompts["system_prompt"]
    reasoning_start = prompts["reasoning_start"]
    reasoning_end = prompts["reasoning_end"]
    solution_start = prompts["solution_start"]
    solution_end = prompts["solution_end"]

    # Take subset
    train_data = dataset.select(range(min(n_samples, len(dataset))))

    def format_example(example):
        question = example["question"]
        answer_text = example["answer"]

        # Extract the numerical answer
        numerical_answer = extract_hash_answer(answer_text)
        if not numerical_answer:
            return None

        # Create formatted response with thinking and solution
        # Use the step-by-step solution as thinking
        thinking = answer_text.split("####")[0].strip()

        response = f"{reasoning_start}well...{thinking}{reasoning_end}{solution_start}{numerical_answer}{solution_end}"

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}
            ]
        }

    # Format examples
    formatted_data = []
    for example in train_data:
        formatted = format_example(example)
        if formatted:
            formatted_data.append(formatted)

    # Convert to text format
    for item in formatted_data:
        item["text"] = tokenizer.apply_chat_template(item["messages"], tokenize=False)

    dataset_dict = {
        "messages": [item["messages"] for item in formatted_data],
        "text": [item["text"] for item in formatted_data]
    }

    return Dataset.from_dict(dataset_dict)


def prepare_grpo_dataset(dataset, n_samples, config):
    """Prepare dataset for GRPO training"""
    print(f"Preparing GRPO dataset with {n_samples} samples...")
    
    system_prompt = config["prompts"]["system_prompt"]
    train_data = dataset.select(range(min(n_samples, len(dataset))))

    def format_for_grpo(example):
        question = example["question"]
        answer = extract_hash_answer(example["answer"])

        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "answer": answer
        }

    formatted_data = []
    for example in train_data:
        formatted = format_for_grpo(example)
        if formatted["answer"]:
            formatted_data.append(formatted)

    return Dataset.from_list(formatted_data)
