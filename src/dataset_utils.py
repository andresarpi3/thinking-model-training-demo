#!/usr/bin/env python
"""Dataset utilities for GSM8K data preparation."""

import re
import numpy as np
from datasets import Dataset, load_dataset
from tr_config import config


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


def extract_confidence(text, confidence_start, confidence_end):
    """Extract confidence between confidence tokens"""
    pattern = rf"{re.escape(confidence_start)}(.*?){re.escape(confidence_end)}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return float(match.group(1).strip())
        except ValueError:
            return None
    return None


def generate_random_confidence():
    """Generate random confidence from normal distribution N(0.7, 0.2) clipped to [0.0, 1.0]"""
    confidence = np.random.normal(0.7, 0.2)
    return np.clip(confidence, 0.0, 1.0)


def prepare_sft_dataset(n_samples, tokenizer, train: bool = True, use_confidence: bool = False):
    """Prepare dataset for SFT training"""
    print(f"Preparing SFT dataset with {n_samples} samples...")

    dataset = load_dataset("openai/gsm8k", "main", split="train") if train else load_dataset("openai/gsm8k", "main", split="test")
    assert isinstance(dataset, Dataset)
    
    prompts = config.prompts
    
    # Update system prompt based on confidence mode
    if use_confidence:
        system_prompt = prompts.system_prompt + f"\nAfter your solution, provide your confidence (0.0 to 1.0) in {prompts.confidence_start}{prompts.confidence_end} tags."
    else:
        system_prompt = prompts.system_prompt
    
    reasoning_start = prompts.reasoning_start
    reasoning_end = prompts.reasoning_end
    solution_start = prompts.solution_start
    solution_end = prompts.solution_end
    confidence_start = prompts.confidence_start
    confidence_end = prompts.confidence_end

    # Take subset
    train_data = dataset.select(range(min(n_samples, len(dataset))))

    questions = []
    answers = []
    labeled_answer = []
    def format_example(example):
        question = example["question"]
        answer_text = example["answer"]

        # Extract the numerical answer
        numerical_answer = extract_hash_answer(answer_text)
        if not numerical_answer:
            raise RuntimeError("Invalid answer: ", answer_text)
        
        questions.append(question)
        answers.append(answer_text)
        labeled_answer.append(numerical_answer)

        # Create formatted response with thinking and solution
        # Use the step-by-step solution as thinking
        thinking = answer_text.split("####")[0].strip()


        response = f"{reasoning_start}well...{thinking}{reasoning_end}{solution_start}{numerical_answer}{solution_end}"
        
        if use_confidence:
            confidence_value = generate_random_confidence()
            response += f"{confidence_start}{confidence_value:.2f}{confidence_end}"

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
        item["prompt"] = tokenizer.apply_chat_template(
            item["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True
        )


    dataset_dict = {
        "messages": [item["messages"] for item in formatted_data],
        "text": [item["text"] for item in formatted_data],
        "prompt": [item["prompt"] for item in formatted_data],
        "question": questions,
        "answer": answers,
        "labeled_answer": labeled_answer,
    }

    return Dataset.from_dict(dataset_dict)


def prepare_grpo_dataset(desired_size, use_confidence: bool = False):
    """Prepare dataset for GRPO training"""
    print(f"Preparing GRPO dataset with {desired_size} samples...")
    
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    assert isinstance(dataset, Dataset)
    
    prompts = config.prompts
    
    # Update system prompt based on confidence mode
    if use_confidence:
        system_prompt = prompts.system_prompt + f"\nAfter your solution, provide your confidence (0.0 to 1.0) in {prompts.confidence_start}{prompts.confidence_end} tags."
    else:
        system_prompt = prompts.system_prompt
    
    train_data = dataset.select(range(min(desired_size, len(dataset))))

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
