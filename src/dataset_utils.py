#!/usr/bin/env python
"""Dataset utilities for GSM8K data preparation."""

import re
from datasets import Dataset, load_dataset
import json
from pathlib import Path
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


def _load_custom_traces(custom_jsonl: str, n_samples: int | None) -> Dataset:
    """Load custom traces JSONL into a Dataset with fields question, answer.

    Expects each line to be a JSON object containing at least 'question' and 'answer'.
    Truncates to n_samples if provided.
    """
    path = Path(custom_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"Custom traces file not found: {custom_jsonl}")
    questions_raw: list[str] = []
    answers_raw: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON line in {custom_jsonl}: {line[:120]}") from e
            if "question" not in obj or "answer" not in obj:
                continue
            questions_raw.append(obj["question"])
            answers_raw.append(obj["answer"])

    if n_samples is not None:
        questions_raw = questions_raw[:n_samples]
        answers_raw = answers_raw[:n_samples]

    dataset_dict = {"question": questions_raw, "answer": answers_raw}
    return Dataset.from_dict(dataset_dict)


def prepare_sft_dataset(n_samples, tokenizer, train: bool = True, custom_jsonl: str | None = None):
    """Prepare dataset for SFT training.

    If custom_jsonl is provided, it loads from that JSONL instead of the GSM8K split.
    """
    source = custom_jsonl if custom_jsonl else ("train" if train else "test")
    print(f"Preparing SFT dataset with {n_samples} samples (source={source})...")

    if custom_jsonl:
        dataset = _load_custom_traces(custom_jsonl, n_samples)
    else:
        dataset = load_dataset("openai/gsm8k", "main", split="train") if train else load_dataset("openai/gsm8k", "main", split="test")
        assert isinstance(dataset, Dataset)
    
    prompts = config.prompts
    system_prompt = prompts.system_prompt
    reasoning_start = prompts.reasoning_start
    reasoning_end = prompts.reasoning_end
    solution_start = prompts.solution_start
    solution_end = prompts.solution_end

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


def prepare_grpo_dataset(desired_size):
    """Prepare dataset for GRPO training"""
    print(f"Preparing GRPO dataset with {desired_size} samples...")
    
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    assert isinstance(dataset, Dataset)
    
    system_prompt = config.prompts.system_prompt
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
