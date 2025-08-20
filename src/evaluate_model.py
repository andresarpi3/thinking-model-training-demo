#!/usr/bin/env python
"""Evaluate model on GSM8K test set."""

import unsloth
import argparse
import os
import pandas as pd
from vllm import SamplingParams

from tr_config import config
from model_utils import load_model
from dataset_utils import load_gsm8k_datasets, extract_hash_answer, extract_thinking, extract_solution


def evaluate_model(model, tokenizer, dataset, model_path=None):
    """Evaluate model on GSM8K test set using fast_generate with batching"""
    eval_config = config.evaluation
    prompts = config.prompts
    
    eval_n = eval_config.num_samples
    batch_size = eval_config.batch_size
    
    system_prompt = prompts.system_prompt
    reasoning_start = prompts.reasoning_start
    reasoning_end = prompts.reasoning_end
    solution_start = prompts.solution_start
    solution_end = prompts.solution_end
    
    print(f"Evaluating model on {eval_n} examples with batch size {batch_size}...")

    # Load LoRA adapter if model_path is provided
    lora_adapter = None
    if model_path:
        print(f"Loading LoRA adapter from {model_path}")
        lora_adapter = model.load_lora(model_path)

    # Take subset for evaluation
    eval_dataset = dataset.select(range(min(eval_n, len(dataset))))

    results = []
    correct_count = 0
    thinking_count = 0
    extracted_answer_count = 0

    # Process in batches
    for batch_start in range(0, len(eval_dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(eval_dataset))
        batch = eval_dataset.select(range(batch_start, batch_end))

        print(f"Processing batch {batch_start//batch_size + 1}/{(len(eval_dataset)-1)//batch_size + 1} (examples {batch_start}-{batch_end-1})")

        # Prepare batch prompts
        batch_prompts = []
        batch_questions = []
        batch_labeled_cots = []
        batch_labeled_answers = []

        for example in batch:
            question = example["question"]
            labeled_cot = example["answer"]
            labeled_answer = extract_hash_answer(labeled_cot)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            batch_prompts.append(text)
            batch_questions.append(question)
            batch_labeled_cots.append(labeled_cot)
            batch_labeled_answers.append(labeled_answer)

        # Generate responses using fast_generate
        sampling_params = SamplingParams(
            temperature=eval_config.temperature,
            top_k=eval_config.top_k,
            max_tokens=eval_config.max_tokens,
            stop=[tokenizer.eos_token],
        )

        outputs = model.fast_generate(
            batch_prompts,
            sampling_params=sampling_params,
            lora_request=lora_adapter,
        )

        # Process batch results
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text

            question = batch_questions[i]
            labeled_cot = batch_labeled_cots[i]
            labeled_answer = batch_labeled_answers[i]

            # Extract components
            extracted_thinking = extract_thinking(reasoning_start + generated_text, reasoning_start, reasoning_end)
            extracted_answer = extract_solution(generated_text, solution_start, solution_end)

            # Check correctness
            correct = False
            if extracted_answer and labeled_answer:
                try:
                    # Try numerical comparison
                    extracted_num = float(extracted_answer.replace(",", "").strip())
                    labeled_num = float(labeled_answer.replace(",", "").strip())
                    correct = abs(extracted_num - labeled_num) < 1e-6
                except:
                    # Fall back to string comparison
                    correct = extracted_answer.strip() == labeled_answer.strip()

            # Update counts
            if correct:
                correct_count += 1
            if extracted_thinking:
                thinking_count += 1
            if extracted_answer:
                extracted_answer_count += 1

            results.append({
                "question": question,
                "labeled_cot": labeled_cot,
                "labeled_answer": labeled_answer,
                "generated_text": generated_text,
                "extracted_thinking": extracted_thinking,
                "extracted_answer": extracted_answer,
                "correct": correct
            })

    # Calculate metrics
    total = len(results)
    accuracy = correct_count / total
    thinking_prop = thinking_count / total
    answer_prop = extracted_answer_count / total

    print(f"Accuracy: {accuracy:.3f} ({correct_count}/{total})")
    print(f"Proportion with thinking: {thinking_prop:.3f} ({thinking_count}/{total})")
    print(f"Proportion with extracted answer: {answer_prop:.3f} ({extracted_answer_count}/{total})")

    return results, accuracy, thinking_prop, answer_prop


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K")
    # Config is now imported directly from config.py
    parser.add_argument("--model-path", type=str, help="Model name (e.g., 'sft_model', 'grpo_model') or None for base model")
    parser.add_argument("--output-file", type=str, required=True, help="Output CSV file")
    
    args = parser.parse_args()
    
    # Config is imported directly from config.py
    
    # Create output directories
    os.makedirs(config.outputs.get_debug_path(), exist_ok=True)
    
    # Resolve model path from config
    model_path = None
    if args.model_path:
        # Map model names to their full paths
        model_path_map = {
            'sft_model': config.outputs.get_sft_model_path(),
            'rl_sft_model': config.outputs.get_rl_sft_model_path(),
            'grpo_model': config.outputs.get_grpo_model_path()
        }
        model_path = model_path_map[args.model_path]
        print(f"Using model: {args.model_path} -> {model_path}")
    else:
        print("Evaluating base model")
    
    # Load model and datasets
    model, tokenizer = load_model()
    _, gsm8k_test = load_gsm8k_datasets()
    
    # Evaluate model
    results, accuracy, thinking_prop, answer_prop = evaluate_model(
        model, tokenizer, gsm8k_test, model_path
    )
    
    # Save results
    output_path = os.path.join(config.outputs.get_debug_path(), args.output_file)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()