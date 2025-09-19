#!/usr/bin/env python
"""Evaluate model on GSM8K test set."""

from typing import Mapping
import unsloth
import argparse
import os
import pandas as pd
import numpy as np
from vllm import SamplingParams
from wandb import Run

from tr_config import config
from model_utils import load_model
from dataset_utils import prepare_sft_dataset, extract_hash_answer, extract_thinking, extract_solution, extract_confidence
from datasets import Dataset
from dataclasses import dataclass, field

@dataclass
class EvalResults:
    df: pd.DataFrame
    accuracy: float
    thinking_proportion: float
    answer_proportion: float
    confidence_proportion: float = 0.0
    expected_calibration_error: float = 0.0
    confidence_accuracy_correlation: float = 0.0
    calibration_bins: list = field(default_factory=list)


def calculate_calibration_metrics(confidences, correctness, n_bins=10):
    """Calculate Expected Calibration Error and calibration bins"""
    if len(confidences) == 0:
        return 0.0, 0.0, []
    
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    calibration_data = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = correctness[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # ECE contribution
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            calibration_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'accuracy': accuracy_in_bin,
                'confidence': avg_confidence_in_bin,
                'count': in_bin.sum(),
                'proportion': prop_in_bin
            })
        else:
            calibration_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'accuracy': 0.0,
                'confidence': 0.0,
                'count': 0,
                'proportion': 0.0
            })
    
    # Calculate correlation
    if len(confidences) > 1:
        correlation = np.corrcoef(confidences, correctness)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    
    return ece, correlation, calibration_data


def evaluate_model(model, tokenizer, lora_adapter, eval_dataset: Dataset, use_confidence: bool = False):
    """Evaluate model on GSM8K test set using fast_generate with batching"""
    eval_config = config.evaluation
    prompts = config.prompts
    
    batch_size = eval_config.batch_size
    
    system_prompt = prompts.system_prompt
    reasoning_start = prompts.reasoning_start
    reasoning_end = prompts.reasoning_end
    solution_start = prompts.solution_start
    solution_end = prompts.solution_end
    confidence_start = prompts.confidence_start
    confidence_end = prompts.confidence_end
    

    results = []
    correct_count = 0
    thinking_count = 0
    extracted_answer_count = 0
    extracted_confidence_count = 0
    confidences = []
    correctness = []

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
            assert isinstance(example, Mapping)

            question = example["question"]
            labeled_cot = example["answer"]
            labeled_answer = example["labeled_answer"]
            text = example["prompt"]

            batch_prompts.append(text)
            batch_questions.append(question)
            batch_labeled_cots.append(labeled_cot)
            batch_labeled_answers.append(labeled_answer)

        # Generate responses using fast_generate
        sampling_params = SamplingParams(
            temperature=eval_config.temperature,
            top_k=eval_config.top_k,
            max_tokens=config.model.max_seq_length,
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
            extracted_confidence = extract_confidence(generated_text, confidence_start, confidence_end) if use_confidence else None

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
            if extracted_confidence is not None:
                extracted_confidence_count += 1
                confidences.append(extracted_confidence)
                correctness.append(correct)

            results.append({
                "question": question,
                "labeled_cot": labeled_cot,
                "labeled_answer": labeled_answer,
                "generated_text": generated_text,
                "extracted_thinking": extracted_thinking,
                "extracted_answer": extracted_answer,
                "extracted_confidence": extracted_confidence,
                "correct": correct
            })

    # Calculate metrics
    total = len(results)
    accuracy = correct_count / total
    thinking_prop = thinking_count / total
    answer_prop = extracted_answer_count / total
    confidence_prop = extracted_confidence_count / total

    print(f"Accuracy: {accuracy:.3f} ({correct_count}/{total})")
    print(f"Proportion with thinking: {thinking_prop:.3f} ({thinking_count}/{total})")
    print(f"Proportion with extracted answer: {answer_prop:.3f} ({extracted_answer_count}/{total})")
    
    # Calculate confidence calibration metrics
    ece = 0.0
    correlation = 0.0
    calibration_data = []
    
    if use_confidence and len(confidences) > 0:
        print(f"Proportion with extracted confidence: {confidence_prop:.3f} ({extracted_confidence_count}/{total})")
        ece, correlation, calibration_data = calculate_calibration_metrics(
            confidences, correctness, n_bins=config.evaluation.confidence_bins
        )
        print(f"Expected Calibration Error: {ece:.3f}")
        print(f"Confidence-Accuracy Correlation: {correlation:.3f}")

    return EvalResults(
        df=pd.DataFrame(results), 
        accuracy=accuracy, 
        thinking_proportion=thinking_prop, 
        answer_proportion=answer_prop,
        confidence_proportion=confidence_prop,
        expected_calibration_error=ece,
        confidence_accuracy_correlation=correlation,
        calibration_bins=calibration_data
    )


def save_outputs_from_eval(output_file: str, results: EvalResults, run: Run | None = None, debug_dir: str | None = None):
    # Use provided debug_dir or default to 'debug'
    if debug_dir is None:
        debug_dir = "debug"
    
    os.makedirs(debug_dir, exist_ok=True)

    output_path = os.path.join(debug_dir, output_file)
    results.df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    with open(os.path.join(debug_dir, output_file.replace(".csv", "_metrics.txt")), "w") as f:
        f.write(f"Accuracy: {results.accuracy:.3f}\n")
        f.write(f"Proportion with thinking: {results.thinking_proportion:.3f}\n")
        f.write(f"Proportion with extracted answer: {results.answer_proportion:.3f}\n")
        f.write(f"Proportion with extracted confidence: {results.confidence_proportion:.3f}\n")
        f.write(f"Expected Calibration Error: {results.expected_calibration_error:.3f}\n")
        f.write(f"Confidence-Accuracy Correlation: {results.confidence_accuracy_correlation:.3f}\n")
        
    # Save calibration data if available
    if results.calibration_bins:
        calibration_df = pd.DataFrame(results.calibration_bins)
        calibration_path = os.path.join(debug_dir, output_file.replace(".csv", "_calibration.csv"))
        calibration_df.to_csv(calibration_path, index=False)
        print(f"Calibration data saved to {calibration_path}")
        
    if run:
        run.summary["accuracy"] = results.accuracy
        run.summary["proportion_with_thinking"] = results.thinking_proportion
        run.summary["proportion_with_extracted_answer"] = results.answer_proportion
        run.summary["proportion_with_extracted_confidence"] = results.confidence_proportion
        run.summary["expected_calibration_error"] = results.expected_calibration_error
        run.summary["confidence_accuracy_correlation"] = results.confidence_accuracy_correlation

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K")
    parser.add_argument("--model-path", type=str, help="Model output directory path (e.g., 'outputs/prep_sft') or None for base model")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for results (default: current directory)")
    parser.add_argument("--output-file", type=str, required=True, help="Output CSV file")
    parser.add_argument("--use-confidence", action="store_true", help="Enable confidence evaluation mode")
    
    args = parser.parse_args()
    
    
    # Resolve model path
    model_path = None
    if args.model_path:
        # Look for model in provided directory/model subdirectory
        model_path = os.path.join(args.model_path, "model")
        print(f"Using model: {model_path}")
    else:
        print("Evaluating base model")
    
    # Load model and datasets
    model, tokenizer = load_model(model_path)
    lora_adapter = model.load_lora(model_path) if model_path else None
    
    eval_n = config.evaluation.num_samples
    eval_dataset = prepare_sft_dataset(eval_n, tokenizer, train=False, use_confidence=args.use_confidence)
    
    # Evaluate model
    results = evaluate_model(
        model, tokenizer, lora_adapter, eval_dataset, use_confidence=args.use_confidence
    )
    
    # Save results to specified output directory
    debug_dir = os.path.join(args.output_dir, "debug")
    save_outputs_from_eval(args.output_file, results, debug_dir=debug_dir)



if __name__ == "__main__":
    main()