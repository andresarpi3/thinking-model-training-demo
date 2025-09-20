

import argparse
import json
import os
import time
from dataclasses import dataclass

from datasets import Dataset
from vllm import SamplingParams

from tr_config import config
from model_utils import load_model
from dataset_utils import (
    prepare_sft_dataset,
    extract_solution,
)

# Reuse tokens from config
prompts_cfg = config.prompts
REASONING_START = prompts_cfg.reasoning_start
REASONING_END = prompts_cfg.reasoning_end
SOLUTION_START = prompts_cfg.solution_start
SOLUTION_END = prompts_cfg.solution_end


@dataclass
class SampleResult:
    question: str
    final_answer_text: str  # GSM8K formatted trace text (original or generated)
    used_generated: bool
    attempts_used: int
    generated_trace:  str | None = None


def extract_numeric_answer(text: str) ->  str | None:
    """Extract answer between solution markers, returning stripped numeric string.
    Returns None if not found."""
    extracted = extract_solution(text, SOLUTION_START, SOLUTION_END)
    if extracted is None:
        return None
    # Basic clean
    return extracted.strip()


def answers_match(generated: str | None, labeled: str | None) -> bool:
    if not generated or not labeled:
        return False
    g = generated.replace(",", "").strip()
    l = labeled.replace(",", "").strip()
    # Try float comparison then fallback string
    try:
        return abs(float(g) - float(l)) < 1e-9
    except Exception:
        return g == l


def build_gsm8k_answer_from_generation(original_answer: str, generated_text: str, labeled_answer: str) -> str:
    """Reconstruct GSM8K style answer using the model's reasoning + labeled answer.

    The requirement: if model succeeds we want the *generated reasoning trace*.
    We rely on extracted reasoning between REASONING_START/END and attach the final answer
    as #### <answer> to match original dataset style.

    If we cannot cleanly extract reasoning we fallback to original full answer.
    """
    # Attempt to isolate reasoning. The reasoning template in training wraps thinking in tokens.
    # We search for reasoning markers; if absent we fallback.
    import re
    pattern = rf"{REASONING_START}(.*?){REASONING_END}"
    m = re.search(pattern, generated_text, re.DOTALL)
    if not m:
        return original_answer  # fallback - formatting mismatch
    reasoning_body = m.group(1).strip()

    # Rebuild in GSM8K style: reasoning lines, then #### answer
    return f"{reasoning_body}\n\n#### {labeled_answer}"


def generate_traces(
    model,
    tokenizer,
    lora_adapter,
    dataset: Dataset,
    batch_size: int,
    max_attempts: int,
    temperature: float,
) -> list[SampleResult]:
    results: list[SampleResult] = []

    total = len(dataset)
    idx = 0

    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=config.evaluation.top_k,
        max_tokens=config.model.max_seq_length,
        stop=[tokenizer.eos_token],
    )

    start_time = time.time()

    while idx < total:
        # How many samples to include this meta-batch
        samples_this_batch = max(1, batch_size // max_attempts)
        remaining = total - idx
        samples_this_batch = min(samples_this_batch, remaining)

        batch = dataset.select(range(idx, idx + samples_this_batch))

        # Prepare prompts and metadata
        prompts: list[str] = []
        meta = []  # (sample_local_index, attempt_index)
        questions = []
        labeled_answers = []
        original_answers = []

        for local_i, example in enumerate(batch):
            prompt = example["prompt"]
            question = example["question"]
            original_answer_full = example["answer"]  # full GSM8K trace
            labeled_answer = example["labeled_answer"]

            questions.append(question)
            original_answers.append(original_answer_full)
            labeled_answers.append(labeled_answer)

            # replicate prompt max_attempts times
            for attempt_j in range(max_attempts):
                prompts.append(prompt)
                meta.append((local_i, attempt_j))

        # If over batch size because integer division maybe 0? we guard
        if len(prompts) > batch_size:
            prompts = prompts[:batch_size]
            meta = meta[:batch_size]

        # Generate
        generations = model.fast_generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=lora_adapter,
        )

        # Collect per-sample attempts
        per_sample_attempts: list[list[str]] = [[] for _ in range(samples_this_batch)]
        for gen_obj, (local_i, attempt_j) in zip(generations, meta):
            # Only take first output
            text = gen_obj.outputs[0].text
            per_sample_attempts[local_i].append(text)

        # Decide final trace per sample
        for local_i in range(samples_this_batch):
            question = questions[local_i]
            labeled_answer = labeled_answers[local_i]
            original_answer_full = original_answers[local_i]

            chosen_trace = None
            attempts_used = 0
            used_generated = False

            for attempt_text in per_sample_attempts[local_i]:
                attempts_used += 1
                extracted_numeric = extract_numeric_answer(attempt_text)
                if answers_match(extracted_numeric, labeled_answer):
                    # success
                    gsm8k_style = build_gsm8k_answer_from_generation(
                        original_answer_full, attempt_text, labeled_answer
                    )
                    chosen_trace = gsm8k_style
                    used_generated = True
                    break

            if chosen_trace is None:
                # fallback: original full answer already contains #### numeric answer
                chosen_trace = original_answer_full
                attempts_used = len(per_sample_attempts[local_i])

            results.append(
                SampleResult(
                    question=question,
                    final_answer_text=chosen_trace,
                    used_generated=used_generated,
                    attempts_used=attempts_used,
                    generated_trace=chosen_trace if used_generated else None,
                )
            )

        idx += samples_this_batch
        done = len(results)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0.0
        print(
            f"Processed {done}/{total} samples | Batch {samples_this_batch} | Elapsed {elapsed:.1f}s | {rate:.2f} samples/s"
        )

    return results


def write_outputs(
    results: list[SampleResult],
    output_path: str,
    stats_path: str,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # JSONL dataset
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            obj = {"question": r.question, "answer": r.final_answer_text}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Stats
    total = len(results)
    successes = sum(1 for r in results if r.used_generated)
    fallback = total - successes
    avg_attempts_success = (
        sum(r.attempts_used for r in results if r.used_generated) / successes
        if successes > 0
        else 0.0
    )
    avg_attempts_overall = sum(r.attempts_used for r in results) / total if total else 0.0

    stats = {
        "total_samples": total,
        "grpo_successes": successes,
        "fallback_count": fallback,
        "success_rate": successes / total if total else 0.0,
        "avg_attempts_for_successes": avg_attempts_success,
        "avg_attempts_overall": avg_attempts_overall,
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote dataset to {output_path}")
    print(f"Wrote stats to {stats_path}")
    print("Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate reasoning traces using a GRPO fine-tuned model")
    parser.add_argument("--grpo-model", type=str, required=False, help="Path to trained GRPO model directory (expects subdir 'model')")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSONL file name (e.g., traces.jsonl)")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples (default: full dataset as per config.dataset_size.train_samples)")
    parser.add_argument("--dataset-split", type=str, choices=["train", "test"], default="train", help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=32, help="Generation batch size")
    parser.add_argument("--max-attempts", type=int, default=5, help="Max attempts per sample")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for output files")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve model path (LoRA adapter)
    lora_path = None
    if args.grpo_model:
        lora_path = os.path.join(args.grpo_model, "model")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA model path not found: {lora_path}")
        print(f"Using GRPO model adapter at {lora_path}")
    else:
        print("Using base model (no GRPO adapter)")

    model, tokenizer = load_model(lora_path)
    lora_adapter = model.load_lora(lora_path) if lora_path else None

    # Determine number of samples
    if args.num_samples is not None:
        n_samples = args.num_samples
    else:
        # Use config sizes (train vs test)
        n_samples = config.dataset_size.train_samples if args.dataset_split == "train" else config.evaluation.num_samples

    dataset = prepare_sft_dataset(n_samples, tokenizer, train=(args.dataset_split == "train"))

    print(f"Generating traces for {len(dataset)} samples | batch_size={args.batch_size} | max_attempts={args.max_attempts}")

    results = generate_traces(
        model=model,
        tokenizer=tokenizer,
        lora_adapter=lora_adapter,
        dataset=dataset,
        batch_size=args.batch_size,
        max_attempts=args.max_attempts,
        temperature=args.temperature,
    )

    output_path = os.path.join(args.output_dir, args.output_file)
    stats_path = output_path + ".stats.json"
    write_outputs(results, output_path, stats_path)


if __name__ == "__main__":
    main()
