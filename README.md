# GSM8K Training Pipeline

This project implements a compares results on GSM8K mathematical reasoning of Supervised Fine-Tuning (SFT) vs Group Relative Policy Optimization (GRPO).

## Setup

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv sync
```

## Full Training Pipeline

### Step 1: Evaluate Base Model
```bash
uv run src/evaluate_model.py --output-dir outputs/base_eval --output-file base_model_eval.csv
```

### Step 2: Train Preparation SFT Model (fewer samples)
Training automatically evaluates the model after completion.
```bash
uv run src/train_sft.py --output-dir outputs/prep_sft --stage prep
```

### Step 3: Train Full SFT Model (using prep model as base)
Training automatically evaluates the model after completion.
```bash
uv run src/train_sft.py --output-dir outputs/full_sft --stage full --base-model outputs/prep_sft
```

### Step 4: Train GRPO Model (using prep model as base)
Training automatically evaluates the model after completion.
```bash
uv run src/train_grpo.py --output-dir outputs/grpo --base-model outputs/prep_sft
```

## Standalone Evaluation

To evaluate any model without training:

### Base Model
```bash
uv run src/evaluate_model.py --output-dir outputs/base_eval --output-file base_model_eval.csv
```

### Preparation SFT Model
```bash
uv run src/evaluate_model.py --model-path outputs/prep_sft --output-dir outputs/prep_sft_eval --output-file prep_sft_eval.csv
```

### Full SFT Model
```bash
uv run src/evaluate_model.py --model-path outputs/full_sft --output-dir outputs/full_sft_eval --output-file full_sft_eval.csv
```

### GRPO Model
```bash
uv run src/evaluate_model.py --model-path outputs/grpo --output-dir outputs/grpo_eval --output-file grpo_eval.csv
```

## Generate GRPO Reasoning Traces

Use the new script `src/generate_traces.py` to create a dataset of reasoning traces where each example uses the first correct GRPO-generated trace (or falls back to the original GSM8K solution if none of the attempts are correct).

Example:

```bash
uv run src/generate_traces.py \
  --grpo-model outputs/compare-train-samples/1024/grpo \
  --output-dir outputs/traces \
  --output-file grpo_traces_train.jsonl \
  --dataset-split train \
  --num-samples 2048 \
  --batch-size 32 \
  --max-attempts 5 \
  --temperature 0.8
```

Outputs:
- JSONL dataset: `outputs/compare-train-samples/512/grpo_traces_train.jsonl`
- Stats JSON: `.../grpo_traces_train.jsonl.stats.json`

Stats fields:
- `total_samples`
- `grpo_successes`
- `fallback_count`
- `success_rate`
- `avg_attempts_for_successes`
- `avg_attempts_overall`

You can then train an SFT model on the generated traces by pointing your existing SFT training script to this JSONL (adapt loader if needed) and compare against original traces.

## Train SFT with Generated Traces

You can train using the generated reasoning traces JSONL instead of the original GSM8K answers:

```bash
uv run src/train_sft.py \
  --output-dir outputs/sft_from_grpo \
  --stage full \
  --traces-file outputs/traces/grpo_traces_train.jsonl
```

The JSONL must contain lines with:
```json
{"question": "...", "answer": "<reasoning>\n\n#### 42"}
```
The loader will build chat messages using your configured prompt tokens.

## Output Format

The model uses special tokens to structure its reasoning:
- `<start_working_out>` ... `<end_working_out>`: Contains step-by-step reasoning
- `<SOLUTION>` ... `</SOLUTION>`: Contains the final numerical answer

## Evaluation Metrics

Each evaluation produces:
- **Accuracy**: Percentage of correct final answers
- **Thinking Proportion**: Percentage of responses containing reasoning
- **Answer Proportion**: Percentage of responses with extractable final answers

Results are saved as CSV files in `[output-dir]/debug/` with detailed per-example analysis.

## Training Stages

1. **Base Model Evaluation**: Establishes baseline performance
2. **Preparation SFT**: Trains on fewer examples to establish basic format learning (saved to `outputs/prep_sft/`)
3. **Full SFT**: Trains on full dataset using prep model as starting point (saved to `outputs/full_sft/`)
4. **GRPO Training**: Uses reinforcement learning with reward functions, starting from prep model to improve reasoning quality (saved to `outputs/grpo/`)

## Directory Structure

Each training run creates the following structure:
```
[output-dir]/
├── model/          # Saved model files (LoRA weights)
└── debug/          # Evaluation results and metrics
    ├── [stage]_[run_id].csv       # Detailed evaluation results
    └── [stage]_[run_id]_metrics.txt  # Summary metrics
```

## File Descriptions

- `src/tr_config.py`: Central configuration for all parameters
- `src/model_utils.py`: Model loading, LoRA setup, and memory management
- `src/dataset_utils.py`: GSM8K data loading and formatting for training
- `src/reward_functions.py`: Reward functions that score format adherence and answer correctness
- `src/train_sft.py`: Supervised fine-tuning with configurable output directories
- `src/train_grpo.py`: GRPO training with configurable output directories
- `src/evaluate_model.py`: Batch evaluation with detailed result logging

## Usage Notes

# Experiments Repro

## Appendix A. Reproduction Steps for Experiment 1

Environment setup:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # adapt if using zsh/fish
uv sync
```

Prep formatting stage (teaches delimiter grammar):
```bash
uv run src/train_sft.py --stage prep \
  --output-dir outputs/compare-train-samples/prep_sft
```

Full SFT vs GRPO at a chosen sample budget (example: 512):
```bash
TRAIN_SAMPLES=512 uv run src/train_sft.py \
  --stage full \
  --base-model outputs/compare-train-samples/prep_sft \
  --output-dir outputs/compare-train-samples/512/full_sft

TRAIN_SAMPLES=512 uv run src/train_grpo.py \
  --base-model outputs/compare-train-samples/prep_sft \
  --output-dir outputs/compare-train-samples/512/grpo
```

Repeat for:
```bash
for n in 256 512 1024 2048; do
  TRAIN_SAMPLES=$n uv run src/train_sft.py --stage full \
    --base-model outputs/compare-train-samples/prep_sft \
    --output-dir outputs/compare-train-samples/$n/full_sft
  TRAIN_SAMPLES=$n uv run src/train_grpo.py \
    --base-model outputs/compare-train-samples/prep_sft \
    --output-dir outputs/compare-train-samples/$n/grpo
done
```

Optional standalone evaluation (auto-run during training):
```bash
uv run src/evaluate_model.py \
  --model-path outputs/compare-train-samples/512/full_sft \
  --output-dir outputs/compare-train-samples/512/full_sft_eval \
  --output-file full_sft_eval.csv

uv run src/evaluate_model.py \
  --model-path outputs/compare-train-samples/512/grpo \
  --output-dir outputs/compare-train-samples/512/grpo_eval \
  --output-file grpo_eval.csv
```

Key adjustable environment variables:
- TRAIN_SAMPLES / PREP_TRAIN_SAMPLES (dataset slice sizes)
- EVAL_SAMPLES (evaluation subset size)
- NUM_GENERATIONS (GRPO: sampled completions per prompt)
- OUTPUT_DIR (override default path root)

All outputs (adapter weights + metrics CSV + summary text file) land under the specified run directory: `model/` for weights/tokenizer artifacts and `debug/` for per-example records.

---
## Appendix B. Reproduction Steps for Experiment 2

Generate synthetic traces from a chosen GRPO checkpoint (example uses 1024-sample GRPO run):
```bash
uv run src/generate_traces.py \
  --grpo-model outputs/compare-train-samples/1024/grpo \
  --output-dir outputs/traces \
  --output-file grpo_traces_train.jsonl \
  --dataset-split train \
  --num-samples 2048 \
  --batch-size 32 \
  --max-attempts 5 \
  --temperature 0.8
```

Train SFT purely on generated traces (match TRAIN_SAMPLES to intended budget):
```bash
TRAIN_SAMPLES=1024 uv run src/train_sft.py \
  --stage full \
  --base-model outputs/compare-train-samples/prep_sft \
  --traces-file outputs/traces/grpo_traces_train.jsonl \
  --output-dir outputs/traces/1024
```

Evaluate (if needed beyond auto-run):
```bash
uv run src/evaluate_model.py \
  --model-path outputs/traces/1024 \
  --output-dir outputs/traces/1024_eval \
  --output-file traces_sft_eval.csv
```

Key parameters to adjust:
- --num-samples (how many original training problems to cover)
- --max-attempts (sampling attempts per problem before fallback)
- --temperature (diversity of sampled reasoning)
- --batch-size (throughput during trace generation)
- TRAIN_SAMPLES (size of supervised fine-tuning subset)

Trace stats sidecar (JSON) summarizes success rate, fallbacks, and mean attempts-per-success for auditability.


- **Output directories**: All scripts now use `--output-dir` to specify where to save files
- **Base models**: When using a previously trained model as base, pass its output directory to `--base-model`
- **Model files**: Models are saved in `[output-dir]/model/` and debug files in `[output-dir]/debug/`
- **Evaluation**: Use `--output-dir` to specify where evaluation results should be saved
