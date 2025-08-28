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
uv run src/evaluate_model.py  --output-file base_model_eval.csv
```

### Step 2: Train Preparation SFT Model (fewer samples)
Training automatically evaluates the model after completion.
```bash
uv run src/train_sft.py --stage prep
```

### Step 3: Train Full SFT Model (using prep model as base)
Training automatically evaluates the model after completion.
```bash
uv run src/train_sft.py --stage full --base-model prep_sft_model
```

### Step 4: Train GRPO Model (using prep model as base)
Training automatically evaluates the model after completion.
```bash
uv run src/train_grpo.py --base-model prep_sft_model
```

## Standalone Evaluation

To evaluate any model without training:

### Base Model
```bash
uv run src/evaluate_model.py --output-file base_model_eval.csv
```

### Preparation SFT Model
```bash
uv run src/evaluate_model.py --model-path prep_sft_model --output-file prep_sft_eval.csv
```

### Full SFT Model
```bash
uv run src/evaluate_model.py --model-path sft_model --output-file sft_model_eval.csv
```

### GRPO Model
```bash
uv run src/evaluate_model.py --model-path grpo_model --output-file grpo_model_eval.csv
```

## Output Format

The model uses special tokens to structure its reasoning:
- `<start_working_out>` ... `<end_working_out>`: Contains step-by-step reasoning
- `<SOLUTION>` ... `</SOLUTION>`: Contains the final numerical answer

## Evaluation Metrics

Each evaluation produces:
- **Accuracy**: Percentage of correct final answers
- **Thinking Proportion**: Percentage of responses containing reasoning
- **Answer Proportion**: Percentage of responses with extractable final answers

Results are saved as CSV files in `outputs/debug/` with detailed per-example analysis.

## Training Stages

1. **Base Model Evaluation**: Establishes baseline performance
2. **Preparation SFT**: Trains on fewer examples to establish basic format learning
3. **Full SFT**: Trains on full dataset using prep model as starting point
4. **GRPO Training**: Uses reinforcement learning with reward functions, starting from prep model to improve reasoning quality

## File Descriptions

- `src/tr_config.py`: Central configuration for all parameters
- `src/model_utils.py`: Model loading, LoRA setup, and memory management
- `src/dataset_utils.py`: GSM8K data loading and formatting for training
- `src/reward_functions.py`: Reward functions that score format adherence and answer correctness
- `src/train_sft.py`: Supervised fine-tuning with configurable sample sizes
- `src/train_grpo.py`: GRPO training with multiple reward functions
- `src/evaluate_model.py`: Batch evaluation with detailed result logging
