# GSM8K Training Pipeline

This project implements a complete training pipeline for GSM8K mathematical reasoning using Supervised Fine-Tuning (SFT) followed by Group Relative Policy Optimization (GRPO).

## Project Structure

```
├── config.json                 # Configuration file
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── src/
│   ├── model_utils.py         # Model loading and setup utilities
│   ├── dataset_utils.py       # Dataset preparation and text extraction
│   ├── reward_functions.py    # Reward functions for GRPO training
│   ├── evaluate_model.py      # Model evaluation script
│   ├── train_sft.py          # Supervised fine-tuning script
│   └── train_grpo.py         # GRPO reinforcement learning script
└── outputs/
    ├── models/               # Trained model checkpoints
    │   ├── prep_sft_model/   # Preparation SFT model (trained on smaller dataset)
    │   ├── sft_model/        # Full SFT model (trained on larger dataset)
    │   └── grpo_model/       # GRPO model (trained with reinforcement learning)
    └── debug/               # Evaluation results and debug info
        └── *.csv           # Evaluation CSV files
```

## Command Line Arguments

The scripts use minimal command line arguments, with most configuration handled via `config.json`:

- `--config`: Path to configuration file (required for all scripts)
- `--model-path`: Model name from config (e.g., `sft_model`, `grpo_model`) or None for base model (evaluate_model.py)
- `--base-model`: Base model name from config for GRPO training (train_grpo.py) and optionally for full SFT training (train_sft.py)
- `--stage`: Training stage - `full` or `prep` (train_sft.py)
- `--output-file`: Output CSV filename for evaluation results

Model paths are automatically resolved from the config file, so you reference models by name rather than full paths.

## Setup

After restarting a pod, install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

## Configuration

Edit `config.json` to adjust:
- Model settings (name, sequence length, LoRA parameters)
- Training hyperparameters (learning rates, batch sizes, epochs)
- Evaluation settings (number of samples, generation parameters)
- Output directories
- Prompt templates and special tokens

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

- `config.json`: Central configuration for all parameters
- `src/model_utils.py`: Model loading, LoRA setup, and memory management
- `src/dataset_utils.py`: GSM8K data loading and formatting for training
- `src/reward_functions.py`: Reward functions that score format adherence and answer correctness
- `src/train_sft.py`: Supervised fine-tuning with configurable sample sizes
- `src/train_grpo.py`: GRPO training with multiple reward functions
- `src/evaluate_model.py`: Batch evaluation with detailed result logging


export  OUTPUT_DIR="outputs/short-run" \
        EVAL_SAMPLES=128 \
        TRAIN_SAMPLES=256 
uv run src/evaluate_model.py --output-file base_model_eval.csv