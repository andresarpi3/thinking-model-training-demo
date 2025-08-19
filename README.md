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
    │   ├── sft_model/
    │   ├── rl_sft_model/
    │   └── grpo_model/
    └── debug/               # Evaluation results and debug info
        └── *.csv           # Evaluation CSV files
```

## Command Line Arguments

The scripts use minimal command line arguments, with most configuration handled via `config.json`:

- `--config`: Path to configuration file (required for all scripts)
- `--model-path`: Model name from config (e.g., `sft_model`, `grpo_model`) or None for base model (evaluate_model.py)
- `--base-model`: Base model name from config for GRPO training (train_grpo.py)
- `--stage`: Training stage - `full` or `rl_prep` (train_sft.py)
- `--output-file`: Output CSV filename for evaluation results

Model paths are automatically resolved from the config file, so you reference models by name rather than full paths.

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
uv run src/evaluate_model.py --config config.json --output-file base_model_eval.csv
```

### Step 2: Train SFT Model (Full)
```bash
uv run src/train_sft.py --config config.json --stage full
```

### Step 3: Evaluate SFT Model
```bash
uv run src/evaluate_model.py --config config.json --model-path sft_model --output-file sft_model_eval.csv
```

### Step 4: Train RL Preparation Model (SFT with fewer samples)
```bash
uv run src/train_sft.py --config config.json --stage rl_prep
```

### Step 5: Evaluate RL-SFT Model
```bash
uv run src/evaluate_model.py --config config.json --model-path rl_sft_model --output-file rl_sft_eval.csv
```

### Step 6: Train GRPO Model
```bash
uv run src/train_grpo.py --config config.json --base-model rl_sft_model
```

### Step 7: Evaluate GRPO Model
```bash
uv run src/evaluate_model.py --config config.json --model-path grpo_model --output-file grpo_model_eval.csv
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
2. **Full SFT**: Trains on 1024 examples to learn the output format
3. **RL Preparation SFT**: Trains on 256 examples as base for RL
4. **GRPO Training**: Uses reinforcement learning with reward functions to improve reasoning quality

## File Descriptions

- `config.json`: Central configuration for all parameters
- `src/model_utils.py`: Model loading, LoRA setup, and memory management
- `src/dataset_utils.py`: GSM8K data loading and formatting for training
- `src/reward_functions.py`: Reward functions that score format adherence and answer correctness
- `src/train_sft.py`: Supervised fine-tuning with configurable sample sizes
- `src/train_grpo.py`: GRPO training with multiple reward functions
- `src/evaluate_model.py`: Batch evaluation with detailed result logging