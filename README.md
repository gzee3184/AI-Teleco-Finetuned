# AI-Teleco-Finetuned
Finetuning code for of AI Teleco Challenge model

# V5D Model — README

The v5d model is a LoRA fine-tuned Qwen 2.5 1.5B Instruct model for 5G root cause classification, guided by an XGBoost classifier that provides an initial prediction and confidence score.

---

## 1. Dependencies

| Package | Why it's needed |
|---|---|
| `torch` | Runs the Qwen base model on CPU or GPU |
| `transformers` | Loads the `Qwen/Qwen2.5-1.5B-Instruct` tokenizer and model weights |
| `peft` | Applies and merges the LoRA adapter onto the base model |
| `datasets` | Provides the `Dataset` class used during LoRA training |
| `xgboost` | The XGBoost classifier that produces the initial prediction fed into the prompt |
| `scikit-learn` | Label encoding and TF-IDF vectorizer (used at XGBoost training time) |
| `pandas` / `numpy` | Data loading and feature vector construction |
| `tqdm` | Progress bars during prediction generation |



---

## 2. Training the Qwen LoRA Model (`lora_train.py`)

### Training data
The `lora_data_v5d/` folder contains three JSON files:

| File | Purpose |
|---|---|
| `lora_data_v5d/train.json` | Training samples |
| `lora_data_v5d/val.json` | Validation samples (used for early stopping) |
| `lora_data_v5d/holdout.json` | Held-out samples (not used during training) |

Each JSON entry has three fields:
- `Input_Prompt` — the v5d prompt fed to the model
- `Reasoning_Trace` — the expected reasoning + boxed answer (training target)
- `Generated_Label` — the target class label (`C1`–`C8`)

### Basic training command

```bash
python lora_train.py \
  --train_data lora_data_v5d/train.json \
  --val_data   lora_data_v5d/val.json \
  --output_dir lora_output_v5d
```

The final adapter is saved to `lora_output_v5d/final/`.

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--train_data` | `lora_data/train.json` | Path to training JSON |
| `--val_data` | `lora_data/val.json` | Path to validation JSON |
| `--model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model |
| `--output_dir` | `lora_output` | Where to save the adapter |
| `--lora_r` | `8` | LoRA rank (lower = less overfitting) |
| `--lora_alpha` | `16` | LoRA scaling factor |
| `--lora_dropout` | `0.1` | Dropout for regularisation |
| `--num_epochs` | `5` | Max epochs (early stopping applies) |
| `--learning_rate` | `2e-5` | Learning rate |
| `--batch_size` | `2` | Per-device batch size |
| `--early_stopping_patience` | `3` | Stop after N evals with no improvement |

Training uses `fp16` and evaluates every 50 steps on validation loss. The best checkpoint is automatically restored at the end.

---

## 3. Generating Predictions (`generate_v5d_predictions.py`)

Runs the v5d model over `phase_1_test.csv` and `train.csv` and writes raw predictions to CSV files.

```bash
python generate_v5d_predictions.py
```

### What it does
1. Loads the XGBoost classifier (`xgboost_model.pkl`) and v5d LoRA adapter (`lora_output_v5d/final/`)
2. Parses each question's network data tables (`seperate_values.py`)
3. Extracts numerical features (`rule_based_classifier.py`)
4. Builds the v5d prompt (`generate_submissions_moe.py`)
5. Generates a response with `max_new_tokens=100, temperature=0.1`
6. Saves to:

| Output file | Contents |
|---|---|
| `v5d_phase1_predictions.csv` | Predictions for `phase_1_test.csv` (864 questions) |
| `v5d_train_predictions.csv` | Predictions for `train.csv` (112K questions) |

> **Note:** Runs on CPU (~5.9 s/sample). Phase 1 takes ~80 minutes. Train.csv requires ~185 hours.

---

## 4. Analyzing Predictions (`analyze_v5d_predictions.py`)

Compares generated predictions against `phase_1_test_truth.csv` and prints accuracy metrics and failure patterns. Must be run after `generate_v5d_predictions.py`.

```bash
python analyze_v5d_predictions.py
```

### What it does
1. Loads `v5d_phase1_predictions.csv`
2. Re-extracts answers using a robust regex that handles multiple output formats:
   - `\boxed{CX}`
   - `the answer is: CX`
   - `Output: CX`
   - Any standalone `C[1-8]` in the text
3. Loads ground truth from `phase_1_test_truth.csv`
4. Evaluates: a prediction is **correct** if it matches any of the 4 sub-question labels (`_1`–`_4`) for that base question ID
5. Saves detailed results to `v5d_phase1_analysis_predictions.csv`

### Output includes
- Overall accuracy
- Per-class accuracy (C1–C8)
- Top failure patterns (e.g. `C1 → C3`)
- Class distribution comparison (predicted vs truth)

---

## File Summary

```
lora_train.py                    # LoRA fine-tuning script
lora_data_v5d/
  train.json                     # Training data
  val.json                       # Validation data
  holdout.json                   # Held-out data
generate_v5d_predictions.py      # Run v5d on test/train data
analyze_v5d_predictions.py          # Compare predictions vs ground truth

# Dependencies (called internally)
integrated_classifier_v3.py      # Loads LoRA model + runs inference
xgboost_tool.py                  # Loads XGBoost, returns prediction + confidence
seperate_values.py               # Parses network data tables from question text
rule_based_classifier.py         # Extracts numerical features from parsed tables
generate_submissions_moe.py      # Builds the v5d prompt

# Model artifacts
lora_output_v5d/final/           # Trained LoRA adapter weights
xgboost_model.pkl                # Trained XGBoost model

# Data
phase_1_test.csv                 # Phase 1 test questions
phase_1_test_truth.csv           # Phase 1 ground truth labels
train.csv                        # Training questions with labels
```

