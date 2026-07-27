# CSA-GMoE-Context-Aware-Multi-Label-Object-Presence-Prediction-in-Autonomous-Driving
CSA-GMoE: Condition-Supervised Adaptive Gating MoE for Context-Aware Multi-Label Object-Presence Prediction in Autonomous Driving



# BDD100K Condition-Specialized Mixture of Experts

This repository contains the baseline training and evaluation pipeline for a
condition-specialized Mixture-of-Experts (MoE) model on BDD100K road images.
The model predicts the image-level presence of cars, pedestrians, and traffic
signs. It is an image classification model, not an object detector.

Three experts specialize in weather, scene, and time of day. A learned router
combines the expert predictions into the final MoE prediction.

## Files in this repository

```text
.
├── README.md
├── requirements.txt
└── scripts/
    ├── bdd100k_moe_train.py
    ├── bdd100k_moe_test.py
    └── preprocessing/
        ├── check_bdd100k_paths.py
        └── download_bdd100k_kagglehub.py
```

- `bdd100k_moe_train.py` prepares fixed metadata splits and trains the model.
- `bdd100k_moe_test.py` evaluates checkpoints and creates CSV reports and plots.
- `check_bdd100k_paths.py` verifies the extracted images and annotations.
- `download_bdd100k_kagglehub.py` optionally downloads a community mirror.

## Dataset

Download BDD100K from the [official BDD100K download
page](https://bdd-data.berkeley.edu/download.html). This project requires:

- **100K Images**
- **Detection 2020 Labels**

The official toolkit and documentation are available in the [BDD100K GitHub
repository](https://github.com/bdd100k/bdd100k).

An optional helper uses the community Kaggle mirror
[`awsaf49/bdd100k-dataset`](https://www.kaggle.com/datasets/awsaf49/bdd100k-dataset):

```bash
python scripts/preprocessing/download_bdd100k_kagglehub.py
```

Use the official source when possible and comply with the BDD100K dataset terms.
Do not commit the dataset to GitHub.

## Expected dataset layout

```text
/path/to/bdd100k/
├── images/
│   └── 100k/
│       ├── train/*.jpg
│       └── val/*.jpg
└── labels/
    └── det_20/
        ├── det_train.json
        └── det_val.json
```

Some mirrors use `det_v2_train_release.json` and
`det_v2_val_release.json`. Supply the actual paths on your machine.

## Installation

Python 3.10 or newer is recommended. A CUDA-capable GPU is strongly recommended
for training. Install the correct PyTorch build for your system from the
[official PyTorch installer](https://pytorch.org/get-started/locally/), then run:

```bash
git clone <BASELINE-REPOSITORY-URL>
cd <BASELINE-REPOSITORY>
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Suggested `requirements.txt`:

```text
numpy>=1.26
pillow>=10.0
matplotlib>=3.8
scikit-learn>=1.4
timm>=1.0.15
torch>=2.1
torchvision>=0.16
torchmetrics>=1.3
kagglehub>=0.3
```

## Run order

Run every command from the repository root. Define local paths first:

```bash
DATA_ROOT=/path/to/bdd100k
TRAIN_LABELS="$DATA_ROOT/labels/det_20/det_train.json"
VAL_LABELS="$DATA_ROOT/labels/det_20/det_val.json"
METADATA_DIR=outputs/metadata
RUN_DIR=experiments/moe_stage3
```

### 1. Check the dataset paths

This is the first script to run:

```bash
python scripts/preprocessing/check_bdd100k_paths.py \
  --data-root "$DATA_ROOT" \
  --det-train-json "$TRAIN_LABELS" \
  --det-val-json "$VAL_LABELS" \
  --num-samples 100
```

### 2. Prepare deterministic metadata

```bash
python scripts/bdd100k_moe_train.py prepare \
  --data-root "$DATA_ROOT" \
  --det-train-json "$TRAIN_LABELS" \
  --det-val-json "$VAL_LABELS" \
  --output-dir "$METADATA_DIR" \
  --val-ratio 0.10 \
  --seed 42
```

This produces `train.json`, `val.json`, `test_fixed.json`,
`metadata_bundle.json`, and statistics. Preserve these files for reproducible
experiments and for the attack and robustification repositories.

### 3. Train the baseline MoE

```bash
python scripts/bdd100k_moe_train.py train \
  --data-root "$DATA_ROOT" \
  --metadata-dir "$METADATA_DIR" \
  --experiment-dir "$RUN_DIR" \
  --stage stage3_moe \
  --backbone-name convnextv2_tiny.fcmae_ft_in1k \
  --epochs 25 \
  --batch-size 32 \
  --num-workers 8 \
  --seed 42
```

Available stages are `stage0_baseline`, `stage1_condition`,
`stage2_experts_obj`, `stage3_moe`, and `stage4_finetune`. Continue from an
earlier checkpoint with `--resume /path/to/checkpoints/best.pt`.

Important outputs include:

```text
experiments/moe_stage3/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── config.json
└── history.json
```

### 4. Evaluate the fixed test set

```bash
python scripts/bdd100k_moe_test.py \
  --train-script scripts/bdd100k_moe_train.py \
  --metadata-dir "$METADATA_DIR" \
  --run-dirs "$RUN_DIR" \
  --checkpoint-name best.pt \
  --output-dir evaluation_outputs \
  --batch-size 64 \
  --device cuda:0
```

Use `--device cpu` if CUDA is unavailable. The evaluator writes per-run and
aggregate metrics, prediction CSV files, confusion matrices, ROC and
precision-recall curves, calibration plots, and router analyses. Multiple seed
directories may be supplied after `--run-dirs`.

## Outputs needed by the other repositories

The adversarial evaluation and robustification repositories require:

- `scripts/bdd100k_moe_train.py`, because they import the model architecture;
- the metadata directory, especially `metadata_bundle.json` and
  `test_fixed.json`;
- a trained run directory containing `checkpoints/best.pt` and `config.json`.

Pass these locations explicitly on the command line. Several original scripts
contain development-machine absolute defaults, which should not be relied upon.

## Reproducibility

Keep the metadata JSON files, seed, dependency versions, model configuration,
and checkpoint together. The test set is intentionally persisted in
`test_fixed.json` to prevent evaluation-time resampling.




