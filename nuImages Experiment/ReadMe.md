# nuImages Condition-Specialized Mixture of Experts

This repository contains the baseline training and evaluation pipeline for a
condition-specialized Mixture-of-Experts (MoE) model using nuImages camera data.
It performs **image-level object-presence classification** for cars,
pedestrians, and traffic cones; it is not an object detector or instance
segmentation model.

The model has three specialist experts and a learned router:

- **Location expert:** location from `log.location`.
- **Illumination expert:** illumination bucket derived from image timestamp.
- **Motion expert:** ego-motion bucket derived from `ego_pose.speed`.

The official nuImages validation split becomes a fixed held-out test set. A
validation subset is carved from the official training split for model
selection.

## Repository contents

```text
.
├── README.md
├── requirements.txt
└── scripts/
    ├── nuimages_moe_train.py
    └── nuimages_moe_test.py
```

- `nuimages_moe_train.py` prepares metadata and performs staged MoE training.
- `nuimages_moe_test.py` evaluates one or more checkpoints on the fixed test
  split and creates CSV reports and plots.

## Dataset download

Register or log in and download nuImages from the [official nuImages
page](https://www.nuscenes.org/nuimages). The official page states that the
dataset is available free of charge for non-commercial use and also links its
downloads, data format, tutorials, and devkit.

For a small setup check, the official tutorial provides the mini split:

```bash
wget https://www.nuscenes.org/data/nuimages-v1.0-mini.tgz
```

The full experiment requires the **training and validation images and metadata**,
not only the mini split. Do not commit downloaded images or metadata to GitHub.
Review the current [nuImages terms](https://www.nuscenes.org/terms-of-use) before
redistributing data or derived artifacts.

## Expected dataset layout

The training script reads the nuImages JSON tables directly; the nuImages devkit
is not required by the current implementation.

```text
/path/to/nuimages/
├── samples/
│   ├── CAM_BACK/
│   ├── CAM_FRONT/
│   └── ...
├── sweeps/                         # optional for this project
└── nuimages-metadata/
    ├── v1.0-train/
    │   ├── sample.json
    │   ├── sample_data.json
    │   ├── object_ann.json
    │   ├── category.json
    │   ├── ego_pose.json
    │   ├── log.json
    │   └── ...
    └── v1.0-val/
        ├── sample.json
        ├── sample_data.json
        ├── object_ann.json
        ├── category.json
        ├── ego_pose.json
        ├── log.json
        └── ...
```

If metadata was extracted directly as `v1.0-train/` and `v1.0-val/` beneath the
data root, adjust `--train-version` and `--val-version` accordingly.

## Installation

Python 3.10 or newer is recommended. A CUDA-capable NVIDIA GPU is strongly
recommended for training. Install the appropriate PyTorch build using the
[official PyTorch installer](https://pytorch.org/get-started/locally/), then:

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
```

## Run order

Run commands from the repository root and define portable local paths:

```bash
DATA_ROOT=/path/to/nuimages
METADATA_DIR=outputs/nuimages_metadata
RUN_DIR=experiments/nuimages_moe_stage3
```

On Windows PowerShell, replace these variables with full paths in each command.

### 1. Prepare deterministic metadata

This is the first script to run:

```bash
python scripts/nuimages_moe_train.py prepare \
  --data-root "$DATA_ROOT" \
  --train-version nuimages-metadata/v1.0-train \
  --val-version nuimages-metadata/v1.0-val \
  --output-dir "$METADATA_DIR" \
  --val-ratio 0.10 \
  --seed 42
```

The preparation step writes:

```text
outputs/nuimages_metadata/
├── metadata_bundle.json
├── train.json
├── val.json
├── test_fixed.json
└── stats.json
```

Inspect `stats.json` and several rows in each split before training. Paths saved
in the metadata must point to readable image files.

### 2. Train the baseline MoE

```bash
python scripts/nuimages_moe_train.py train \
  --metadata-dir "$METADATA_DIR" \
  --experiment-dir "$RUN_DIR" \
  --stage stage3_moe \
  --backbone-name convnextv2_tiny.fcmae_ft_in1k \
  --epochs 50 \
  --batch-size 32 \
  --num-workers 8 \
  --early-stopping-patience 10 \
  --seed 42
```

Reduce `--batch-size` if GPU memory is insufficient. Available stages are
`stage0_baseline`, `stage1_condition`, `stage2_experts_obj`, `stage3_moe`, and
`stage4_finetune`. Use `--resume /path/to/checkpoints/best.pt` to continue from
an earlier checkpoint.

Training produces `config.json`, `history.json`, and:

```text
experiments/nuimages_moe_stage3/checkpoints/
├── best.pt
└── last.pt
```

### 3. Evaluate the fixed test split

```bash
python scripts/nuimages_moe_test.py \
  --train-script scripts/nuimages_moe_train.py \
  --metadata-dir "$METADATA_DIR" \
  --run-dirs "$RUN_DIR" \
  --checkpoint-name best.pt \
  --output-dir evaluation_outputs \
  --batch-size 64 \
  --num-workers 8 \
  --device cuda:0
```

Use `--device cpu` when CUDA is unavailable. Supply multiple directories after
`--run-dirs` to aggregate independent seeds. Evaluation generates prediction
CSVs, per-run and aggregate metrics, confusion matrices, router analyses,
calibration plots, and ROC and precision-recall curves.

## Inputs needed by the other repositories

The attack-evaluation and robustification repositories require:

- `nuimages_moe_train.py`, because they import the architecture;
- `nuimages_moe_test.py`, because they reuse evaluation functions;
- the prepared metadata directory;
- a run directory containing `config.json` and `checkpoints/best.pt`;
- continued access to images referenced by the metadata.

Pass all paths explicitly. Several scripts retain absolute defaults from the
development machine and those defaults will not work in a fresh clone.

## Reproducibility and responsible publishing

Keep the fixed metadata, seed, dependency versions, configuration, and
checkpoint together. Do not commit `.venv/`, `__pycache__/`, raw nuImages data,
generated output folders, or large `.pt` files to ordinary Git history. Use Git
LFS or a release/artifact store if model weights need to be shared.


