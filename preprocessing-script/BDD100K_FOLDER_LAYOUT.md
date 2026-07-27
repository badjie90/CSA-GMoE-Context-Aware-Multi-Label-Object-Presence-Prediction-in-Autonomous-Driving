# BDD100K folder layout expected by the training scaffold

## Minimal expected layout

```text
/path/to/bdd100k/
├── images/
│   └── 100k/
│       ├── train/
│       │   ├── 0000f77c-6257be58.jpg
│       │   └── ...
│       └── val/
│           ├── 0000f77c-62c2a288.jpg
│           └── ...
└── labels/
    └── det_20/
        ├── det_train.json
        └── det_val.json
```

## Prepare command

```bash
python bdd100k_moe.py prepare \
  --data-root /path/to/bdd100k \
  --det-train-json /path/to/bdd100k/labels/det_20/det_train.json \
  --det-val-json /path/to/bdd100k/labels/det_20/det_val.json \
  --output-dir outputs/metadata \
  --seed 42
```

## Train command

```bash
python bdd100k_moe.py train \
  --metadata-dir outputs/metadata \
  --experiment-dir experiments/moe_stage3 \
  --stage stage3_moe \
  --epochs 25 \
  --batch-size 32
```

## Path resolution used by the script

The main script checks image paths in this order:

1. `data_root/images/100k/{split}/{image_name}`
2. `data_root/images/{split}/{image_name}`
3. `data_root/{split}/{image_name}`

## Required local files

- training images
- validation images
- `det_train.json`
- `det_val.json`

## Reproducible test artifact

The prepare step writes:

- `train.json`
- `val.json`
- `test_fixed.json`
- `metadata_bundle.json`
