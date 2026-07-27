#!/usr/bin/env python3
# path: download_bdd100k_kagglehub.py

from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub


def find_data_root(download_path: Path) -> Path:
    candidates = [
        download_path / "bdd100k" / "bdd100k",
        download_path / "bdd100k",
        download_path,
    ]
    for candidate in candidates:
        if (candidate / "images").exists() and (candidate / "labels").exists():
            return candidate
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BDD100K from KaggleHub and print next commands")
    parser.add_argument("--dataset", type=str, default="awsaf49/bdd100k-dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/metadata")
    parser.add_argument("--experiment-dir", type=str, default="experiments/moe_stage3")
    parser.add_argument("--stage", type=str, default="stage3_moe")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-script", type=str, default="bdd100k_moe.py")
    parser.add_argument("--checker-script", type=str, default="check_bdd100k_paths.py")
    args = parser.parse_args()

    download_path = Path(kagglehub.dataset_download(args.dataset)).resolve()
    data_root = find_data_root(download_path)

    labels_dir = data_root / "labels"
    det_train_json = labels_dir / "det_v2_train_release.json"
    det_val_json = labels_dir / "det_v2_val_release.json"

    print("\nDownload complete")
    print(f"download_path: {download_path}")
    print(f"data_root: {data_root}")
    print(f"train_json_exists: {det_train_json.exists()} -> {det_train_json}")
    print(f"val_json_exists: {det_val_json.exists()} -> {det_val_json}")

    print("\nRun this checker first:")
    print(
        f'python "{args.checker_script}" '
        f'--data-root "{data_root}" '
        f'--det-train-json "{det_train_json}" '
        f'--det-val-json "{det_val_json}"'
    )

    print("\nThen run metadata preparation:")
    print(
        f'python "{args.training_script}" prepare '
        f'--data-root "{data_root}" '
        f'--det-train-json "{det_train_json}" '
        f'--det-val-json "{det_val_json}" '
        f'--output-dir "{args.output_dir}" '
        f'--seed 42'
    )

    print("\nThen run training:")
    print(
        f'python "{args.training_script}" train '
        f'--metadata-dir "{args.output_dir}" '
        f'--experiment-dir "{args.experiment_dir}" '
        f'--stage "{args.stage}" '
        f'--epochs {args.epochs} '
        f'--batch-size {args.batch_size}'
    )


if __name__ == "__main__":
    main()
