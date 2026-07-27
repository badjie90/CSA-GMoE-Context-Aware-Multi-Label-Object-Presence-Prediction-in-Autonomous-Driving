#!/usr/bin/env python3
# path: check_bdd100k_paths.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_image_path(data_root: Path, split: str, image_name: str) -> Path:
    candidates = [
        data_root / "images" / "100k" / split / image_name,
        data_root / "images" / split / image_name,
        data_root / split / image_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def check_json_samples(data_root: Path, split: str, json_path: Path, num_samples: int) -> Dict[str, Any]:
    frames = load_json(json_path)
    checked = 0
    found = 0
    missing_examples: List[str] = []

    for frame in frames[:num_samples]:
        image_name = frame.get("name") or Path(frame.get("url", "")).name
        if not image_name:
            continue

        checked += 1
        image_path = infer_image_path(data_root, split, image_name)
        if image_path.exists():
            found += 1
        elif len(missing_examples) < 10:
            missing_examples.append(str(image_path))

    return {
        "split": split,
        "json_path": str(json_path),
        "checked_samples": checked,
        "found_samples": found,
        "missing_samples": checked - found,
        "missing_examples": missing_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local BDD100K paths")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--det-train-json", type=str, required=True)
    parser.add_argument("--det-val-json", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    det_train_json = Path(args.det_train_json)
    det_val_json = Path(args.det_val_json)

    print(f"data_root exists: {data_root.exists()} -> {data_root}")
    print(f"det_train_json exists: {det_train_json.exists()} -> {det_train_json}")
    print(f"det_val_json exists: {det_val_json.exists()} -> {det_val_json}")

    train_dir = data_root / "images" / "100k" / "train"
    val_dir = data_root / "images" / "100k" / "val"

    print(f"train image dir exists: {train_dir.exists()} -> {train_dir}")
    print(f"val image dir exists: {val_dir.exists()} -> {val_dir}")

    if det_train_json.exists():
        train_report = check_json_samples(
            data_root=data_root,
            split="train",
            json_path=det_train_json,
            num_samples=args.num_samples,
        )
        print("\\nTRAIN SAMPLE CHECK")
        for key, value in train_report.items():
            print(f"{key}: {value}")

    if det_val_json.exists():
        val_report = check_json_samples(
            data_root=data_root,
            split="val",
            json_path=det_val_json,
            num_samples=args.num_samples,
        )
        print("\\nVAL SAMPLE CHECK")
        for key, value in val_report.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
