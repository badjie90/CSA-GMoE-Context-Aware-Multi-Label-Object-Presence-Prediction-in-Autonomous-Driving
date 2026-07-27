#!/usr/bin/env python3
# path: bdd100k_moe_test.py
"""
BDD100K Condition-Specialized MoE Test/Evaluation Script

Purpose
-------
Evaluate trained checkpoints on the fixed held-out test split only.

This script is designed for thesis reporting:
- evaluates 5 independent runs / seeds
- computes per-run metrics and aggregated statistics
- saves metrics to CSV
- saves plots to PNG
- analyzes:
  1) fused MoE output
  2) router behavior
  3) each specialist expert
  4) condition heads

Expected inputs
---------------
- metadata directory containing:
  - metadata_bundle.json
  - test_fixed.json
- one checkpoint per run, typically:
  - <run_dir>/checkpoints/best.pt

Design notes
------------
- The evaluation uses the fixed test set only.
- Training/validation are intentionally excluded here.
- The script imports architecture helpers from the train script so model
  definition stays consistent with the training code.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageFile
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

TARGET_OBJECTS = ["car", "pedestrian", "traffic_sign"]
EXPERT_NAMES = ["weather", "scene", "time"]
EXPERT_INDEX_TO_NAME = {0: "weather", 1: "scene", 2: "time"}

DEFAULT_TRAIN_SCRIPT = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/BDD100k/scripts/bdd100k_moe_train.py"
DEFAULT_METADATA_DIR = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/BDD100k/data/metadata_files/metadata-New"
DEFAULT_RUN_GLOB = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/BDD100k/New-train-models/moe_stage3"
DEFAULT_OUTPUT_DIR = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/BDD100k/evaluation_outputs-New"


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    import json
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    import json
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    rows = list(rows)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def import_train_module(train_script: Path):
    import sys

    module_name = "bdd100k_moe_train_module"
    spec = importlib.util.spec_from_file_location(module_name, train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import train module from: {train_script}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module




def list_run_dirs(run_dirs: Sequence[str], run_glob: str) -> List[Path]:
    resolved: List[Path] = []

    if run_dirs:
        resolved.extend(Path(p).resolve() for p in run_dirs)
    else:
        glob_path = Path(run_glob).expanduser()

        if glob_path.is_absolute():
            parent = glob_path.parent
            pattern = glob_path.name
            resolved.extend(sorted(parent.glob(pattern)))
        else:
            resolved.extend(sorted(Path(".").glob(run_glob)))

    resolved = [p.resolve() for p in resolved if p.exists()]
    if not resolved:
        raise FileNotFoundError(
            f"No run directories found. Pass --run-dirs or a valid --run-glob. Got: {run_glob}"
        )
    return resolved



def get_t_critical_95(n: int) -> float:
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }
    if n <= 1:
        return 0.0
    return table.get(n, 1.96)


def mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    half = 0.0
    if arr.size > 1:
        half = get_t_critical_95(arr.size) * (std / math.sqrt(arr.size))
    return mean, std, mean - half, mean + half


def safe_metric(fn, *args, **kwargs) -> float:
    try:
        value = fn(*args, **kwargs)
        if isinstance(value, np.ndarray):
            return float(np.mean(value))
        return float(value)
    except Exception:
        return float("nan")


def multilabel_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        scores.append(balanced_accuracy_score(y_true[:, c], y_pred[:, c]))
    return float(np.mean(scores)) if scores else float("nan")


def macro_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    scores = []
    for c in range(y_true.shape[1]):
        if np.sum(y_true[:, c]) == 0:
            continue
        scores.append(average_precision_score(y_true[:, c], y_prob[:, c]))
    return float(np.mean(scores)) if scores else float("nan")


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    scores = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        scores.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
    return float(np.mean(scores)) if scores else float("nan")


def micro_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return safe_metric(average_precision_score, y_true.ravel(), y_prob.ravel())


def micro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true.ravel())) < 2:
        return float("nan")
    return safe_metric(roc_auc_score, y_true.ravel(), y_prob.ravel())


def expected_calibration_error_binary(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    y_true = y_true.astype(np.float64)
    y_prob = y_prob.astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= start) & (y_prob < end if end < 1.0 else y_prob <= end)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += np.abs(acc - conf) * (mask.sum() / max(n, 1))
    return float(ece)


def multilabel_ece(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    scores = []
    for c in range(y_true.shape[1]):
        scores.append(expected_calibration_error_binary(y_true[:, c], y_prob[:, c], bins=bins))
    return float(np.mean(scores)) if scores else float("nan")


def brier_score_multilabel(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_true.astype(np.float64) - y_prob.astype(np.float64)) ** 2))


def infer_batch_time_ms(model, images: torch.Tensor, device: torch.device, amp_dtype: torch.dtype) -> float:
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        starter.record()
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=True):
                _ = model(images)
        ender.record()
        torch.cuda.synchronize(device)
        elapsed_ms = starter.elapsed_time(ender)
        return float(elapsed_ms)
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(images)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return float(elapsed_ms)


def make_test_loader(train_module, metadata_dir: Path, image_size: int, batch_size: int, num_workers: int):
    bundle = load_json(metadata_dir / "metadata_bundle.json")
    test_rows = load_json(metadata_dir / "test_fixed.json")
    _, eval_transform = train_module.build_transforms(image_size=image_size)
    dataset = train_module.BDD100KPresenceDataset(test_rows, eval_transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=train_module.seed_worker if hasattr(train_module, "seed_worker") else None,
    )
    return loader, bundle, test_rows


def numpy_from_list(chunks: List[np.ndarray], ndim: int = 2) -> np.ndarray:
    if not chunks:
        shape = (0,) if ndim == 1 else (0, 0)
        return np.empty(shape, dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def aggregate_confusion_matrices(mats: Sequence[np.ndarray]) -> np.ndarray:
    if not mats:
        return np.empty((0, 0), dtype=np.float64)
    return np.mean(np.stack(mats, axis=0), axis=0)











def _style_ticks(ax, tick_fontsize: int) -> None:
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")


def _style_legend(ax, legend_fontsize: int) -> None:
    legend = ax.legend(fontsize=legend_fontsize)
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontweight("bold")


def _style_axes(
    ax,
    title: str,
    xlabel: str,
    ylabel: str,
    plot_cfg: Dict[str, Any],
    use_legend: bool = False,
) -> None:
    ax.set_title(title, fontsize=plot_cfg["title_fontsize"], fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=plot_cfg["label_fontsize"], fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=plot_cfg["label_fontsize"], fontweight="bold")
    _style_ticks(ax, plot_cfg["tick_fontsize"])
    if use_legend:
        _style_legend(ax, plot_cfg["legend_fontsize"])


def plot_matrix(
    matrix: np.ndarray,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    title: str,
    path: Path,
    plot_cfg: Dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=plot_cfg["tick_fontsize"])
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)

    _style_axes(ax, title=title, xlabel="", ylabel="", plot_cfg=plot_cfg, use_legend=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_errorbar_summary(
    rows: Sequence[Dict[str, Any]],
    metric_key: str,
    title: str,
    path: Path,
    plot_cfg: Dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    labels = [row["name"] for row in rows]
    means = [row[f"{metric_key}_mean"] for row in rows]
    stds = [row[f"{metric_key}_std"] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(range(len(labels)), means, yerr=stds, fmt="o", linewidth=plot_cfg["linewidth"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    _style_axes(ax, title=title, xlabel="", ylabel=metric_key, plot_cfg=plot_cfg, use_legend=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    path: Path,
    plot_cfg: Dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(10, 10))

    for c, label in enumerate(TARGET_OBJECTS):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        frac_pos, mean_pred = calibration_curve(y_true[:, c], y_prob[:, c], n_bins=15, strategy="uniform")
        ax.plot(
            mean_pred,
            frac_pos,
            marker="o",
            linewidth=plot_cfg["linewidth"],
            label=label,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="ideal")
    _style_axes(
        ax,
        title=title,
        xlabel="Predicted probability",
        ylabel="Observed frequency",
        plot_cfg=plot_cfg,
        use_legend=True,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multilabel_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    title: str,
    path: Path,
    plot_cfg: Dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(12, 10))

    for c, class_name in enumerate(class_names):
        if np.sum(y_true[:, c]) == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_true[:, c], y_prob[:, c])
        ap = average_precision_score(y_true[:, c], y_prob[:, c])
        ax.plot(
            recall,
            precision,
            linewidth=plot_cfg["linewidth"],
            label=f"{class_name} (AP={ap:.3f})",
        )

    _style_axes(
        ax,
        title=title,
        xlabel="Recall",
        ylabel="Precision",
        plot_cfg=plot_cfg,
        use_legend=True,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)






def plot_mean_multirun_pr_curves(
    curve_payloads: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    title: str,
    path: Path,
    plot_cfg: Dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    recall_grid = np.linspace(0.0, 1.0, 201)

    fig, ax = plt.subplots(figsize=(12, 10))

    for c, class_name in enumerate(class_names):
        precision_runs: List[np.ndarray] = []
        ap_runs: List[float] = []

        for payload in curve_payloads:
            y_true = payload["y_true"][:, c]
            y_prob = payload["y_prob"][:, c]

            if np.sum(y_true) == 0:
                continue

            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)

            order = np.argsort(recall)
            recall_sorted = recall[order]
            precision_sorted = precision[order]

            interp_precision = np.interp(recall_grid, recall_sorted, precision_sorted)
            precision_runs.append(interp_precision)
            ap_runs.append(ap)

        if not precision_runs:
            continue

        precision_runs_np = np.vstack(precision_runs)
        mean_precision = precision_runs_np.mean(axis=0)

        if precision_runs_np.shape[0] > 1:
            std_precision = precision_runs_np.std(axis=0, ddof=1)
            ci_precision = 1.96 * std_precision / np.sqrt(precision_runs_np.shape[0])
        else:
            ci_precision = np.zeros_like(mean_precision)

        line = ax.plot(
            recall_grid,
            mean_precision,
            linewidth=plot_cfg["linewidth"],
            label=f"{class_name} (mean AP={np.mean(ap_runs):.3f})",
        )[0]

        ax.fill_between(
            recall_grid,
            np.clip(mean_precision - ci_precision, 0.0, 1.0),
            np.clip(mean_precision + ci_precision, 0.0, 1.0),
            alpha=0.20,
            color=line.get_color(),
        )

    _style_axes(
        ax,
        title=title,
        xlabel="Recall",
        ylabel="Precision",
        plot_cfg=plot_cfg,
        use_legend=True,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mean_multirun_roc_curves(
    curve_payloads: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    title: str,
    path: Path,
    plot_cfg: Dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    fpr_grid = np.linspace(0.0, 1.0, 201)

    fig, ax = plt.subplots(figsize=(12, 10))

    for c, class_name in enumerate(class_names):
        tpr_runs: List[np.ndarray] = []
        auc_runs: List[float] = []

        for payload in curve_payloads:
            y_true = payload["y_true"][:, c]
            y_prob = payload["y_prob"][:, c]

            if len(np.unique(y_true)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)

            interp_tpr = np.interp(fpr_grid, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tpr[-1] = 1.0

            tpr_runs.append(interp_tpr)
            auc_runs.append(auc)

        if not tpr_runs:
            continue

        tpr_runs_np = np.vstack(tpr_runs)
        mean_tpr = tpr_runs_np.mean(axis=0)

        if tpr_runs_np.shape[0] > 1:
            std_tpr = tpr_runs_np.std(axis=0, ddof=1)
            ci_tpr = 1.96 * std_tpr / np.sqrt(tpr_runs_np.shape[0])
        else:
            ci_tpr = np.zeros_like(mean_tpr)

        line = ax.plot(
            fpr_grid,
            mean_tpr,
            linewidth=plot_cfg["linewidth"],
            label=f"{class_name} (mean AUC={np.mean(auc_runs):.3f})",
        )[0]

        ax.fill_between(
            fpr_grid,
            np.clip(mean_tpr - ci_tpr, 0.0, 1.0),
            np.clip(mean_tpr + ci_tpr, 0.0, 1.0),
            alpha=0.20,
            color=line.get_color(),
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="chance")

    _style_axes(
        ax,
        title=title,
        xlabel="False positive rate",
        ylabel="True positive rate",
        plot_cfg=plot_cfg,
        use_legend=True,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multilabel_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    title: str,
    path: Path,
    plot_cfg: Dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(12, 10))

    for c, class_name in enumerate(class_names):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, c], y_prob[:, c])
        auc = roc_auc_score(y_true[:, c], y_prob[:, c])
        ax.plot(
            fpr,
            tpr,
            linewidth=plot_cfg["linewidth"],
            label=f"{class_name} (AUC={auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="chance")
    _style_axes(
        ax,
        title=title,
        xlabel="False positive rate",
        ylabel="True positive rate",
        plot_cfg=plot_cfg,
        use_legend=True,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)




def collect_predictions(
    model: torch.nn.Module,
    loader,
    train_module,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Dict[str, Any]:
    model.eval()

    y_true_chunks: List[np.ndarray] = []
    weather_id_chunks: List[np.ndarray] = []
    scene_id_chunks: List[np.ndarray] = []
    time_id_chunks: List[np.ndarray] = []
    weather_mask_chunks: List[np.ndarray] = []
    scene_mask_chunks: List[np.ndarray] = []
    time_mask_chunks: List[np.ndarray] = []

    probs_fused_chunks: List[np.ndarray] = []
    probs_weather_chunks: List[np.ndarray] = []
    probs_scene_chunks: List[np.ndarray] = []
    probs_time_chunks: List[np.ndarray] = []

    weather_logits_chunks: List[np.ndarray] = []
    scene_logits_chunks: List[np.ndarray] = []
    time_logits_chunks: List[np.ndarray] = []
    alpha_chunks: List[np.ndarray] = []

    image_paths: List[str] = []
    batch_ms: List[float] = []
    num_images = 0

    with torch.no_grad():
        for batch in loader:
            image_paths.extend(batch["image_path"])
            batch = train_module.move_batch_to_device(batch, device)
            images = batch["image"]

            ms = infer_batch_time_ms(model, images, device, amp_dtype)
            batch_ms.append(ms)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                outputs = model(images)

            y_true_chunks.append(batch["objects"].detach().cpu().numpy())
            weather_id_chunks.append(batch["weather_id"].detach().cpu().numpy())
            scene_id_chunks.append(batch["scene_id"].detach().cpu().numpy())
            time_id_chunks.append(batch["time_id"].detach().cpu().numpy())
            weather_mask_chunks.append(batch["weather_mask"].detach().cpu().numpy())
            scene_mask_chunks.append(batch["scene_mask"].detach().cpu().numpy())
            time_mask_chunks.append(batch["time_mask"].detach().cpu().numpy())

            probs_fused_chunks.append(torch.sigmoid(outputs["obj_fused_logits"]).detach().cpu().numpy())

            if "obj_w_logits" in outputs:
                probs_weather_chunks.append(torch.sigmoid(outputs["obj_w_logits"]).detach().cpu().numpy())
                probs_scene_chunks.append(torch.sigmoid(outputs["obj_s_logits"]).detach().cpu().numpy())
                probs_time_chunks.append(torch.sigmoid(outputs["obj_t_logits"]).detach().cpu().numpy())

                weather_logits_chunks.append(outputs["weather_logits"].detach().cpu().numpy())
                scene_logits_chunks.append(outputs["scene_logits"].detach().cpu().numpy())
                time_logits_chunks.append(outputs["time_logits"].detach().cpu().numpy())
                alpha_chunks.append(outputs["alpha"].detach().cpu().numpy())

            num_images += images.shape[0]

    per_image_ms = float(np.sum(batch_ms) / max(num_images, 1))

    return {
        "image_path": image_paths,
        "y_true": numpy_from_list(y_true_chunks),
        "weather_id": numpy_from_list(weather_id_chunks, ndim=1),
        "scene_id": numpy_from_list(scene_id_chunks, ndim=1),
        "time_id": numpy_from_list(time_id_chunks, ndim=1),
        "weather_mask": numpy_from_list(weather_mask_chunks, ndim=1),
        "scene_mask": numpy_from_list(scene_mask_chunks, ndim=1),
        "time_mask": numpy_from_list(time_mask_chunks, ndim=1),
        "probs_fused": numpy_from_list(probs_fused_chunks),
        "probs_weather": numpy_from_list(probs_weather_chunks),
        "probs_scene": numpy_from_list(probs_scene_chunks),
        "probs_time": numpy_from_list(probs_time_chunks),
        "weather_logits": numpy_from_list(weather_logits_chunks),
        "scene_logits": numpy_from_list(scene_logits_chunks),
        "time_logits": numpy_from_list(time_logits_chunks),
        "alpha": numpy_from_list(alpha_chunks),
        "inference_ms_per_image": per_image_ms,
    }


def compute_multilabel_metrics(name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)

    row = {
        "name": name,
        "micro_precision": safe_metric(precision_score, y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": safe_metric(precision_score, y_true, y_pred, average="macro", zero_division=0),
        "micro_recall": safe_metric(recall_score, y_true, y_pred, average="micro", zero_division=0),
        "macro_recall": safe_metric(recall_score, y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": safe_metric(f1_score, y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": safe_metric(f1_score, y_true, y_pred, average="macro", zero_division=0),
        "accuracy": safe_metric(accuracy_score, y_true, y_pred),
        "balanced_accuracy": multilabel_balanced_accuracy(y_true, y_pred),
        "mAP": macro_average_precision(y_true, y_prob),
        "micro_mAP": micro_average_precision(y_true, y_prob),
        "macro_auroc": macro_auroc(y_true, y_prob),
        "micro_auroc": micro_auroc(y_true, y_prob),
        "ece": multilabel_ece(y_true, y_prob),
        "brier": brier_score_multilabel(y_true, y_prob),
    }
    return row


def compute_per_class_object_metrics(head_name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> List[Dict[str, Any]]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    rows: List[Dict[str, Any]] = []
    for c, obj_name in enumerate(TARGET_OBJECTS):
        row = {
            "head": head_name,
            "object": obj_name,
            "precision": safe_metric(precision_score, y_true[:, c], y_pred[:, c], zero_division=0),
            "recall": safe_metric(recall_score, y_true[:, c], y_pred[:, c], zero_division=0),
            "f1": safe_metric(f1_score, y_true[:, c], y_pred[:, c], zero_division=0),
            "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true[:, c], y_pred[:, c]) if len(np.unique(y_true[:, c])) > 1 else float("nan"),
            "ap": safe_metric(average_precision_score, y_true[:, c], y_prob[:, c]) if np.sum(y_true[:, c]) > 0 else float("nan"),
            "auroc": safe_metric(roc_auc_score, y_true[:, c], y_prob[:, c]) if len(np.unique(y_true[:, c])) > 1 else float("nan"),
        }
        rows.append(row)
    return rows


def compute_condition_metrics(task_name: str, logits: np.ndarray, y_true: np.ndarray, mask: np.ndarray, class_names: Sequence[str]) -> Tuple[Dict[str, float], np.ndarray]:
    valid = mask.astype(bool)
    if not np.any(valid):
        empty_matrix = np.zeros((len(class_names), len(class_names)), dtype=np.float64)
        return {
            "task": task_name,
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "balanced_accuracy": float("nan"),
        }, empty_matrix

    pred = np.argmax(logits[valid], axis=1)
    target = y_true[valid].astype(np.int64)
    cm = confusion_matrix(target, pred, labels=np.arange(len(class_names)))
    row = {
        "task": task_name,
        "accuracy": safe_metric(accuracy_score, target, pred),
        "macro_f1": safe_metric(f1_score, target, pred, average="macro", zero_division=0),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, target, pred),
    }
    return row, cm.astype(np.float64)







def compute_router_per_class_rows(preds: Dict[str, Any]) -> List[Dict[str, Any]]:
    alpha = preds["alpha"]
    if alpha.size == 0:
        return []

    top1 = np.argmax(alpha, axis=1)
    rows: List[Dict[str, Any]] = []

    for obj_id, obj_name in enumerate(TARGET_OBJECTS):
        mask = preds["y_true"][:, obj_id] == 1
        if not np.any(mask):
            continue

        alpha_subset = alpha[mask]
        mean_alpha = alpha_subset.mean(axis=0)
        entropy = -(alpha_subset * np.log(alpha_subset + 1e-12)).sum(axis=1)
        normalized_entropy = entropy / np.log(alpha_subset.shape[1])

        rows.append(
            {
                "object": obj_name,
                "alpha_weather": float(mean_alpha[0]),
                "alpha_scene": float(mean_alpha[1]),
                "alpha_time": float(mean_alpha[2]),
                "top1_weather_fraction": float(np.mean(top1[mask] == 0)),
                "top1_scene_fraction": float(np.mean(top1[mask] == 1)),
                "top1_time_fraction": float(np.mean(top1[mask] == 2)),
                "entropy": float(entropy.mean()),
                "normalized_entropy": float(normalized_entropy.mean()),
                "collapse_score": float(mean_alpha.max()),
            }
        )

    return rows


def compute_router_condition_rows(
    preds: Dict[str, Any],
    weather_names: Sequence[str],
    scene_names: Sequence[str],
    time_names: Sequence[str],
) -> List[Dict[str, Any]]:
    alpha = preds["alpha"]
    if alpha.size == 0:
        return []

    top1 = np.argmax(alpha, axis=1)
    rows: List[Dict[str, Any]] = []

    condition_specs = [
        ("weather", preds["weather_id"], preds["weather_mask"], weather_names),
        ("scene", preds["scene_id"], preds["scene_mask"], scene_names),
        ("time", preds["time_id"], preds["time_mask"], time_names),
    ]

    for task_name, ids, masks, class_names in condition_specs:
        for class_id, class_name in enumerate(class_names):
            mask = (masks > 0) & (ids == class_id)
            if not np.any(mask):
                continue

            alpha_subset = alpha[mask]
            mean_alpha = alpha_subset.mean(axis=0)
            entropy = -(alpha_subset * np.log(alpha_subset + 1e-12)).sum(axis=1)
            normalized_entropy = entropy / np.log(alpha_subset.shape[1])

            rows.append(
                {
                    "task": task_name,
                    "subset": class_name,
                    "alpha_weather": float(mean_alpha[0]),
                    "alpha_scene": float(mean_alpha[1]),
                    "alpha_time": float(mean_alpha[2]),
                    "top1_weather_fraction": float(np.mean(top1[mask] == 0)),
                    "top1_scene_fraction": float(np.mean(top1[mask] == 1)),
                    "top1_time_fraction": float(np.mean(top1[mask] == 2)),
                    "entropy": float(entropy.mean()),
                    "normalized_entropy": float(normalized_entropy.mean()),
                    "collapse_score": float(mean_alpha.max()),
                }
            )

    return rows


def compute_router_metrics(preds: Dict[str, Any], weather_names: Sequence[str], scene_names: Sequence[str], time_names: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    alpha = preds["alpha"]
    if alpha.size == 0:
        return [], {}

    top1 = np.argmax(alpha, axis=1)
    mean_alpha = alpha.mean(axis=0)
    entropy = -(alpha * np.log(alpha + 1e-12)).sum(axis=1)
    normalized_entropy = entropy / np.log(alpha.shape[1])

    summary = [
        {"name": "router_mean_alpha_weather", "value": float(mean_alpha[0])},
        {"name": "router_mean_alpha_scene", "value": float(mean_alpha[1])},
        {"name": "router_mean_alpha_time", "value": float(mean_alpha[2])},
        {"name": "router_entropy", "value": float(entropy.mean())},
        {"name": "router_normalized_entropy", "value": float(normalized_entropy.mean())},
        {"name": "router_collapse_score", "value": float(mean_alpha.max())},
        {"name": "router_top1_weather_fraction", "value": float(np.mean(top1 == 0))},
        {"name": "router_top1_scene_fraction", "value": float(np.mean(top1 == 1))},
        {"name": "router_top1_time_fraction", "value": float(np.mean(top1 == 2))},
    ]

    detailed: Dict[str, List[Dict[str, Any]]] = {
        "by_weather": [],
        "by_scene": [],
        "by_time": [],
        "by_object_presence": [],
    }

    for label_id, label_name in enumerate(weather_names):
        mask = (preds["weather_mask"] > 0) & (preds["weather_id"] == label_id)
        if np.any(mask):
            row = {"group": label_name}
            for i, expert_name in enumerate(EXPERT_NAMES):
                row[f"alpha_{expert_name}"] = float(alpha[mask].mean(axis=0)[i])
            detailed["by_weather"].append(row)

    for label_id, label_name in enumerate(scene_names):
        mask = (preds["scene_mask"] > 0) & (preds["scene_id"] == label_id)
        if np.any(mask):
            row = {"group": label_name}
            for i, expert_name in enumerate(EXPERT_NAMES):
                row[f"alpha_{expert_name}"] = float(alpha[mask].mean(axis=0)[i])
            detailed["by_scene"].append(row)

    for label_id, label_name in enumerate(time_names):
        mask = (preds["time_mask"] > 0) & (preds["time_id"] == label_id)
        if np.any(mask):
            row = {"group": label_name}
            for i, expert_name in enumerate(EXPERT_NAMES):
                row[f"alpha_{expert_name}"] = float(alpha[mask].mean(axis=0)[i])
            detailed["by_time"].append(row)

    for obj_id, obj_name in enumerate(TARGET_OBJECTS):
        for state in [0, 1]:
            mask = preds["y_true"][:, obj_id] == state
            if np.any(mask):
                row = {"group": f"{obj_name}_{state}"}
                for i, expert_name in enumerate(EXPERT_NAMES):
                    row[f"alpha_{expert_name}"] = float(alpha[mask].mean(axis=0)[i])
                detailed["by_object_presence"].append(row)

    return summary, detailed


def compute_specialist_metrics(preds: Dict[str, Any], weather_names: Sequence[str], scene_names: Sequence[str], time_names: Sequence[str], threshold: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    head_map = {
        "weather": preds["probs_weather"],
        "scene": preds["probs_scene"],
        "time": preds["probs_time"],
    }
    subset_spec = {
        "weather": (preds["weather_id"], preds["weather_mask"], weather_names),
        "scene": (preds["scene_id"], preds["scene_mask"], scene_names),
        "time": (preds["time_id"], preds["time_mask"], time_names),
    }

    for expert_name, probs in head_map.items():
        if probs.size == 0:
            continue

        global_metrics = compute_multilabel_metrics(f"{expert_name}_expert_object_head", preds["y_true"], probs, threshold)
        rows.append({"expert": expert_name, "subset": "global", **{k: v for k, v in global_metrics.items() if k != "name"}})

        labels, mask, class_names = subset_spec[expert_name]
        for label_id, label_name in enumerate(class_names):
            subset_mask = (mask > 0) & (labels == label_id)
            if not np.any(subset_mask):
                continue
            subset_metrics = compute_multilabel_metrics(
                f"{expert_name}_{label_name}",
                preds["y_true"][subset_mask],
                probs[subset_mask],
                threshold,
            )
            rows.append(
                {
                    "expert": expert_name,
                    "subset": label_name,
                    **{k: v for k, v in subset_metrics.items() if k != "name"},
                }
            )

    if preds["alpha"].size > 0:
        top1 = np.argmax(preds["alpha"], axis=1)
        fused_pred = (preds["probs_fused"] >= threshold).astype(np.int64)
        for idx, expert_name in enumerate(EXPERT_NAMES):
            routed_mask = top1 == idx
            if not np.any(routed_mask):
                continue
            rows.append(
                {
                    "expert": expert_name,
                    "subset": "top1_routed_fused_contribution",
                    "micro_precision": safe_metric(
                        precision_score,
                        preds["y_true"][routed_mask],
                        fused_pred[routed_mask],
                        average="micro",
                        zero_division=0,
                    ),
                    "macro_precision": safe_metric(
                        precision_score,
                        preds["y_true"][routed_mask],
                        fused_pred[routed_mask],
                        average="macro",
                        zero_division=0,
                    ),
                    "micro_recall": safe_metric(
                        recall_score,
                        preds["y_true"][routed_mask],
                        fused_pred[routed_mask],
                        average="micro",
                        zero_division=0,
                    ),
                    "macro_recall": safe_metric(
                        recall_score,
                        preds["y_true"][routed_mask],
                        fused_pred[routed_mask],
                        average="macro",
                        zero_division=0,
                    ),
                    "micro_f1": safe_metric(
                        f1_score,
                        preds["y_true"][routed_mask],
                        fused_pred[routed_mask],
                        average="micro",
                        zero_division=0,
                    ),
                    "macro_f1": safe_metric(
                        f1_score,
                        preds["y_true"][routed_mask],
                        fused_pred[routed_mask],
                        average="macro",
                        zero_division=0,
                    ),
                    "accuracy": safe_metric(accuracy_score, preds["y_true"][routed_mask], fused_pred[routed_mask]),
                    "balanced_accuracy": multilabel_balanced_accuracy(preds["y_true"][routed_mask], fused_pred[routed_mask]),
                    "mAP": macro_average_precision(preds["y_true"][routed_mask], preds["probs_fused"][routed_mask]),
                    "micro_mAP": micro_average_precision(preds["y_true"][routed_mask], preds["probs_fused"][routed_mask]),
                    "macro_auroc": macro_auroc(preds["y_true"][routed_mask], preds["probs_fused"][routed_mask]),
                    "micro_auroc": micro_auroc(preds["y_true"][routed_mask], preds["probs_fused"][routed_mask]),
                    "ece": multilabel_ece(preds["y_true"][routed_mask], preds["probs_fused"][routed_mask]),
                    "brier": brier_score_multilabel(preds["y_true"][routed_mask], preds["probs_fused"][routed_mask]),
                }
            )

    return rows


def summarize_table(rows: Sequence[Dict[str, Any]], key_cols: Sequence[str]) -> List[Dict[str, Any]]:
    rows = list(rows)
    if not rows:
        return []

    metric_cols = [
        col for col in rows[0].keys()
        if col not in key_cols and isinstance(rows[0][col], (int, float, np.floating)) and col != "seed"
    ]

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[col] for col in key_cols)
        grouped[key].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key, group_rows in grouped.items():
        summary_row = {col: value for col, value in zip(key_cols, key)}
        for metric in metric_cols:
            values = [float(r[metric]) for r in group_rows if not (isinstance(r[metric], float) and math.isnan(r[metric]))]
            mean, std, ci_low, ci_high = mean_std_ci(values)
            summary_row[f"{metric}_mean"] = mean
            summary_row[f"{metric}_std"] = std
            summary_row[f"{metric}_ci_low"] = ci_low
            summary_row[f"{metric}_ci_high"] = ci_high
        summary_rows.append(summary_row)
    return summary_rows




def evaluate_one_run(
    run_dir: Path,
    train_module,
    metadata_dir: Path,
    checkpoint_name: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    threshold: float,
    plots_dir: Path,
    plot_cfg: Dict[str, Any],
) -> Dict[str, Any]:



    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    config = load_json(config_path)
    image_size = int(config.get("image_size", 224))
    stage = str(config.get("stage", "stage3_moe"))
    backbone_name = str(config.get("backbone_name", "convnextv2_tiny.fcmae_ft_in1k"))
    no_pretrained = bool(config.get("no_pretrained", False))
    seed = int(config.get("seed", -1))

    loader, bundle, _ = make_test_loader(
        train_module=train_module,
        metadata_dir=metadata_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    weather_names = [k for k, _ in sorted(bundle["weather_to_id"].items(), key=lambda kv: kv[1])]
    scene_names = [k for k, _ in sorted(bundle["scene_to_id"].items(), key=lambda kv: kv[1])]
    time_names = [k for k, _ in sorted(bundle["time_to_id"].items(), key=lambda kv: kv[1])]

    model = train_module.build_model(
        stage=stage,
        backbone_name=backbone_name,
        num_weather=len(weather_names),
        num_scene=len(scene_names),
        num_time=len(time_names),
        pretrained=not no_pretrained,
    )
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device)
    model.eval()

    preds = collect_predictions(
        model=model,
        loader=loader,
        train_module=train_module,
        device=device,
        amp_dtype=amp_dtype,
    )

    fused_summary = compute_multilabel_metrics("fused_moe", preds["y_true"], preds["probs_fused"], threshold)
    fused_summary["inference_ms"] = preds["inference_ms_per_image"]
    fused_summary["seed"] = seed
    fused_summary["run_dir"] = str(run_dir)

    fused_per_class = compute_per_class_object_metrics("fused_moe", preds["y_true"], preds["probs_fused"], threshold)
    for row in fused_per_class:
        row["seed"] = seed
        row["run_dir"] = str(run_dir)

    expert_rows = []
    if preds["probs_weather"].size > 0:
        for head_name, probs in [
            ("weather_expert", preds["probs_weather"]),
            ("scene_expert", preds["probs_scene"]),
            ("time_expert", preds["probs_time"]),
        ]:
            per_class_rows = compute_per_class_object_metrics(head_name, preds["y_true"], probs, threshold)
            for row in per_class_rows:
                row["seed"] = seed
                row["run_dir"] = str(run_dir)
            expert_rows.extend(per_class_rows)

    specialist_rows = compute_specialist_metrics(preds, weather_names, scene_names, time_names, threshold)
    for row in specialist_rows:
        row["seed"] = seed
        row["run_dir"] = str(run_dir)

    condition_rows = []
    condition_confusions = {}
    if preds["weather_logits"].size > 0:
        weather_metrics, weather_cm = compute_condition_metrics(
            "weather",
            preds["weather_logits"],
            preds["weather_id"],
            preds["weather_mask"],
            weather_names,
        )
        scene_metrics, scene_cm = compute_condition_metrics(
            "scene",
            preds["scene_logits"],
            preds["scene_id"],
            preds["scene_mask"],
            scene_names,
        )
        time_metrics, time_cm = compute_condition_metrics(
            "time",
            preds["time_logits"],
            preds["time_id"],
            preds["time_mask"],
            time_names,
        )
        for row in [weather_metrics, scene_metrics, time_metrics]:
            row["seed"] = seed
            row["run_dir"] = str(run_dir)
        condition_rows.extend([weather_metrics, scene_metrics, time_metrics])
        condition_confusions = {"weather": weather_cm, "scene": scene_cm, "time": time_cm}

    
    
    router_summary_rows, router_detail_tables = compute_router_metrics(preds, weather_names, scene_names, time_names)
    router_per_class_rows = compute_router_per_class_rows(preds)
    router_condition_rows = compute_router_condition_rows(preds, weather_names, scene_names, time_names)

    for row in router_summary_rows:
        row["seed"] = seed
        row["run_dir"] = str(run_dir)

    for row in router_per_class_rows:
        row["seed"] = seed
        row["run_dir"] = str(run_dir)

    for row in router_condition_rows:
        row["seed"] = seed
        row["run_dir"] = str(run_dir)
    
    
    

    run_plot_dir = plots_dir / f"seed_{seed if seed >= 0 else run_dir.name}"
    ensure_dir(run_plot_dir)
    
    
    
    plot_multilabel_pr_curves(
        y_true=preds["y_true"],
        y_prob=preds["probs_fused"],
        class_names=TARGET_OBJECTS,
        title="Precision-Recall Curves - Fused MoE",
        path=run_plot_dir / "pr_fused_all_objects.png",
        plot_cfg=plot_cfg,
    )

    plot_multilabel_roc_curves(
        y_true=preds["y_true"],
        y_prob=preds["probs_fused"],
        class_names=TARGET_OBJECTS,
        title="ROC Curves - Fused MoE",
        path=run_plot_dir / "roc_fused_all_objects.png",
        plot_cfg=plot_cfg,
    )
    
    
    
    plot_reliability_curve(
        preds["y_true"],
        preds["probs_fused"],
        title="Reliability Diagram - Fused MoE",
        path=run_plot_dir / "reliability_fused.png",
        plot_cfg=plot_cfg,
    )
    
    
    

    for key, table in router_detail_tables.items():
        if not table:
            continue
        xlabels = [f"alpha_{name}" for name in EXPERT_NAMES]
        ylabels = [row["group"] for row in table]
        matrix = np.asarray([[row[x] for x in xlabels] for row in table], dtype=np.float64)
        
        
        plot_matrix(
            matrix=matrix,
            xlabels=EXPERT_NAMES,
            ylabels=ylabels,
            title=f"Router Weights {key.replace('_', ' ').title()}",
            path=run_plot_dir / f"router_{key}.png",
            plot_cfg=plot_cfg,
        )
        
        

    for task_name, cm in condition_confusions.items():
        class_names = weather_names if task_name == "weather" else scene_names if task_name == "scene" else time_names
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        
        
        plot_matrix(
            matrix=cm_norm,
            xlabels=class_names,
            ylabels=class_names,
            title=f"Condition Confusion Matrix - {task_name}",
            path=run_plot_dir / f"confusion_{task_name}.png",
            plot_cfg=plot_cfg,
        )
        
        

    prediction_rows = []
    if preds["alpha"].size > 0:
        top1 = np.argmax(preds["alpha"], axis=1)
    else:
        top1 = np.full((preds["y_true"].shape[0],), -1, dtype=np.int64)

    for i, image_path in enumerate(preds["image_path"]):
        row = {
            "image_path": image_path,
            "seed": seed,
            "fused_prob_car": float(preds["probs_fused"][i, 0]),
            "fused_prob_pedestrian": float(preds["probs_fused"][i, 1]),
            "fused_prob_traffic_sign": float(preds["probs_fused"][i, 2]),
            "router_alpha_weather": float(preds["alpha"][i, 0]) if preds["alpha"].size > 0 else float("nan"),
            "router_alpha_scene": float(preds["alpha"][i, 1]) if preds["alpha"].size > 0 else float("nan"),
            "router_alpha_time": float(preds["alpha"][i, 2]) if preds["alpha"].size > 0 else float("nan"),
            "router_top1_expert": EXPERT_INDEX_TO_NAME.get(int(top1[i]), "none"),
            "true_car": int(preds["y_true"][i, 0]),
            "true_pedestrian": int(preds["y_true"][i, 1]),
            "true_traffic_sign": int(preds["y_true"][i, 2]),
        }
        prediction_rows.append(row)

    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "fused_summary": fused_summary,
        "fused_per_class": fused_per_class,
        "expert_per_class": expert_rows,
        "specialist_rows": specialist_rows,
        "condition_rows": condition_rows,
        "condition_confusions": condition_confusions,
        "router_summary_rows": router_summary_rows,
        "router_detail_tables": router_detail_tables,
        "prediction_rows": prediction_rows,
        "router_per_class_rows": router_per_class_rows,
        "router_condition_rows": router_condition_rows,
            "curve_payload": {
            "seed": seed,
            "run_dir": str(run_dir),
            "y_true": preds["y_true"],
            "y_prob": preds["probs_fused"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BDD100K MoE fixed-test evaluation script")
    parser.add_argument("--train-script", type=str, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--metadata-dir", type=str, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--run-glob", type=str, default=DEFAULT_RUN_GLOB)
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--plot-title-fontsize", type=int, default=25)
    parser.add_argument("--plot-label-fontsize", type=int, default=23)
    parser.add_argument("--plot-tick-fontsize", type=int, default=22)
    parser.add_argument("--plot-legend-fontsize", type=int, default=22)
    parser.add_argument("--plot-linewidth", type=float, default=3.5)
    args = parser.parse_args()

    train_script = Path(args.train_script).resolve()
    metadata_dir = Path(args.metadata_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    csv_dir = output_dir / "csv"
    plots_dir = output_dir / "plots"
    preds_dir = output_dir / "predictions"

    ensure_dir(csv_dir)
    ensure_dir(plots_dir)
    ensure_dir(preds_dir)

    train_module = import_train_module(train_script)
    run_dirs = list_run_dirs(args.run_dirs, args.run_glob)
    device = torch.device(args.device)
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    
    
    plot_cfg = {
        "title_fontsize": args.plot_title_fontsize,
        "label_fontsize": args.plot_label_fontsize,
        "tick_fontsize": args.plot_tick_fontsize,
        "legend_fontsize": args.plot_legend_fontsize,
        "linewidth": args.plot_linewidth,
    }
    

    print(f"[{timestamp()}] Found {len(run_dirs)} run(s) for evaluation.")
    print(f"[{timestamp()}] Output directory: {output_dir}")
    
    
    
    all_fused_summary: List[Dict[str, Any]] = []
    all_fused_per_class: List[Dict[str, Any]] = []
    all_expert_per_class: List[Dict[str, Any]] = []
    all_specialist_rows: List[Dict[str, Any]] = []
    all_condition_rows: List[Dict[str, Any]] = []
    all_router_summary_rows: List[Dict[str, Any]] = []
    all_router_per_class_rows: List[Dict[str, Any]] = []
    all_router_condition_rows: List[Dict[str, Any]] = []
    all_curve_payloads: List[Dict[str, Any]] = []
    
    

    
    weather_cms: List[np.ndarray] = []
    scene_cms: List[np.ndarray] = []
    time_cms: List[np.ndarray] = []

    for run_dir in run_dirs:
        print(f"[{timestamp()}] Evaluating run: {run_dir}")
        
        
        
        result = evaluate_one_run(
            run_dir=run_dir.resolve(),
            train_module=train_module,
            metadata_dir=metadata_dir,
            checkpoint_name=args.checkpoint_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            amp_dtype=amp_dtype,
            threshold=args.threshold,
            plots_dir=plots_dir,
            plot_cfg=plot_cfg,
        )
        
        
        
        
        all_fused_summary.append(result["fused_summary"])
        all_fused_per_class.extend(result["fused_per_class"])
        all_expert_per_class.extend(result["expert_per_class"])
        all_specialist_rows.extend(result["specialist_rows"])
        all_condition_rows.extend(result["condition_rows"])
        all_router_summary_rows.extend(result["router_summary_rows"])
        all_router_per_class_rows.extend(result["router_per_class_rows"])
        all_router_condition_rows.extend(result["router_condition_rows"])
        all_curve_payloads.append(result["curve_payload"])
        
        
        

        

        write_csv(preds_dir / f"predictions_seed_{result['seed']}.csv", result["prediction_rows"])

        if "weather" in result["condition_confusions"]:
            weather_cms.append(result["condition_confusions"]["weather"])
        if "scene" in result["condition_confusions"]:
            scene_cms.append(result["condition_confusions"]["scene"])
        if "time" in result["condition_confusions"]:
            time_cms.append(result["condition_confusions"]["time"])

    
    
    
    write_csv(csv_dir / "fused_summary_per_run.csv", all_fused_summary)
    write_csv(csv_dir / "fused_per_class_per_run.csv", all_fused_per_class)
    write_csv(csv_dir / "expert_per_class_per_run.csv", all_expert_per_class)
    write_csv(csv_dir / "expert_specialist_per_run.csv", all_specialist_rows)
    write_csv(csv_dir / "condition_per_run.csv", all_condition_rows)
    write_csv(csv_dir / "router_summary_per_run.csv", all_router_summary_rows)
    write_csv(csv_dir / "router_per_class_per_run.csv", all_router_per_class_rows)
    write_csv(csv_dir / "router_condition_per_run.csv", all_router_condition_rows)
    
    
    
    
    
    fused_summary_agg = summarize_table(all_fused_summary, key_cols=["name"])
    fused_per_class_agg = summarize_table(all_fused_per_class, key_cols=["head", "object"])
    expert_per_class_agg = summarize_table(all_expert_per_class, key_cols=["head", "object"])
    expert_specialist_agg = summarize_table(all_specialist_rows, key_cols=["expert", "subset"])
    condition_agg = summarize_table(all_condition_rows, key_cols=["task"])
    router_agg = summarize_table(all_router_summary_rows, key_cols=["name"])
    router_per_class_agg = summarize_table(all_router_per_class_rows, key_cols=["object"])
    router_condition_agg = summarize_table(all_router_condition_rows, key_cols=["task", "subset"])
    
    
    
    
    write_csv(csv_dir / "fused_summary_aggregate.csv", fused_summary_agg)
    write_csv(csv_dir / "fused_per_class_aggregate.csv", fused_per_class_agg)
    write_csv(csv_dir / "expert_per_class_aggregate.csv", expert_per_class_agg)
    write_csv(csv_dir / "expert_specialist_aggregate.csv", expert_specialist_agg)
    write_csv(csv_dir / "condition_aggregate.csv", condition_agg)
    write_csv(csv_dir / "router_summary_aggregate.csv", router_agg)
    write_csv(csv_dir / "router_per_class_aggregate.csv", router_per_class_agg)
    write_csv(csv_dir / "router_condition_aggregate.csv", router_condition_agg)
    
    


    save_json(
        {
            "num_runs": len(run_dirs),
            "run_dirs": [str(p) for p in run_dirs],
            "checkpoint_name": args.checkpoint_name,
            "threshold": args.threshold,
        },
        output_dir / "evaluation_manifest.json",
    )

    if fused_summary_agg:
        
        plot_errorbar_summary(
            fused_summary_agg,
            metric_key="mAP",
            title="Fused MoE mAP Across Runs",
            path=plots_dir / "aggregate_fused_map.png",
            plot_cfg=plot_cfg,
        )
        plot_errorbar_summary(
            fused_summary_agg,
            metric_key="macro_auroc",
            title="Fused MoE Macro AUROC Across Runs",
            path=plots_dir / "aggregate_fused_macro_auroc.png",
            plot_cfg=plot_cfg,
        )
        plot_errorbar_summary(
            fused_summary_agg,
            metric_key="inference_ms",
            title="Fused MoE Inference Time Across Runs",
            path=plots_dir / "aggregate_fused_inference_ms.png",
            plot_cfg=plot_cfg,
        )
        
        
        plot_mean_multirun_pr_curves(
            curve_payloads=all_curve_payloads,
            class_names=TARGET_OBJECTS,
            title="Mean Precision-Recall Curves Across Runs - Fused MoE",
            path=plots_dir / "aggregate_pr_fused_all_objects.png",
            plot_cfg=plot_cfg,
        )

        plot_mean_multirun_roc_curves(
            curve_payloads=all_curve_payloads,
            class_names=TARGET_OBJECTS,
            title="Mean ROC Curves Across Runs - Fused MoE",
            path=plots_dir / "aggregate_roc_fused_all_objects.png",
            plot_cfg=plot_cfg,
        )
        
     
        

    train_bundle = load_json(metadata_dir / "metadata_bundle.json")
    weather_names = [k for k, _ in sorted(train_bundle["weather_to_id"].items(), key=lambda kv: kv[1])]
    scene_names = [k for k, _ in sorted(train_bundle["scene_to_id"].items(), key=lambda kv: kv[1])]
    time_names = [k for k, _ in sorted(train_bundle["time_to_id"].items(), key=lambda kv: kv[1])]

    if weather_cms:
        cm = aggregate_confusion_matrices(weather_cms)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        plot_matrix(cm, weather_names, weather_names, "Mean Weather Confusion Matrix", plots_dir / "mean_confusion_weather.png", plot_cfg=plot_cfg,)
        
        
        
        
    if scene_cms:
        cm = aggregate_confusion_matrices(scene_cms)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        
        
        
        plot_matrix(
            cm,
            scene_names,
            scene_names,
            "Mean Scene Confusion Matrix",
            plots_dir / "mean_confusion_scene.png",
            plot_cfg=plot_cfg,
        )
        
        
        
        
    if time_cms:
        cm = aggregate_confusion_matrices(time_cms)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
       
        
        plot_matrix(
            cm,
            time_names,
            time_names,
            "Mean Time Confusion Matrix",
            plots_dir / "mean_confusion_time.png",
            plot_cfg=plot_cfg,
        )
                
        

    print(f"[{timestamp()}] Evaluation complete.")
    print(f"[{timestamp()}] CSV files saved to: {csv_dir}")
    print(f"[{timestamp()}] PNG plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
