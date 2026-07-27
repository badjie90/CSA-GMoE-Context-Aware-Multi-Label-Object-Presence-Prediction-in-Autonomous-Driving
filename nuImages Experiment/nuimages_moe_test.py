#!/usr/bin/env python3
# path: nuimages_moe_test.py
"""
nuImages Condition-Specialized MoE Test/Evaluation Script

This script evaluates the fixed held-out test split created by the train script.
It keeps the same general evaluation settings as the BDD100K MoE evaluator while
adapting labels and semantics to nuImages.

Important adaptation
--------------------
- Internal "weather / scene / time" branches correspond to:
  - weather -> location
  - scene   -> illumination
  - time    -> motion
- The third object-presence slot corresponds to traffic_cone.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest import result

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

TARGET_OBJECTS = ["car", "pedestrian", "traffic_cone"]
EXPERT_NAMES = ["location", "illumination", "motion"]



DEFAULT_TRAIN_SCRIPT = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Nuimages/Scripts/nuimages_moe_train.py"
DEFAULT_METADATA_DIR = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Nuimages/metaoutputs/nuimages_metadata"
DEFAULT_OUTPUT_DIR = "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Nuimages/nuimages_Test_Outputs-New"


DEFAULT_RUN_DIRS = [
    "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Nuimages/experiments/nuimages_moe_stage3",
]



# DEFAULT_RUN_DIRS = [
#     "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Nuimages/experiments/nuimages_moe_stage3",
# ]



# DEFAULT_RUN_DIRS = [
#     "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Nuimages/experiments/nuimages_moe_stage3_seed1",
#     "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Nuimages/experiments/nuimages_moe_stage3_seed2",
#     "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Nuimages/experiments/nuimages_moe_stage3_seed3",
#     "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Nuimages/experiments/nuimages_moe_stage3_seed4",
#     "/mnt/nvme1n1/bbadjie/SEAS/AAutonomous/Nuimages/experiments/nuimages_moe_stage3_seed5",
# ]




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
    module_name = "nuimages_moe_train_module"
    spec = importlib.util.spec_from_file_location(module_name, train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import train module from: {train_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# def list_run_dirs(run_dirs: Sequence[str]) -> List[Path]:
#     resolved = [Path(p).resolve() for p in run_dirs if Path(p).exists()]
#     if not resolved:
#         raise FileNotFoundError("No run directories found. Pass them explicitly with --run-dirs.")
#     return resolved

def list_run_dirs(run_dirs: Sequence[str]) -> List[Path]:
    resolved = [Path(p).resolve() for p in run_dirs if Path(p).exists()]
    if not resolved:
        raise FileNotFoundError(f"No run directories found. Got: {run_dirs}")
    return resolved


def get_t_critical_95(n: int) -> float:
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    return 0.0 if n <= 1 else table.get(n, 1.96)


def mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    half = get_t_critical_95(arr.size) * (std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, std, mean - half, mean + half


def safe_metric(fn, *args, **kwargs) -> float:
    try:
        value = fn(*args, **kwargs)
        return float(np.mean(value)) if isinstance(value, np.ndarray) else float(value)
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
        if np.sum(y_true[:, c]) > 0:
            scores.append(average_precision_score(y_true[:, c], y_prob[:, c]))
    return float(np.mean(scores)) if scores else float("nan")


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    scores = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:
            scores.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
    return float(np.mean(scores)) if scores else float("nan")


def micro_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return safe_metric(average_precision_score, y_true.ravel(), y_prob.ravel())


def micro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true.ravel())) < 2:
        return float("nan")
    return safe_metric(roc_auc_score, y_true.ravel(), y_prob.ravel())


def expected_calibration_error_binary(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= start) & (y_prob < end if end < 1.0 else y_prob <= end)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += abs(acc - conf) * (mask.sum() / max(n, 1))
    return float(ece)


def multilabel_ece(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    return float(np.mean([expected_calibration_error_binary(y_true[:, c], y_prob[:, c], bins=bins) for c in range(y_true.shape[1])]))


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
        return float(starter.elapsed_time(ender))
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(images)
    return float((time.perf_counter() - start) * 1000.0)


def make_test_loader(train_module, metadata_dir: Path, image_size: int, batch_size: int, num_workers: int):
    bundle = load_json(metadata_dir / "metadata_bundle.json")
    test_rows = load_json(metadata_dir / "test_fixed.json")
    _, eval_transform = train_module.build_transforms(image_size=image_size)
    dataset = train_module.NuImagesPresenceDataset(test_rows, eval_transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=train_module.seed_worker,
    )
    return loader, bundle


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


def _style_axes(ax, title: str, xlabel: str, ylabel: str, plot_cfg: Dict[str, Any], use_legend: bool = False) -> None:
    ax.set_title(title, fontsize=plot_cfg["title_fontsize"], fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=plot_cfg["label_fontsize"], fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=plot_cfg["label_fontsize"], fontweight="bold")
    _style_ticks(ax, plot_cfg["tick_fontsize"])
    if use_legend:
        _style_legend(ax, plot_cfg["legend_fontsize"])


def plot_matrix(matrix: np.ndarray, xlabels: Sequence[str], ylabels: Sequence[str], title: str, path: Path, plot_cfg: Dict[str, Any]) -> None:
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


def plot_errorbar_summary(rows: Sequence[Dict[str, Any]], metric_key: str, title: str, path: Path, plot_cfg: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    labels = [row["name"] for row in rows]
    means = [row[f"{metric_key}_mean"] for row in rows]
    stds = [row[f"{metric_key}_std"] for row in rows]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.errorbar(range(len(labels)), means, yerr=stds, fmt="o", linewidth=plot_cfg["linewidth"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    _style_axes(ax, title=title, xlabel="", ylabel=metric_key, plot_cfg=plot_cfg, use_legend=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, title: str, path: Path, plot_cfg: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(10, 10))
    for c, label in enumerate(TARGET_OBJECTS):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        frac_pos, mean_pred = calibration_curve(y_true[:, c], y_prob[:, c], n_bins=15, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=plot_cfg["linewidth"], label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="ideal")
    _style_axes(ax, title=title, xlabel="Predicted probability", ylabel="Observed frequency", plot_cfg=plot_cfg, use_legend=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multilabel_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, class_names: Sequence[str], title: str, path: Path, plot_cfg: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(12, 10))
    for c, class_name in enumerate(class_names):
        if np.sum(y_true[:, c]) == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_true[:, c], y_prob[:, c])
        ap = average_precision_score(y_true[:, c], y_prob[:, c])
        ax.plot(recall, precision, linewidth=plot_cfg["linewidth"], label=f"{class_name} (AP={ap:.3f})")
    _style_axes(ax, title=title, xlabel="Recall", ylabel="Precision", plot_cfg=plot_cfg, use_legend=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multilabel_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, class_names: Sequence[str], title: str, path: Path, plot_cfg: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(12, 10))
    for c, class_name in enumerate(class_names):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, c], y_prob[:, c])
        auc = roc_auc_score(y_true[:, c], y_prob[:, c])
        ax.plot(fpr, tpr, linewidth=plot_cfg["linewidth"], label=f"{class_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="chance")
    _style_axes(ax, title=title, xlabel="False positive rate", ylabel="True positive rate", plot_cfg=plot_cfg, use_legend=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mean_multirun_pr_curves(curve_payloads: Sequence[Dict[str, Any]], class_names: Sequence[str], title: str, path: Path, plot_cfg: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    recall_grid = np.linspace(0.0, 1.0, 201)
    fig, ax = plt.subplots(figsize=(12, 10))

    for c, class_name in enumerate(class_names):
        precision_runs = []
        ap_runs = []
        for payload in curve_payloads:
            y_true = payload["y_true"][:, c]
            y_prob = payload["y_prob"][:, c]
            if np.sum(y_true) == 0:
                continue
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            order = np.argsort(recall)
            interp_precision = np.interp(recall_grid, recall[order], precision[order])
            precision_runs.append(interp_precision)
            ap_runs.append(average_precision_score(y_true, y_prob))
        if not precision_runs:
            continue
        precision_runs_np = np.vstack(precision_runs)
        mean_precision = precision_runs_np.mean(axis=0)
        ci_precision = 1.96 * precision_runs_np.std(axis=0, ddof=1) / np.sqrt(precision_runs_np.shape[0]) if precision_runs_np.shape[0] > 1 else np.zeros_like(mean_precision)
        line = ax.plot(recall_grid, mean_precision, linewidth=plot_cfg["linewidth"], label=f"{class_name} (mean AP={np.mean(ap_runs):.3f})")[0]
        ax.fill_between(recall_grid, np.clip(mean_precision - ci_precision, 0.0, 1.0), np.clip(mean_precision + ci_precision, 0.0, 1.0), alpha=0.20, color=line.get_color())

    _style_axes(ax, title=title, xlabel="Recall", ylabel="Precision", plot_cfg=plot_cfg, use_legend=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_mean_multirun_roc_curves(curve_payloads: Sequence[Dict[str, Any]], class_names: Sequence[str], title: str, path: Path, plot_cfg: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    fpr_grid = np.linspace(0.0, 1.0, 201)
    fig, ax = plt.subplots(figsize=(12, 10))

    for c, class_name in enumerate(class_names):
        tpr_runs = []
        auc_runs = []
        for payload in curve_payloads:
            y_true = payload["y_true"][:, c]
            y_prob = payload["y_prob"][:, c]
            if len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            interp_tpr = np.interp(fpr_grid, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tpr[-1] = 1.0
            tpr_runs.append(interp_tpr)
            auc_runs.append(roc_auc_score(y_true, y_prob))
        if not tpr_runs:
            continue
        tpr_runs_np = np.vstack(tpr_runs)
        mean_tpr = tpr_runs_np.mean(axis=0)
        ci_tpr = 1.96 * tpr_runs_np.std(axis=0, ddof=1) / np.sqrt(tpr_runs_np.shape[0]) if tpr_runs_np.shape[0] > 1 else np.zeros_like(mean_tpr)
        line = ax.plot(fpr_grid, mean_tpr, linewidth=plot_cfg["linewidth"], label=f"{class_name} (mean AUC={np.mean(auc_runs):.3f})")[0]
        ax.fill_between(fpr_grid, np.clip(mean_tpr - ci_tpr, 0.0, 1.0), np.clip(mean_tpr + ci_tpr, 0.0, 1.0), alpha=0.20, color=line.get_color())

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="chance")
    _style_axes(ax, title=title, xlabel="False positive rate", ylabel="True positive rate", plot_cfg=plot_cfg, use_legend=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def compute_multilabel_metrics(name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
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


def compute_per_class_object_metrics(head_name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> List[Dict[str, Any]]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    rows = []
    for c, obj_name in enumerate(TARGET_OBJECTS):
        rows.append(
            {
                "head": head_name,
                "object": obj_name,
                "precision": safe_metric(precision_score, y_true[:, c], y_pred[:, c], zero_division=0),
                "recall": safe_metric(recall_score, y_true[:, c], y_pred[:, c], zero_division=0),
                "f1": safe_metric(f1_score, y_true[:, c], y_pred[:, c], zero_division=0),
                "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true[:, c], y_pred[:, c]) if len(np.unique(y_true[:, c])) > 1 else float("nan"),
                "ap": safe_metric(average_precision_score, y_true[:, c], y_prob[:, c]) if np.sum(y_true[:, c]) > 0 else float("nan"),
                "auroc": safe_metric(roc_auc_score, y_true[:, c], y_prob[:, c]) if len(np.unique(y_true[:, c])) > 1 else float("nan"),
            }
        )
    return rows




def compute_condition_metrics(task_name: str, logits: np.ndarray, y_true: np.ndarray, mask: np.ndarray, class_names: Sequence[str]) -> Tuple[Dict[str, float], np.ndarray]:
    valid = mask.astype(bool)
    if not np.any(valid):
        empty = np.zeros((len(class_names), len(class_names)), dtype=np.float64)
        return {"task": task_name, "accuracy": float("nan"), "macro_f1": float("nan"), "balanced_accuracy": float("nan")}, empty

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


def compute_router_metrics(preds: Dict[str, Any], cond1_names: Sequence[str], cond2_names: Sequence[str], cond3_names: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    alpha = preds["alpha"]
    if alpha.size == 0:
        return [], {}

    top1 = np.argmax(alpha, axis=1)
    mean_alpha = alpha.mean(axis=0)
    entropy = -(alpha * np.log(alpha + 1e-12)).sum(axis=1)
    normalized_entropy = entropy / np.log(alpha.shape[1])

    summary = [
        {"name": "router_mean_alpha_location", "value": float(mean_alpha[0])},
        {"name": "router_mean_alpha_illumination", "value": float(mean_alpha[1])},
        {"name": "router_mean_alpha_motion", "value": float(mean_alpha[2])},
        {"name": "router_entropy", "value": float(entropy.mean())},
        {"name": "router_normalized_entropy", "value": float(normalized_entropy.mean())},
        {"name": "router_collapse_score", "value": float(mean_alpha.max())},
        {"name": "router_top1_location_fraction", "value": float(np.mean(top1 == 0))},
        {"name": "router_top1_illumination_fraction", "value": float(np.mean(top1 == 1))},
        {"name": "router_top1_motion_fraction", "value": float(np.mean(top1 == 2))},
    ]

    detailed = {"by_location": [], "by_illumination": [], "by_motion": [], "by_object_presence": []}
    for label_id, label_name in enumerate(cond1_names):
        mask = (preds["weather_mask"] > 0) & (preds["weather_id"] == label_id)
        if np.any(mask):
            detailed["by_location"].append(
                {
                    "group": label_name,
                    "alpha_location": float(alpha[mask].mean(axis=0)[0]),
                    "alpha_illumination": float(alpha[mask].mean(axis=0)[1]),
                    "alpha_motion": float(alpha[mask].mean(axis=0)[2]),
                }
            )
    for label_id, label_name in enumerate(cond2_names):
        mask = (preds["scene_mask"] > 0) & (preds["scene_id"] == label_id)
        if np.any(mask):
            detailed["by_illumination"].append(
                {
                    "group": label_name,
                    "alpha_location": float(alpha[mask].mean(axis=0)[0]),
                    "alpha_illumination": float(alpha[mask].mean(axis=0)[1]),
                    "alpha_motion": float(alpha[mask].mean(axis=0)[2]),
                }
            )
    for label_id, label_name in enumerate(cond3_names):
        mask = (preds["time_mask"] > 0) & (preds["time_id"] == label_id)
        if np.any(mask):
            detailed["by_motion"].append(
                {
                    "group": label_name,
                    "alpha_location": float(alpha[mask].mean(axis=0)[0]),
                    "alpha_illumination": float(alpha[mask].mean(axis=0)[1]),
                    "alpha_motion": float(alpha[mask].mean(axis=0)[2]),
                }
            )
    for obj_id, obj_name in enumerate(TARGET_OBJECTS):
        for state in [0, 1]:
            mask = preds["y_true"][:, obj_id] == state
            if np.any(mask):
                detailed["by_object_presence"].append(
                    {
                        "group": f"{obj_name}_{state}",
                        "alpha_location": float(alpha[mask].mean(axis=0)[0]),
                        "alpha_illumination": float(alpha[mask].mean(axis=0)[1]),
                        "alpha_motion": float(alpha[mask].mean(axis=0)[2]),
                    }
                )
    return summary, detailed




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
                "alpha_location": float(mean_alpha[0]),
                "alpha_illumination": float(mean_alpha[1]),
                "alpha_motion": float(mean_alpha[2]),
                "top1_location_fraction": float(np.mean(top1[mask] == 0)),
                "top1_illumination_fraction": float(np.mean(top1[mask] == 1)),
                "top1_motion_fraction": float(np.mean(top1[mask] == 2)),
                "entropy": float(entropy.mean()),
                "normalized_entropy": float(normalized_entropy.mean()),
                "collapse_score": float(mean_alpha.max()),
            }
        )

    return rows




def compute_router_condition_rows(
    preds: Dict[str, Any],
    location_names: Sequence[str],
    illumination_names: Sequence[str],
    motion_names: Sequence[str],
) -> List[Dict[str, Any]]:
    alpha = preds["alpha"]
    if alpha.size == 0:
        return []

    top1 = np.argmax(alpha, axis=1)
    rows: List[Dict[str, Any]] = []

    condition_specs = [
        ("location", preds["weather_id"], preds["weather_mask"], location_names),
        ("illumination", preds["scene_id"], preds["scene_mask"], illumination_names),
        ("motion", preds["time_id"], preds["time_mask"], motion_names),
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
                    "alpha_location": float(mean_alpha[0]),
                    "alpha_illumination": float(mean_alpha[1]),
                    "alpha_motion": float(mean_alpha[2]),
                    "top1_location_fraction": float(np.mean(top1[mask] == 0)),
                    "top1_illumination_fraction": float(np.mean(top1[mask] == 1)),
                    "top1_motion_fraction": float(np.mean(top1[mask] == 2)),
                    "entropy": float(entropy.mean()),
                    "normalized_entropy": float(normalized_entropy.mean()),
                    "collapse_score": float(mean_alpha.max()),
                }
            )

    return rows



def compute_expert_condition_rows(
    preds: Dict[str, Any],
    location_names: Sequence[str],
    illumination_names: Sequence[str],
    motion_names: Sequence[str],
    threshold: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    expert_specs = [
        ("location_expert", preds["probs_weather"], preds["weather_id"], preds["weather_mask"], location_names),
        ("illumination_expert", preds["probs_scene"], preds["scene_id"], preds["scene_mask"], illumination_names),
        ("motion_expert", preds["probs_time"], preds["time_id"], preds["time_mask"], motion_names),
    ]

    for expert_name, probs, labels, masks, class_names in expert_specs:
        if probs.size == 0:
            continue

        for class_id, class_name in enumerate(class_names):
            mask = (masks > 0) & (labels == class_id)
            if not np.any(mask):
                continue

            metrics = compute_multilabel_metrics(
                name=f"{expert_name}_{class_name}",
                y_true=preds["y_true"][mask],
                y_prob=probs[mask],
                threshold=threshold,
            )

            rows.append(
                {
                    "expert": expert_name,
                    "subset": class_name,
                    **{k: v for k, v in metrics.items() if k != "name"},
                }
            )

    return rows



def summarize_table(rows: Sequence[Dict[str, Any]], key_cols: Sequence[str]) -> List[Dict[str, Any]]:
    rows = list(rows)
    if not rows:
        return []
    metric_cols = [col for col in rows[0].keys() if col not in key_cols and col != "seed" and isinstance(rows[0][col], (int, float, np.floating))]
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[col] for col in key_cols)
        grouped[key].append(row)

    summary_rows = []
    for key, group_rows in grouped.items():
        summary = {col: value for col, value in zip(key_cols, key)}
        for metric in metric_cols:
            values = [float(r[metric]) for r in group_rows if not (isinstance(r[metric], float) and math.isnan(r[metric]))]
            mean, std, ci_low, ci_high = mean_std_ci(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_ci_low"] = ci_low
            summary[f"{metric}_ci_high"] = ci_high
        summary_rows.append(summary)
    return summary_rows




def collect_predictions(model: torch.nn.Module, loader, train_module, device: torch.device, amp_dtype: torch.dtype) -> Dict[str, Any]:
    model.eval()
    chunks: Dict[str, List[np.ndarray]] = defaultdict(list)
    image_paths: List[str] = []
    batch_ms: List[float] = []
    num_images = 0

    with torch.no_grad():
        for batch in loader:
            image_paths.extend(batch["image_path"])
            batch = train_module.move_batch_to_device(batch, device)
            images = batch["image"]
            batch_ms.append(infer_batch_time_ms(model, images, device, amp_dtype))

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                outputs = model(images)

            chunks["y_true"].append(batch["objects"].detach().cpu().numpy())
            chunks["weather_id"].append(batch["weather_id"].detach().cpu().numpy())
            chunks["scene_id"].append(batch["scene_id"].detach().cpu().numpy())
            chunks["time_id"].append(batch["time_id"].detach().cpu().numpy())
            chunks["weather_mask"].append(batch["weather_mask"].detach().cpu().numpy())
            chunks["scene_mask"].append(batch["scene_mask"].detach().cpu().numpy())
            chunks["time_mask"].append(batch["time_mask"].detach().cpu().numpy())
            chunks["probs_fused"].append(torch.sigmoid(outputs["obj_fused_logits"]).detach().cpu().numpy())

            if "obj_w_logits" in outputs:
                chunks["probs_weather"].append(torch.sigmoid(outputs["obj_w_logits"]).detach().cpu().numpy())
                chunks["probs_scene"].append(torch.sigmoid(outputs["obj_s_logits"]).detach().cpu().numpy())
                chunks["probs_time"].append(torch.sigmoid(outputs["obj_t_logits"]).detach().cpu().numpy())
                chunks["weather_logits"].append(outputs["weather_logits"].detach().cpu().numpy())
                chunks["scene_logits"].append(outputs["scene_logits"].detach().cpu().numpy())
                chunks["time_logits"].append(outputs["time_logits"].detach().cpu().numpy())
                chunks["alpha"].append(outputs["alpha"].detach().cpu().numpy())

            num_images += images.shape[0]

    def pack(name: str, ndim: int = 2) -> np.ndarray:
        if name not in chunks or not chunks[name]:
            return np.empty((0,) if ndim == 1 else (0, 0), dtype=np.float32)
        return np.concatenate(chunks[name], axis=0)

    return {
        "image_path": image_paths,
        "y_true": pack("y_true"),
        "weather_id": pack("weather_id", ndim=1),
        "scene_id": pack("scene_id", ndim=1),
        "time_id": pack("time_id", ndim=1),
        "weather_mask": pack("weather_mask", ndim=1),
        "scene_mask": pack("scene_mask", ndim=1),
        "time_mask": pack("time_mask", ndim=1),
        "probs_fused": pack("probs_fused"),
        "probs_weather": pack("probs_weather"),
        "probs_scene": pack("probs_scene"),
        "probs_time": pack("probs_time"),
        "weather_logits": pack("weather_logits"),
        "scene_logits": pack("scene_logits"),
        "time_logits": pack("time_logits"),
        "alpha": pack("alpha"),
        "inference_ms_per_image": float(np.sum(batch_ms) / max(num_images, 1)),
    }





def evaluate_one_run(run_dir: Path, train_module, metadata_dir: Path, checkpoint_name: str, batch_size: int, num_workers: int, device: torch.device, amp_dtype: torch.dtype, threshold: float, plots_dir: Path, plot_cfg: Dict[str, Any]) -> Dict[str, Any]:
    config = load_json(run_dir / "config.json")
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    payload = torch.load(checkpoint_path, map_location="cpu")

    loader, bundle = make_test_loader(train_module, metadata_dir, int(config.get("image_size", 224)), batch_size, num_workers)
    location_names = [k for k, _ in sorted(bundle["weather_to_id"].items(), key=lambda kv: kv[1])]
    illumination_names = [k for k, _ in sorted(bundle["scene_to_id"].items(), key=lambda kv: kv[1])]
    motion_names = [k for k, _ in sorted(bundle["time_to_id"].items(), key=lambda kv: kv[1])]

    model = train_module.build_model(
        stage=str(config.get("stage", "stage3_moe")),
        backbone_name=str(config.get("backbone_name", "convnextv2_tiny.fcmae_ft_in1k")),
        num_weather=len(location_names),
        num_scene=len(illumination_names),
        num_time=len(motion_names),
        pretrained=False,
    )
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device)
    model.eval()

    preds = collect_predictions(model=model, loader=loader, train_module=train_module, device=device, amp_dtype=amp_dtype)
    seed = int(config.get("seed", -1))

    fused_summary = compute_multilabel_metrics("fused_moe", preds["y_true"], preds["probs_fused"], threshold)
    fused_summary["inference_ms"] = preds["inference_ms_per_image"]
    fused_summary["seed"] = seed
    fused_summary["run_dir"] = str(run_dir)

    fused_per_class = compute_per_class_object_metrics("fused_moe", preds["y_true"], preds["probs_fused"], threshold)
    for row in fused_per_class:
        row["seed"] = seed
        row["run_dir"] = str(run_dir)

    expert_rows = []
    for head_name, probs in [("location_expert", preds["probs_weather"]), ("illumination_expert", preds["probs_scene"]), ("motion_expert", preds["probs_time"])]:
        if probs.size == 0:
            continue
        rows = compute_per_class_object_metrics(head_name, preds["y_true"], probs, threshold)
        for row in rows:
            row["seed"] = seed
            row["run_dir"] = str(run_dir)
        expert_rows.extend(rows)

    condition_rows = []
    condition_confusions = {}
    if preds["weather_logits"].size > 0:
        location_metrics, location_cm = compute_condition_metrics("location", preds["weather_logits"], preds["weather_id"], preds["weather_mask"], location_names)
        illumination_metrics, illumination_cm = compute_condition_metrics("illumination", preds["scene_logits"], preds["scene_id"], preds["scene_mask"], illumination_names)
        motion_metrics, motion_cm = compute_condition_metrics("motion", preds["time_logits"], preds["time_id"], preds["time_mask"], motion_names)
        for row in [location_metrics, illumination_metrics, motion_metrics]:
            row["seed"] = seed
            row["run_dir"] = str(run_dir)
        condition_rows.extend([location_metrics, illumination_metrics, motion_metrics])
        condition_confusions = {"location": location_cm, "illumination": illumination_cm, "motion": motion_cm}
        
        

    router_summary_rows, router_detail_tables = compute_router_metrics(preds, location_names, illumination_names, motion_names)
    router_per_class_rows = compute_router_per_class_rows(preds)
    router_condition_rows = compute_router_condition_rows(preds, location_names, illumination_names, motion_names)
    expert_condition_rows = compute_expert_condition_rows(preds, location_names, illumination_names, motion_names, threshold)
    
    # router_summary_rows, router_detail_tables = compute_router_metrics(preds, location_names, illumination_names, motion_names)
    # router_per_class_rows = compute_router_per_class_rows(preds)
    # router_condition_rows = compute_router_condition_rows(preds, location_names, illumination_names, motion_names)

    for row in router_summary_rows:
        row["seed"] = seed
        row["run_dir"] = str(run_dir)

    for row in router_per_class_rows:
        row["seed"] = seed
        row["run_dir"] = str(run_dir)

    for row in router_condition_rows:
        row["seed"] = seed
        row["run_dir"] = str(run_dir)
        
    for row in expert_condition_rows:
        row["seed"] = seed
        row["run_dir"] = str(run_dir)
        
    
    # router_summary_rows, router_detail_tables = compute_router_metrics(preds, location_names, illumination_names, motion_names)
    # for row in router_summary_rows:
    #     row["seed"] = seed
    #     row["run_dir"] = str(run_dir)

    run_plot_dir = plots_dir / f"seed_{seed if seed >= 0 else run_dir.name}"
    ensure_dir(run_plot_dir)

    plot_multilabel_pr_curves(preds["y_true"], preds["probs_fused"], TARGET_OBJECTS, "Precision-Recall Curves - Fused MoE", run_plot_dir / "pr_fused_all_objects.png", plot_cfg)
    plot_multilabel_roc_curves(preds["y_true"], preds["probs_fused"], TARGET_OBJECTS, "ROC Curves - Fused MoE", run_plot_dir / "roc_fused_all_objects.png", plot_cfg)
    plot_reliability_curve(preds["y_true"], preds["probs_fused"], "Reliability Diagram - Fused MoE", run_plot_dir / "reliability_fused.png", plot_cfg)

    for key, table in router_detail_tables.items():
        if not table:
            continue
        xlabels = ["alpha_location", "alpha_illumination", "alpha_motion"]
        ylabels = [row["group"] for row in table]
        matrix = np.asarray([[row[x] for x in xlabels] for row in table], dtype=np.float64)
        plot_matrix(matrix=matrix, xlabels=["location", "illumination", "motion"], ylabels=ylabels, title=f"Router Weights {key.replace('_', ' ').title()}", path=run_plot_dir / f"router_{key}.png", plot_cfg=plot_cfg)

    for task_name, cm in condition_confusions.items():
        class_names = location_names if task_name == "location" else illumination_names if task_name == "illumination" else motion_names
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        plot_matrix(matrix=cm_norm, xlabels=class_names, ylabels=class_names, title=f"Condition Confusion Matrix - {task_name.title()}", path=run_plot_dir / f"confusion_{task_name}.png", plot_cfg=plot_cfg)

    prediction_rows = []
    top1 = np.argmax(preds["alpha"], axis=1) if preds["alpha"].size > 0 else np.full((preds["y_true"].shape[0],), -1, dtype=np.int64)
    top1_map = {0: "location", 1: "illumination", 2: "motion"}
    for i, image_path in enumerate(preds["image_path"]):
        prediction_rows.append(
            {
                "image_path": image_path,
                "seed": seed,
                "fused_prob_car": float(preds["probs_fused"][i, 0]),
                "fused_prob_pedestrian": float(preds["probs_fused"][i, 1]),
                "fused_prob_traffic_cone": float(preds["probs_fused"][i, 2]),
                "router_alpha_location": float(preds["alpha"][i, 0]) if preds["alpha"].size > 0 else float("nan"),
                "router_alpha_illumination": float(preds["alpha"][i, 1]) if preds["alpha"].size > 0 else float("nan"),
                "router_alpha_motion": float(preds["alpha"][i, 2]) if preds["alpha"].size > 0 else float("nan"),
                "router_top1_expert": top1_map.get(int(top1[i]), "none"),
                "true_car": int(preds["y_true"][i, 0]),
                "true_pedestrian": int(preds["y_true"][i, 1]),
                "true_traffic_cone": int(preds["y_true"][i, 2]),
            }
        )

    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "fused_summary": fused_summary,
        "fused_per_class": fused_per_class,
        "expert_per_class": expert_rows,
        "condition_rows": condition_rows,
        "condition_confusions": condition_confusions,
        "router_summary_rows": router_summary_rows,
        "router_condition_rows": router_condition_rows,
        "router_per_class_rows": router_per_class_rows,
        "expert_condition_rows": expert_condition_rows,
        "router_detail_tables": router_detail_tables,
        "prediction_rows": prediction_rows,
        "curve_payload": {
            "seed": seed,
            "run_dir": str(run_dir),
            "y_true": preds["y_true"],
            "y_prob": preds["probs_fused"],
        },
    }


def aggregate_confusion_matrices(mats: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(mats, axis=0), axis=0) if mats else np.empty((0, 0), dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="nuImages MoE fixed-test evaluation script")
    parser.add_argument("--train-script", type=str, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--metadata-dir", type=str, default=DEFAULT_METADATA_DIR)
    #parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--plot-title-fontsize", type=int, default=25)
    parser.add_argument("--plot-label-fontsize", type=int, default=22)
    parser.add_argument("--plot-tick-fontsize", type=int, default=22)
    parser.add_argument("--plot-legend-fontsize", type=int, default=22)
    parser.add_argument("--plot-linewidth", type=float, default=2.5)
    #parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--run-dirs", nargs="+", default=DEFAULT_RUN_DIRS)
    args = parser.parse_args()

    plot_cfg = {
        "title_fontsize": args.plot_title_fontsize,
        "label_fontsize": args.plot_label_fontsize,
        "tick_fontsize": args.plot_tick_fontsize,
        "legend_fontsize": args.plot_legend_fontsize,
        "linewidth": args.plot_linewidth,
    }

    train_module = import_train_module(Path(args.train_script).resolve())
    run_dirs = list_run_dirs(args.run_dirs)
    metadata_dir = Path(args.metadata_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    csv_dir = output_dir / "csv"
    plots_dir = output_dir / "plots"
    preds_dir = output_dir / "predictions"
    ensure_dir(csv_dir)
    ensure_dir(plots_dir)
    ensure_dir(preds_dir)

    device = torch.device(args.device)
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16

    
    all_fused_summary = []
    all_fused_per_class = []
    all_expert_per_class = []
    all_expert_condition_rows = []
    all_condition_rows = []
    all_router_summary_rows = []
    all_router_per_class_rows = []
    all_router_condition_rows = []
    all_curve_payloads = []
    
    
    # all_fused_summary = []
    # all_fused_per_class = []
    # all_expert_per_class = []
    # all_condition_rows = []
    # all_router_summary_rows = []
    # all_curve_payloads = []

    location_cms = []
    illumination_cms = []
    motion_cms = []

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
        all_expert_condition_rows.extend(result["expert_condition_rows"])
        all_condition_rows.extend(result["condition_rows"])
        all_router_summary_rows.extend(result["router_summary_rows"])
        all_router_per_class_rows.extend(result["router_per_class_rows"])
        all_router_condition_rows.extend(result["router_condition_rows"])
        all_curve_payloads.append(result["curve_payload"])
        
        
        
        # all_fused_summary.append(result["fused_summary"])
        # all_fused_per_class.extend(result["fused_per_class"])
        # all_expert_per_class.extend(result["expert_per_class"])
        # all_condition_rows.extend(result["condition_rows"])
        # all_router_summary_rows.extend(result["router_summary_rows"])
        # all_curve_payloads.append(result["curve_payload"])
        write_csv(preds_dir / f"predictions_seed_{result['seed']}.csv", result["prediction_rows"])

        if "location" in result["condition_confusions"]:
            location_cms.append(result["condition_confusions"]["location"])
        if "illumination" in result["condition_confusions"]:
            illumination_cms.append(result["condition_confusions"]["illumination"])
        if "motion" in result["condition_confusions"]:
            motion_cms.append(result["condition_confusions"]["motion"])

    # write_csv(csv_dir / "fused_summary_per_run.csv", all_fused_summary)
    # write_csv(csv_dir / "fused_per_class_per_run.csv", all_fused_per_class)
    # write_csv(csv_dir / "expert_per_class_per_run.csv", all_expert_per_class)
    # write_csv(csv_dir / "condition_per_run.csv", all_condition_rows)
    # write_csv(csv_dir / "router_summary_per_run.csv", all_router_summary_rows)
    
    
    
    write_csv(csv_dir / "fused_summary_per_run.csv", all_fused_summary)
    write_csv(csv_dir / "fused_per_class_per_run.csv", all_fused_per_class)
    write_csv(csv_dir / "expert_per_class_per_run.csv", all_expert_per_class)
    write_csv(csv_dir / "expert_condition_per_run.csv", all_expert_condition_rows)
    write_csv(csv_dir / "condition_per_run.csv", all_condition_rows)
    write_csv(csv_dir / "router_summary_per_run.csv", all_router_summary_rows)
    write_csv(csv_dir / "router_per_class_per_run.csv", all_router_per_class_rows)
    write_csv(csv_dir / "router_condition_per_run.csv", all_router_condition_rows)
        
    
    fused_summary_agg = summarize_table(all_fused_summary, key_cols=["name"])
    fused_per_class_agg = summarize_table(all_fused_per_class, key_cols=["head", "object"])
    expert_per_class_agg = summarize_table(all_expert_per_class, key_cols=["head", "object"])
    expert_condition_agg = summarize_table(all_expert_condition_rows, key_cols=["expert", "subset"])
    condition_agg = summarize_table(all_condition_rows, key_cols=["task"])
    router_agg = summarize_table(all_router_summary_rows, key_cols=["name"])
    router_per_class_agg = summarize_table(all_router_per_class_rows, key_cols=["object"])
    router_condition_agg = summarize_table(all_router_condition_rows, key_cols=["task", "subset"])
    
    

    # fused_summary_agg = summarize_table(all_fused_summary, key_cols=["name"])
    # fused_per_class_agg = summarize_table(all_fused_per_class, key_cols=["head", "object"])
    # expert_per_class_agg = summarize_table(all_expert_per_class, key_cols=["head", "object"])
    # condition_agg = summarize_table(all_condition_rows, key_cols=["task"])
    # router_agg = summarize_table(all_router_summary_rows, key_cols=["name"])

    # write_csv(csv_dir / "fused_summary_aggregate.csv", fused_summary_agg)
    # write_csv(csv_dir / "fused_per_class_aggregate.csv", fused_per_class_agg)
    # write_csv(csv_dir / "expert_per_class_aggregate.csv", expert_per_class_agg)
    # write_csv(csv_dir / "condition_aggregate.csv", condition_agg)
    # write_csv(csv_dir / "router_summary_aggregate.csv", router_agg)
    
    
    write_csv(csv_dir / "fused_summary_aggregate.csv", fused_summary_agg)
    write_csv(csv_dir / "fused_per_class_aggregate.csv", fused_per_class_agg)
    write_csv(csv_dir / "expert_per_class_aggregate.csv", expert_per_class_agg)
    write_csv(csv_dir / "expert_condition_aggregate.csv", expert_condition_agg)
    write_csv(csv_dir / "condition_aggregate.csv", condition_agg)
    write_csv(csv_dir / "router_summary_aggregate.csv", router_agg)
    write_csv(csv_dir / "router_per_class_aggregate.csv", router_per_class_agg)
    write_csv(csv_dir / "router_condition_aggregate.csv", router_condition_agg)
    

    save_json({"num_runs": len(run_dirs), "run_dirs": [str(p) for p in run_dirs], "checkpoint_name": args.checkpoint_name}, output_dir / "evaluation_manifest.json")

    if fused_summary_agg:
        plot_errorbar_summary(fused_summary_agg, metric_key="mAP", title="Fused MoE mAP Across Runs", path=plots_dir / "aggregate_fused_map.png", plot_cfg=plot_cfg)
        plot_errorbar_summary(fused_summary_agg, metric_key="macro_auroc", title="Fused MoE Macro AUROC Across Runs", path=plots_dir / "aggregate_fused_macro_auroc.png", plot_cfg=plot_cfg)
        plot_errorbar_summary(fused_summary_agg, metric_key="inference_ms", title="Fused MoE Inference Time Across Runs", path=plots_dir / "aggregate_fused_inference_ms.png", plot_cfg=plot_cfg)

    plot_mean_multirun_pr_curves(all_curve_payloads, TARGET_OBJECTS, "Mean Precision-Recall Curves Across Runs - Fused MoE", plots_dir / "aggregate_pr_fused_all_objects.png", plot_cfg)
    plot_mean_multirun_roc_curves(all_curve_payloads, TARGET_OBJECTS, "Mean ROC Curves Across Runs - Fused MoE", plots_dir / "aggregate_roc_fused_all_objects.png", plot_cfg)

    bundle = load_json(metadata_dir / "metadata_bundle.json")
    location_names = [k for k, _ in sorted(bundle["weather_to_id"].items(), key=lambda kv: kv[1])]
    illumination_names = [k for k, _ in sorted(bundle["scene_to_id"].items(), key=lambda kv: kv[1])]
    motion_names = [k for k, _ in sorted(bundle["time_to_id"].items(), key=lambda kv: kv[1])]

    if location_cms:
        cm = aggregate_confusion_matrices(location_cms)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        plot_matrix(cm, location_names, location_names, "Mean Location Confusion Matrix", plots_dir / "mean_confusion_location.png", plot_cfg=plot_cfg)
    if illumination_cms:
        cm = aggregate_confusion_matrices(illumination_cms)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        plot_matrix(cm, illumination_names, illumination_names, "Mean Illumination Confusion Matrix", plots_dir / "mean_confusion_illumination.png", plot_cfg=plot_cfg)
    if motion_cms:
        cm = aggregate_confusion_matrices(motion_cms)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        plot_matrix(cm, motion_names, motion_names, "Mean Motion Confusion Matrix", plots_dir / "mean_confusion_motion.png", plot_cfg=plot_cfg)

    print(f"[{timestamp()}] Evaluation complete.")
    print(f"[{timestamp()}] CSV files saved to: {csv_dir}")
    print(f"[{timestamp()}] PNG plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
