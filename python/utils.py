"""
Utilities Module for Metagenomic Scoring optimization.
Contains common helper functions for data loading, candidate evaluation,
benchmarking data logs, and data visualization.
"""

import os
import csv
from typing import Tuple, Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def load_metagenomic_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads metagenomic datasets from NumPy binary files and validates their types/shapes.

    Args:
        data_dir: Path to the directory containing npy files.

    Returns:
        Tuple of (A, y, T, S, F) as float32 arrays.
    """
    # Define paths
    path_A = os.path.join(data_dir, "matrix_A.npy")
    path_y = os.path.join(data_dir, "labels.npy")
    path_T = os.path.join(data_dir, "T.npy")
    path_S = os.path.join(data_dir, "S.npy")
    path_F = os.path.join(data_dir, "F.npy")

    # Load arrays
    for path in [path_A, path_y, path_T, path_S, path_F]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found: {path}. Run generate_data.py first.")

    A = np.load(path_A)
    y = np.load(path_y)
    T = np.load(path_T)
    S = np.load(path_S)
    F = np.load(path_F)

    # Strict float32 data type validations
    assert A.dtype == np.float32, f"matrix_A must be float32, got {A.dtype}"
    assert y.dtype == np.float32, f"labels must be float32, got {y.dtype}"
    assert T.dtype == np.float32, f"T must be float32, got {T.dtype}"
    assert S.dtype == np.float32, f"S must be float32, got {S.dtype}"
    assert F.dtype == np.float32, f"F must be float32, got {F.dtype}"

    # Shape and dimension consistency checks
    n_samples, n_items = A.shape
    assert y.shape == (n_samples,), f"labels shape {y.shape} mismatch with matrix_A samples {n_samples}"
    assert T.shape == (n_items,), f"T shape {T.shape} mismatch with matrix_A items {n_items}"
    assert S.shape == (n_items,), f"S shape {S.shape} mismatch with matrix_A items {n_items}"
    assert F.shape == (n_items,), f"F shape {F.shape} mismatch with matrix_A items {n_items}"

    return A, y, T, S, F

def evaluate_candidate(
    w: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    F: np.ndarray,
    A: np.ndarray,
    y: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluates a single weight candidate W on the metagenomic dataset.
    Highly optimized: uses vector operations and avoids redundant memory allocations.

    Args:
        w: Weight vector candidate of shape (3,) and dtype float32.
        T: Taxonomic profile of shape (N,) and dtype float32.
        S: Ecological profile of shape (N,) and dtype float32.
        F: Functional profile of shape (N,) and dtype float32.
        A: Sample-by-feature matrix of shape (M, N) and dtype float32.
        y: Binary labels of shape (M,).

    Returns:
        Tuple containing:
            - auc: The ROC AUC score for this candidate.
            - scores: The calculated sample scores array of shape (M,) and dtype float32.
            - w: The input weight vector.
    """
    # 1. Compute unified feature profile P as a weighted combination (vectorized)
    P = w[0] * T + w[1] * S + w[2] * F

    # 2. Compute sample scores using fast matrix-vector product
    scores = A.dot(P)

    # 3. Calculate ROC AUC score
    # Sklearn's roc_auc_score can handle float32 labels and scores
    auc = float(roc_auc_score(y, scores))

    return auc, scores, w

def save_benchmark_results(file_path: str, data: Dict[str, Any]) -> None:
    """
    Appends benchmark metrics to results/benchmark.csv.
    Creates the directory and file with headers if they do not exist.

    Args:
        file_path: Output CSV file path.
        data: Dict containing benchmark fields.
    """
    # Ensure folder exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Complete list of columns
    headers = [
        "implementation", "K", "n_samples", "n_items", "n_runs",
        "mean_execution_time", "std_execution_time", "best_auc",
        "best_w1", "best_w2", "best_w3", "signal_strength", "seed", "dtype"
    ]

    file_exists = os.path.exists(file_path)

    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def generate_plots(
    y: np.ndarray,
    scores: np.ndarray,
    best_w: np.ndarray,
    best_auc: float,
    output_dir: str
) -> None:
    """
    Generates basic diagnostic plots (score histogram and ROC curve)
    and saves them in the output directory.

    Args:
        y: True binary labels array of shape (M,).
        scores: Sample score predictions of shape (M,).
        best_w: Best weights array (W1, W2, W3).
        best_auc: Best ROC AUC score.
        output_dir: Directory where plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Plot 1: Histogram of scores for healthy vs diseased ---
    plt.figure(figsize=(7, 5))
    healthy_scores = scores[y == 0.0]
    diseased_scores = scores[y == 1.0]

    plt.hist(healthy_scores, bins=10, alpha=0.6, label="Healthy (y=0)", color="blue", edgecolor="black")
    plt.hist(diseased_scores, bins=10, alpha=0.6, label="Diseased (y=1)", color="orange", edgecolor="black")

    plt.xlabel("Metagenomic Score")
    plt.ylabel("Frequency")
    plt.title(f"Score Distribution (AUC: {best_auc:.4f})\nBest W: [{best_w[0]:.3f}, {best_w[1]:.3f}, {best_w[2]:.3f}]")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distribution.png"), dpi=150)
    plt.close()

    # --- Plot 2: ROC Curve ---
    fpr, tpr, _ = roc_curve(y, scores)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {best_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random (AUC = 0.5)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close()

    print(f"Plots successfully generated and saved to: {output_dir}/")
