"""
Utilities Module for Metagenomic Scoring optimization.
Contains common helper functions for data loading, candidate evaluation,
benchmarking data logs, and data visualization.
"""

import os
import csv
import platform
import sys
from typing import Tuple, Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd

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

def get_hardware_metadata() -> Dict[str, Any]:
    """
    Retrieves information about the current CPU model, counts, platform, and Python runtime.

    Returns:
        Dict containing hardware info.
    """
    cpu_name = platform.processor()
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            cpu_name = cpu_name.strip()
        except Exception:
            pass

    return {
        "cpu_name": cpu_name,
        "cpu_count": os.cpu_count(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": sys.version.split()[0]
    }

def save_benchmark_results(file_path: str, data: Dict[str, Any]) -> None:
    """
    Appends benchmark metrics to results/benchmark.csv.
    Creates the directory and file with headers if they do not exist.
    Upgrades existing headers in place if they are old/incomplete.

    Args:
        file_path: Output CSV file path.
        data: Dict containing benchmark fields.
    """
    # Ensure folder exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Complete list of columns including Level 2 metrics
    headers = [
        "implementation", "K", "n_samples", "n_items", "n_runs",
        "mean_execution_time", "std_execution_time", "best_auc",
        "best_w1", "best_w2", "best_w3", "signal_strength", "seed", "dtype",
        "n_processes", "speedup", "efficiency",
        "cpu_name", "cpu_count", "platform", "python_version"
    ]

    file_exists = os.path.exists(file_path)

    # Read existing rows and check if we need to upgrade the header
    existing_rows = []
    need_header_upgrade = False
    if file_exists:
        try:
            with open(file_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header_row = next(reader)
                if len(header_row) < len(headers):
                    need_header_upgrade = True
                    # Re-open to read all data as dicts
                    f.seek(0)
                    dict_reader = csv.DictReader(f)
                    for row in dict_reader:
                        existing_rows.append(row)
        except Exception as e:
            print(f"Warning: could not read existing benchmark CSV for upgrade: {e}")

    # Enrich with hardware metadata if not already provided
    enriched_data = data.copy()
    hw_info = get_hardware_metadata()
    for k, v in hw_info.items():
        if k not in enriched_data:
            enriched_data[k] = v

    if need_header_upgrade:
        print(f"Upgrading benchmark CSV header at {file_path} to include new metrics and metadata.")
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(row)
            writer.writerow(enriched_data)
    else:
        with open(file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(enriched_data)

def find_matching_sequential_time(
    file_path: str,
    K: int,
    n_samples: int,
    n_items: int,
    signal_strength: float,
    seed: int
) -> float:
    """
    Searches the benchmark CSV for a matching sequential implementation run
    with the exact same parameters and returns its mean execution time.

    Returns:
        float: The mean execution time of the matching sequential run, or None if not found.
    """
    if not os.path.exists(file_path):
        return None

    try:
        df = pd.read_csv(file_path)
        # Filter for sequential matching runs
        match_cond = (
            (df["implementation"] == "sequential") &
            (df["K"] == K) &
            (df["n_samples"] == n_samples) &
            (df["n_items"] == n_items) &
            (abs(df["signal_strength"] - signal_strength) < 1e-5) &
            (df["seed"] == seed)
        )
        matches = df[match_cond]
        if not matches.empty:
            # Return the latest match
            return float(matches.iloc[-1]["mean_execution_time"])
    except Exception as e:
        print(f"Warning: failed reading matching sequential time from CSV: {e}")
    
    return None

def estimate_amdahl_fraction(processes: np.ndarray, speedups: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Estimates the sequential fraction (f) from observed speedups using Amdahl's Law:
    S(P) = 1 / (f + (1 - f) / P)
    Uses a linear least-squares fit on: 1/S(P) - 1/P = f * (1 - 1/P)

    Args:
        processes: Array of process counts (P > 1).
        speedups: Array of corresponding observed speedups.

    Returns:
        Tuple of (f, theoretical_speedups)
    """
    processes = np.array(processes, dtype=np.float32)
    speedups = np.array(speedups, dtype=np.float32)

    # Exclude P=1 where X = 1 - 1/P = 0
    mask = processes > 1
    if not np.any(mask):
        f = 1.0
    else:
        X = 1.0 - 1.0 / processes[mask]
        Y = 1.0 / speedups[mask] - 1.0 / processes[mask]
        # Least squares slope without intercept
        f = float(np.sum(X * Y) / np.sum(X**2))
        f = max(0.0, min(1.0, f))  # Keep in valid physical range

    theoretical_speedups = 1.0 / (f + (1.0 - f) / processes)
    return f, theoretical_speedups

def generate_hpc_plots(
    csv_path: str,
    plot_dir: str,
    K: int,
    n_samples: int,
    n_items: int,
    signal_strength: float,
    seed: int
) -> None:
    """
    Reads benchmark.csv, filters for the multicore executions matching the given parameters,
    and plots execution time, speedup, and efficiency curves.
    """
    if not os.path.exists(csv_path):
        print(f"Benchmark file not found at {csv_path}. Cannot generate HPC plots.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading benchmark CSV: {e}")
        return

    # Filter matching parameters
    match_cond = (
        (df["K"] == K) &
        (df["n_samples"] == n_samples) &
        (df["n_items"] == n_items) &
        (abs(df["signal_strength"] - signal_strength) < 1e-5) &
        (df["seed"] == seed)
    )
    df_match = df[match_cond].copy()

    if df_match.empty:
        print("No matching benchmark results found to generate HPC plots.")
        return

    # Extract sequential time
    seq_runs = df_match[df_match["implementation"] == "sequential"]
    if seq_runs.empty:
        print("No sequential baseline run found for this configuration. Cannot plot speedup/efficiency.")
        return
    t_seq = seq_runs.iloc[-1]["mean_execution_time"]

    # Extract multicore runs, group by n_processes
    mc_runs = df_match[df_match["implementation"] == "multicore"]
    if mc_runs.empty:
        print("No multicore runs found to plot.")
        return

    # Group and compute average execution time per process count
    mc_grouped = mc_runs.groupby("n_processes").agg(
        mean_time=("mean_execution_time", "mean")
    ).reset_index()

    # Create a full set including sequential as P=1
    plot_df = pd.DataFrame({
        "n_processes": np.concatenate(([1], mc_grouped["n_processes"].values)),
        "mean_time": np.concatenate(([t_seq], mc_grouped["mean_time"].values))
    })
    plot_df = plot_df.sort_values("n_processes").drop_duplicates(subset=["n_processes"])

    # Calculate actual speedup and efficiency
    plot_df["speedup"] = t_seq / plot_df["mean_time"]
    plot_df["efficiency"] = plot_df["speedup"] / plot_df["n_processes"]

    proc_arr = plot_df["n_processes"].values
    speedup_arr = plot_df["speedup"].values

    # Fit Amdahl's Law
    f, theoretical_speedup = estimate_amdahl_fraction(proc_arr, speedup_arr)

    # Save details to txt file for the technical report
    os.makedirs(plot_dir, exist_ok=True)
    report_path = os.path.join(plot_dir, "amdahl_analysis.txt")
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(f"--- Amdahl's Law Analysis ---\n")
        rf.write(f"Dataset Size: N={n_items}, K={K}, Samples={n_samples}\n")
        rf.write(f"Sequential Time (baseline): {t_seq:.4f} s\n")
        rf.write(f"Estimated Serial Fraction (f): {f:.6f} ({f * 100:.2f}%)\n")
        rf.write(f"Estimated Parallel Fraction (1-f): {1 - f:.6f} ({(1 - f) * 100:.2f}%)\n")
        rf.write(f"Theoretical Max Speedup (P -> infinity): {1.0 / max(f, 1e-9):.2f}x\n\n")
        rf.write(f"P\tObserved Time (s)\tObserved Speedup\tTheoretical Speedup\tParallel Efficiency\n")
        for i, p in enumerate(proc_arr):
            rf.write(f"{int(p)}\t{plot_df['mean_time'].iloc[i]:.4f}\t\t{speedup_arr[i]:.2f}x\t\t\t{theoretical_speedup[i]:.2f}x\t\t\t{plot_df['efficiency'].iloc[i] * 100:.1f}%\n")

    print(f"Amdahl analysis saved to {report_path}")

    # Plot 1: Execution Time vs Processes
    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["n_processes"], plot_df["mean_time"], marker="o", color="#2c3e50", lw=2, label="Measured Time")
    plt.xlabel("Number of Processes (P)")
    plt.ylabel("Execution Time (seconds)")
    plt.title(f"HPC Scaling: Execution Time vs Processes\n(N={n_items}, K={K})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(plot_df["n_processes"])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "execution_time_vs_processes.png"), dpi=150)
    plt.close()

    # Plot 2: Speedup vs Processes (Observed vs Linear vs Amdahl)
    plt.figure(figsize=(8, 5))
    plt.plot(proc_arr, speedup_arr, marker="s", color="#e67e22", lw=2, label="Observed Speedup")
    plt.plot(proc_arr, proc_arr, linestyle="--", color="#7f8c8d", label="Ideal Linear Speedup")
    plt.plot(proc_arr, theoretical_speedup, linestyle="-.", color="#2980b9", label=f"Amdahl Fit (f={f:.3f})")
    plt.xlabel("Number of Processes (P)")
    plt.ylabel("Speedup (x)")
    plt.title(f"HPC Scaling: Speedup vs Processes\n(N={n_items}, K={K})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(proc_arr)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "speedup_vs_processes.png"), dpi=150)
    plt.close()

    # Plot 3: Parallel Efficiency vs Processes
    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["n_processes"], plot_df["efficiency"] * 100, marker="^", color="#27ae60", lw=2, label="Observed Efficiency")
    plt.axhline(y=100.0, linestyle="--", color="#7f8c8d", label="Ideal Efficiency (100%)")
    plt.xlabel("Number of Processes (P)")
    plt.ylabel("Efficiency (%)")
    plt.title(f"HPC Scaling: Efficiency vs Processes\n(N={n_items}, K={K})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(plot_df["n_processes"])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "efficiency_vs_processes.png"), dpi=150)
    plt.close()

    print(f"HPC scaling plots saved to {plot_dir}/")

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
