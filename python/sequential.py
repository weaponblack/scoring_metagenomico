#!/usr/bin/env python3
"""
Sequential Random Search Baseline for Metagenomic Scoring
Evaluates K weight candidates across n_runs, logging execution statistics
to results/benchmark.csv and generating ROC and score distribution plots.
"""

import os
import argparse
import time
import numpy as np
from typing import List, Tuple

# Import modular helper functions from utils
from utils import (
    load_metagenomic_data,
    evaluate_candidate,
    save_benchmark_results,
    generate_plots
)

def run_sequential_search(
    data_dir: str,
    K: int,
    n_runs: int,
    search_seed: int,
    data_seed: int,
    signal_strength: float,
    output_csv: str,
    plot_dir: str
) -> None:
    """
    Executes sequential random search to optimize weight vector W.

    Args:
        data_dir: Directory containing input .npy files.
        K: Number of random weight candidates to generate.
        n_runs: Number of times to repeat the optimization for benchmarking stability.
        search_seed: Seed for candidate generation.
        data_seed: Seed used during data generation (logged to CSV).
        signal_strength: Beta parameter used during data generation (logged to CSV).
        output_csv: Path to save/append benchmark CSV results.
        plot_dir: Path to save visual plots.
    """
    # 1. Load and validate data
    A, y, T, S, F = load_metagenomic_data(data_dir)
    n_samples, n_items = A.shape

    print("Dataset Loaded Successfully:")
    print(f"  Matrix A: {A.shape} (dtype: {A.dtype})")
    print(f"  Labels y: {y.shape} (dtype: {y.dtype})")
    print(f"  Profile T: {T.shape} (dtype: {T.dtype})")
    print(f"  Profile S: {S.shape} (dtype: {S.dtype})")
    print(f"  Profile F: {F.shape} (dtype: {F.dtype})")
    print(f"Starting Sequential Random Search (K={K}, runs={n_runs})...")

    # 2. Generate weight candidates using Dirichlet distribution (cast to float32)
    # Using Dirichlet guarantees: W1 + W2 + W3 = 1 and Wi >= 0
    rng = np.random.default_rng(search_seed)
    candidates = rng.dirichlet(np.ones(3), size=K).astype(np.float32)

    # Variables to track benchmarking times
    run_times: List[float] = []
    
    # Best candidate tracking variables
    overall_best_auc = -1.0
    overall_best_w = np.zeros(3, dtype=np.float32)
    overall_best_scores = np.zeros(n_samples, dtype=np.float32)

    # 3. Benchmark repetition loop
    for run in range(1, n_runs + 1):
        print(f"  Run {run}/{n_runs}...")
        
        best_auc = -1.0
        best_w = np.zeros(3, dtype=np.float32)
        best_scores = np.zeros(n_samples, dtype=np.float32)
        
        # Start timing (strictly wrapping the search and evaluation loop)
        t_start = time.perf_counter()
        
        for i in range(K):
            w = candidates[i]
            auc, scores, eval_w = evaluate_candidate(w, T, S, F, A, y)
            
            if auc > best_auc:
                best_auc = auc
                # Using in-place copy to minimize new allocations
                best_w[:] = eval_w
                best_scores[:] = scores

        t_end = time.perf_counter()
        elapsed_time = t_end - t_start
        run_times.append(elapsed_time)
        print(f"    Completed in {elapsed_time:.4f}s | Best AUC: {best_auc:.6f}")

        # Keep track of the absolute best candidate across all runs for visualization
        if best_auc > overall_best_auc:
            overall_best_auc = best_auc
            overall_best_w[:] = best_w
            overall_best_scores[:] = best_scores

    # 4. Compute benchmark statistics
    mean_time = float(np.mean(run_times))
    std_time = float(np.std(run_times)) if len(run_times) > 1 else 0.0

    print("\nBenchmark Summary:")
    print(f"  Mean Search Time: {mean_time:.6f}s")
    print(f"  Std Search Time:  {std_time:.6f}s")
    print(f"  Best AUC ROC:     {overall_best_auc:.6f}")
    print(f"  Best Weights W:   [{overall_best_w[0]:.6f}, {overall_best_w[1]:.6f}, {overall_best_w[2]:.6f}]")

    # 5. Log results to CSV
    benchmark_data = {
        "implementation": "sequential",
        "K": K,
        "n_samples": n_samples,
        "n_items": n_items,
        "n_runs": n_runs,
        "mean_execution_time": mean_time,
        "std_execution_time": std_time,
        "best_auc": overall_best_auc,
        "best_w1": overall_best_w[0],
        "best_w2": overall_best_w[1],
        "best_w3": overall_best_w[2],
        "signal_strength": signal_strength,
        "seed": search_seed,
        "dtype": "float32"
    }
    
    save_benchmark_results(output_csv, benchmark_data)
    print(f"Benchmark logged to: {output_csv}")

    # 6. Generate diagnostic visualizations
    generate_plots(
        y=y,
        scores=overall_best_scores,
        best_w=overall_best_w,
        best_auc=overall_best_auc,
        output_dir=plot_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sequential random search optimization.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to the directory containing synthetic data files (default: 'data').")
    parser.add_argument("-K", type=int, default=5000,
                        help="Number of weight candidates to evaluate (default: 5000).")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of runs to repeat for benchmarking statistics (default: 5).")
    parser.add_argument("--search_seed", type=int, default=42,
                        help="Random seed for random search candidates (default: 42).")
    parser.add_argument("--data_seed", type=int, default=42,
                        help="Seed used during synthetic data generation, for logging purposes (default: 42).")
    parser.add_argument("--signal_strength", type=float, default=0.5,
                        help="Signal strength beta used during data generation, for logging purposes (default: 0.5).")
    parser.add_argument("--output_csv", type=str, default="results/benchmark.csv",
                        help="CSV file where benchmarks are logged (default: 'results/benchmark.csv').")
    parser.add_argument("--plot_dir", type=str, default="results/plots",
                        help="Directory to save diagnostic plots (default: 'results/plots').")

    args = parser.parse_args()

    run_sequential_search(
        data_dir=args.data_dir,
        K=args.K,
        n_runs=args.n_runs,
        search_seed=args.search_seed,
        data_seed=args.data_seed,
        signal_strength=args.signal_strength,
        output_csv=args.output_csv,
        plot_dir=args.plot_dir
    )
