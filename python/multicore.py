#!/usr/bin/env python3
"""
Multicore Parallel Random Search for Metagenomic Scoring
Evaluates K weight candidates across multiple CPU cores, timing runs,
logging results to results/benchmark.csv, and generating HPC plots.
"""

import os
import sys
import argparse
import time
import subprocess
import multiprocessing as mp
import numpy as np
from typing import List, Tuple, Optional

# Import shared utility functions
from utils import (
    load_metagenomic_data,
    evaluate_candidate,
    save_benchmark_results,
    find_matching_sequential_time,
    generate_hpc_plots,
    get_hardware_metadata
)

# Global variables within worker scope to avoid serializing data for each task
_A = None
_y = None
_T = None
_S = None
_F = None

def _worker_init(
    A: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    S: np.ndarray,
    F: np.ndarray
) -> None:
    """
    Initializer function run once when each pool worker process is spawned.
    Stores dataset arrays in worker module-level globals.
    """
    global _A, _y, _T, _S, _F
    _A = A
    _y = y
    _T = T
    _S = S
    _F = F

def _evaluate_chunk(candidate_chunk: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Worker task: Evaluates a chunk of candidate weight vectors sequentially
    and returns the local best result.
    """
    global _A, _y, _T, _S, _F
    
    best_auc = -1.0
    best_w = np.zeros(3, dtype=np.float32)
    best_scores = np.zeros(len(_y), dtype=np.float32)
    
    for w in candidate_chunk:
        auc, scores, eval_w = evaluate_candidate(w, _T, _S, _F, _A, _y)
        if auc > best_auc:
            best_auc = auc
            best_w[:] = eval_w
            best_scores[:] = scores
            
    return best_auc, best_w, best_scores

def run_multicore_search(
    data_dir: str,
    K: int,
    n_processes: int,
    n_runs: int,
    search_seed: int,
    data_seed: int,
    signal_strength: float,
    output_csv: str,
    plot_dir: str
) -> Tuple[float, float]:
    """
    Performs parallel search over K candidates using n_processes.
    Repeats search across n_runs for timing statistics.
    Logs metrics and calculates speedup relative to sequential.
    
    Returns:
        Tuple of (mean_execution_time, best_auc)
    """
    # 1. Load data
    A, y, T, S, F = load_metagenomic_data(data_dir)
    n_samples, n_items = A.shape

    print(f"\n--- Multiprocessing Run (P={n_processes}) ---")
    print(f"Dataset: N={n_items}, Samples={n_samples}, K={K}")
    
    # 2. Lookup sequential baseline time or generate it
    t_seq = find_matching_sequential_time(
        output_csv, K, n_samples, n_items, signal_strength, search_seed
    )
    if t_seq is None:
        print(f"No matching sequential baseline found in {output_csv} for configuration (K={K}, N={n_items}, seed={search_seed}).")
        print("Automatically executing sequential.py to record the baseline...")
        sequential_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sequential.py")
        cmd = [
            sys.executable,
            sequential_script,
            "-K", str(K),
            "--n_runs", str(n_runs),
            "--search_seed", str(search_seed),
            "--data_seed", str(data_seed),
            "--signal_strength", str(signal_strength),
            "--data_dir", data_dir,
            "--output_csv", output_csv,
            "--plot_dir", plot_dir
        ]
        subprocess.run(cmd, check=True)
        t_seq = find_matching_sequential_time(
            output_csv, K, n_samples, n_items, signal_strength, search_seed
        )
        if t_seq is None:
            raise RuntimeError("Failed to establish sequential baseline run.")
        print(f"Sequential baseline established: {t_seq:.4f}s")
    else:
        print(f"Using cached sequential baseline execution time: {t_seq:.4f}s")

    # 3. Generate candidates (using Dirichlet distribution, same sequence as sequential)
    rng = np.random.default_rng(search_seed)
    candidates = rng.dirichlet(np.ones(3), size=K).astype(np.float32)

    # Split candidates into chunks for the processes
    chunks = np.array_split(candidates, n_processes)

    run_times: List[float] = []
    overall_best_auc = -1.0
    overall_best_w = np.zeros(3, dtype=np.float32)
    overall_best_scores = np.zeros(n_samples, dtype=np.float32)

    # 4. Multi-run benchmark loop
    for run in range(1, n_runs + 1):
        # Start timer (strictly wrapping the multiprocessing pool and consolidation)
        t_start = time.perf_counter()
        
        # Create pool with initializer to load data into worker memory once
        # Explicitly setting start_method to 'spawn' for Windows safety
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=n_processes,
            initializer=_worker_init,
            initargs=(A, y, T, S, F)
        ) as pool:
            # Dispatch candidate chunks to workers
            results = pool.map(_evaluate_chunk, chunks)

        # Consolidate best candidates from all processes
        best_auc = -1.0
        best_w = np.zeros(3, dtype=np.float32)
        best_scores = np.zeros(n_samples, dtype=np.float32)
        
        for local_auc, local_w, local_scores in results:
            if local_auc > best_auc:
                best_auc = local_auc
                best_w[:] = local_w
                best_scores[:] = local_scores

        t_end = time.perf_counter()
        elapsed_time = t_end - t_start
        run_times.append(elapsed_time)

        if best_auc > overall_best_auc:
            overall_best_auc = best_auc
            overall_best_w[:] = best_w
            overall_best_scores[:] = best_scores

    mean_time = float(np.mean(run_times))
    std_time = float(np.std(run_times)) if len(run_times) > 1 else 0.0
    
    # Calculate HPC metrics
    speedup = float(t_seq / mean_time)
    efficiency = float(speedup / n_processes)

    print(f"Run Stats: Mean time = {mean_time:.4f}s (std: {std_time:.4f}s)")
    print(f"           Best AUC  = {overall_best_auc:.6f}")
    print(f"           Weights W = [{overall_best_w[0]:.6f}, {overall_best_w[1]:.6f}, {overall_best_w[2]:.6f}]")
    print(f"           Speedup   = {speedup:.2f}x")
    print(f"           Efficiency = {efficiency * 100:.1f}%")

    # 5. Log to CSV
    benchmark_data = {
        "implementation": "multicore",
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
        "dtype": "float32",
        "n_processes": n_processes,
        "speedup": speedup,
        "efficiency": efficiency
    }
    save_benchmark_results(output_csv, benchmark_data)
    print(f"Metrics saved to {output_csv}")

    return mean_time, overall_best_auc

def main() -> None:
    parser = argparse.ArgumentParser(description="Multicore Parallel Random Search optimization.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to data files (default: 'data').")
    parser.add_argument("-K", type=int, default=5000,
                        help="Number of weight candidates (default: 5000).")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of runs to repeat for timing stats (default: 5).")
    parser.add_argument("-P", "--n_processes", type=int, default=None,
                        help="Number of worker processes (default: cpu_count).")
    parser.add_argument("--search_seed", type=int, default=42,
                        help="Random seed for random search candidates (default: 42).")
    parser.add_argument("--data_seed", type=int, default=42,
                        help="Seed used during synthetic data generation (default: 42).")
    parser.add_argument("--signal_strength", type=float, default=0.5,
                        help="Signal strength beta used during data generation (default: 0.5).")
    parser.add_argument("--output_csv", type=str, default="results/benchmark.csv",
                        help="Output benchmark CSV path (default: 'results/benchmark.csv').")
    parser.add_argument("--plot_dir", type=str, default="results/plots",
                        help="Directory to save scaling plots (default: 'results/plots').")
    parser.add_argument("--sweep", action="store_true",
                        help="Run automated sweep for P=1, 2, 4, 8, cpu_count() and generate HPC plots.")

    args = parser.parse_args()

    # Determine process counts
    cpu_limit = os.cpu_count() or 1
    if args.sweep:
        # P = 1, 2, 4, 8, cpu_count() as requested
        process_sweep = [1, 2, 4, 8]
        if cpu_limit not in process_sweep:
            process_sweep.append(cpu_limit)
        # Filter process counts not exceeding the hardware limit (except P=1 which always runs)
        process_sweep = [p for p in process_sweep if p <= cpu_limit]
        # Sort and deduplicate
        process_sweep = sorted(list(set(process_sweep)))
        
        print(f"Starting automated process sweep over cores: {process_sweep}")
        for p in process_sweep:
            run_multicore_search(
                data_dir=args.data_dir,
                K=args.K,
                n_processes=p,
                n_runs=args.n_runs,
                search_seed=args.search_seed,
                data_seed=args.data_seed,
                signal_strength=args.signal_strength,
                output_csv=args.output_csv,
                plot_dir=args.plot_dir
            )
        
        # Once sweep completes, generate HPC plots
        print("\nSweep complete. Generating HPC plots...")
        generate_hpc_plots(
            csv_path=args.output_csv,
            plot_dir=args.plot_dir,
            K=args.K,
            n_samples=2 * len(np.load(os.path.join(args.data_dir, "labels.npy"))) // 2, # calculate M
            n_items=len(np.load(os.path.join(args.data_dir, "T.npy"))),
            signal_strength=args.signal_strength,
            seed=args.search_seed
        )
    else:
        p = args.n_processes or cpu_limit
        run_multicore_search(
            data_dir=args.data_dir,
            K=args.K,
            n_processes=p,
            n_runs=args.n_runs,
            search_seed=args.search_seed,
            data_seed=args.data_seed,
            signal_strength=args.signal_strength,
            output_csv=args.output_csv,
            plot_dir=args.plot_dir
        )
        
        # Generate plots for this specific run config
        generate_hpc_plots(
            csv_path=args.output_csv,
            plot_dir=args.plot_dir,
            K=args.K,
            n_samples=2 * len(np.load(os.path.join(args.data_dir, "labels.npy"))) // 2,
            n_items=len(np.load(os.path.join(args.data_dir, "T.npy"))),
            signal_strength=args.signal_strength,
            seed=args.search_seed
        )

if __name__ == "__main__":
    main()
