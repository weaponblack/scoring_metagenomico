#!/usr/bin/env python3
"""
Metagenomic Synthetic Data Generator
Generates high-performance float32 synthetic datasets for metagenomic scoring.
Injects a statistical signal into diseased samples using a predefined optimal profile.
"""

import os
import argparse
import numpy as np

def generate_synthetic_data(
    n_samples_per_class: int,
    n_items: int,
    signal_strength: float,
    seed: int,
    output_dir: str
) -> None:
    """
    Generates synthetic metagenomic data and saves it in NumPy binary format.

    Args:
        n_samples_per_class: Number of samples in each binary class.
        n_items: Number of features (metagenomic items).
        signal_strength: The statistical separation factor (beta).
        seed: Random seed for reproducibility.
        output_dir: Directory where the generated files will be stored.
    """
    print(f"Generating synthetic metagenomic data...")
    print(f"  Samples per class: {n_samples_per_class} (Total: {2 * n_samples_per_class})")
    print(f"  Items (features):  {n_items}")
    print(f"  Signal strength (beta): {signal_strength}")
    print(f"  Random seed:       {seed}")
    print(f"  Precision:         float32")

    # 1. Initialize random number generator
    rng = np.random.default_rng(seed)

    # 2. Generate taxonomic (T), ecological (S), and functional (F) profiles
    T = rng.random(n_items, dtype=np.float32)
    S = rng.random(n_items, dtype=np.float32)
    F = rng.random(n_items, dtype=np.float32)

    # 3. Define target weight vector W_target and target profile P_target
    # Chosen target: W_target = (0.6, 0.3, 0.1)
    w_target = np.array([0.6, 0.3, 0.1], dtype=np.float32)
    P_target = w_target[0] * T + w_target[1] * S + w_target[2] * F

    # 4. Generate matrix A of shape (2 * n_samples_per_class, n_items)
    n_samples = 2 * n_samples_per_class
    A = np.empty((n_samples, n_items), dtype=np.float32)

    # Healthy samples (first half, y = 0)
    A[:n_samples_per_class] = rng.random((n_samples_per_class, n_items), dtype=np.float32)

    # Diseased samples (second half, y = 1)
    # The signal is added as beta * P_target to each diseased sample
    A[n_samples_per_class:] = rng.random((n_samples_per_class, n_items), dtype=np.float32) + signal_strength * P_target

    # 5. Create label vector y of shape (2 * n_samples_per_class,)
    y = np.zeros(n_samples, dtype=np.float32)
    y[n_samples_per_class:] = 1.0

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 6. Save data files
    np.save(os.path.join(output_dir, "matrix_A.npy"), A)
    np.save(os.path.join(output_dir, "labels.npy"), y)
    np.save(os.path.join(output_dir, "T.npy"), T)
    np.save(os.path.join(output_dir, "S.npy"), S)
    np.save(os.path.join(output_dir, "F.npy"), F)

    print(f"Data generation complete! Files saved to: {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic metagenomic dataset.")
    parser.add_argument("--n_samples_per_class", type=int, default=5,
                        help="Number of samples per class (default: 5, total 10).")
    parser.add_argument("--n_items", type=int, default=10000,
                        help="Number of metagenomic items (default: 10000).")
    parser.add_argument("--signal_strength", type=float, default=0.5,
                        help="Separation signal strength beta (default: 0.5).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Directory to save the generated numpy files (default: 'data').")

    args = parser.parse_args()

    # If output_dir is relative, resolve it relative to workspace or absolute path
    # Here, let's keep it relative which resolved relative to cwd, standard behavior
    generate_synthetic_data(
        n_samples_per_class=args.n_samples_per_class,
        n_items=args.n_items,
        signal_strength=args.signal_strength,
        seed=args.seed,
        output_dir=args.output_dir
    )
