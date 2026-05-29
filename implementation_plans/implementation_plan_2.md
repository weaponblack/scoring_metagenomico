# Implementation Plan - Level 1: Metagenomic Scoring Sequential Baseline (Revised)

This document outlines the revised design and implementation plan for **Level 1** of the metagenomic scoring parallel optimization project. It incorporates all modifications requested for dataset scalability, float32 precision, memory efficiency, isolated candidate evaluation, rigorous benchmarking with repeated runs, and multiprocessing-friendly architecture.

---

## Technical Concept & Mathematical Model

The goal of the system is to find an optimal weight vector $W = (W_1, W_2, W_3)$ that maximizes the Area Under the ROC Curve (AUC ROC) to distinguish between healthy and diseased samples.

### 1. Metagenomic Profiles (Single Precision)
Each item $i \in \{1, \dots, N\}$ (where $N = \text{n\_items}$) is characterized by three profiles:
*   $T_i \ge 0$: Taxonomic profile, shape $(N,)$, type `float32`
*   $S_i \ge 0$: Ecological profile, shape $(N,)$, type `float32`
*   $F_i \ge 0$: Functional profile, shape $(N,)$, type `float32`

### 2. Weighted Feature Profile $P$
For a given weight candidate $W = (W_1, W_2, W_3) \in \mathbb{R}^3$ represented in `float32`:
$$P_i = W_1 T_i + W_2 S_i + W_3 F_i, \quad \text{for } i = 1, \dots, N$$
Subject to:
$$W_1 + W_2 + W_3 = 1, \quad W_j \ge 0 \quad (\forall j \in \{1, 2, 3\})$$
The resulting array $P$ is of shape $(N,)$ and type `float32`.

### 3. Scalable Sample Score Calculation
Instead of a fixed size of 10 samples, the system is fully scalable:
*   $M = 2 \times \text{n\_samples\_per\_class}$ represents the total number of samples.
*   $A \in \mathbb{R}^{M \times N}$: Sample-by-feature matrix, type `float32`.
    *   Rows $0 \dots (M/2 - 1)$ represent Healthy samples.
    *   Rows $(M/2) \dots (M - 1)$ represent Diseased samples.
*   $\text{Score} = A P$ (Matrix-vector product yielding a vector of shape $(M,)$ and type `float32`).

### 4. Classification labels $y$
The label vector $y \in \mathbb{R}^M$ is pre-allocated as:
$$y_k = 0 \text{ for } k < M/2, \quad y_k = 1 \text{ for } k \ge M/2$$
and cast to `float32` or integer depending on `roc_auc_score` requirements (we will keep labels as `int32` or `float32`).

---

## 1. Configurable Synthetic Data Generation

To support full scalability tests, the data generator will use the following parameters:
*   `n_samples_per_class`: Number of healthy (and diseased) samples.
*   `n_items`: Number of metagenomic features ($N$).
*   `signal_strength` ($\beta$): Determines statistical separability.
*   `seed`: Integer seed for NumPy random state reproducibility.

### Generation Steps (`data/generate_data.py`)
1.  Initialize random number generator: `rng = np.random.default_rng(seed)`.
2.  Generate positive reference profiles:
    *   $T, S, F \sim \text{Uniform}(0, 1)$ of shape $(N,)$, cast to `float32`.
3.  Define hidden optimal weights: e.g., $W_{\text{target}} = [0.6, 0.3, 0.1]$ (representing `float32`).
4.  Compute target profile:
    $$P_{\text{target}} = 0.6 \cdot T + 0.3 \cdot S + 0.1 \cdot F \quad (\text{shape } (N,), \text{dtype } \text{float32})$$
5.  Generate sample-by-feature matrix $A$:
    *   Healthy samples ($y_k = 0$, $k=0 \dots \text{n\_samples\_per\_class}-1$):
        $$A_{k, j} \sim \text{Uniform}(0, 1) \quad (\text{dtype } \text{float32})$$
    *   Diseased samples ($y_k = 1$, $k=\text{n\_samples\_per\_class} \dots 2 \cdot \text{n\_samples\_per\_class}-1$):
        $$A_{k, j} \sim \text{Uniform}(0, 1) + \beta \cdot P_{\text{target}, j} \quad (\text{dtype } \text{float32})$$
6.  Save all arrays in `data/` in `.npy` format.

---

## 2. Memory & Performance Optimizations

To avoid unnecessary allocations and overhead:
*   All arrays ($A$, $T$, $S$, $F$, $P$, $y$, $W$) are explicitly instantiated with `dtype=np.float32`.
*   Data types are set during array creation; no dynamic type casting is performed inside critical execution loops.
*   `np.dot` or the `@` operator will be used directly to perform matrix-vector multiplication in high-performance C/BLAS space.
*   Intermediate array generation is minimized by keeping candidate evaluations vectorised.

---

## 3. Isolated Candidate Evaluation

To allow direct reuse in future parallel steps (multiprocessing, CUDA, etc.), the core optimization step is fully isolated into a standalone function:

```python
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
```

---

## 4. Multi-Run Benchmarking Strategy

To avoid measurement noise and unstable benchmarks, we implement multi-run execution:
*   The search will run `n_runs` times (e.g., 5 runs).
*   For each run, we measure the elapsed search time using `time.perf_counter()`.
*   We compute the **mean** and **standard deviation (std)** of the search times across all runs.
*   We log these detailed statistics in `results/benchmark.csv`.

### Extended CSV Columns
We will store the following columns in `results/benchmark.csv`:
1.  `implementation`: String label (e.g., `"sequential"`).
2.  `K`: Number of random search candidates.
3.  `n_samples`: Total number of samples ($2 \times \text{n\_samples\_per\_class}$).
4.  `n_items`: Number of metagenomic features ($N$).
5.  `n_runs`: Number of benchmark repetitions.
6.  `mean_execution_time`: Average search execution time in seconds.
7.  `std_execution_time`: Standard deviation of search execution time in seconds.
8.  `best_auc`: Maximum AUC ROC achieved.
9.  `best_w1`, `best_w2`, `best_w3`: The weights yielding the highest AUC.
10. `signal_strength`: Beta parameter.
11. `seed`: Random seed.
12. `dtype`: Data type used (always `"float32"`).

---

## Project Directory Map

We follow the structure exactly:

```
scoring_metagenomico/
│
├── data/
│   ├── generate_data.py
│   ├── matrix_A.npy
│   ├── labels.npy
│   ├── T.npy
│   ├── S.npy
│   └── F.npy
│
├── python/
│   ├── sequential.py
│   └── utils.py
│
├── results/
│   ├── benchmark.csv
│   └── plots/
│
└── README.md
```

---

## Multiprocessing Preparation & Script Design

To ensure the sequential codebase transitions smoothly to multiprocessing and other HPC runtimes:
*   No execution code is placed in global scope.
*   All executables use the standard guard: `if __name__ == "__main__":`.
*   Execution steps are clearly divided:
    *   **Loading**: Handled by `utils.load_metagenomic_data`.
    *   **Evaluation**: Handled by `utils.evaluate_candidate`.
    *   **Benchmark Log**: Handled by `utils.save_benchmark_results`.
    *   **Visualization**: Handled by `utils.generate_plots`.

---

## Verification Plan

### Automated Verification
1.  Generate a test dataset:
    `python data/generate_data.py --n_samples_per_class 50 --n_items 10000 --signal_strength 0.5 --seed 42`
2.  Run the sequential search and benchmark pipeline:
    `python python/sequential.py --K 2000 --n_runs 5 --seed 42`
3.  Check that:
    *   `results/benchmark.csv` is populated with the extended schema (including `mean_execution_time` and `std_execution_time`).
    *   `results/plots/` contains stable, clear diagnostic plots (ROC curve and a score histogram separating healthy/diseased).
    *   Both scripts run successfully without memory leaks or runtime errors.

### Manual Verification
*   Confirm that the best AUC is close to 1.0 when K is high, verifying the statistical signal added to diseased samples.
*   Verify that `best_w1, best_w2, best_w3` approximate the target weight vector $W_{\text{target}} = [0.6, 0.3, 0.1]$ used in data generation.
*   Ensure all array properties match `float32` exactly.
