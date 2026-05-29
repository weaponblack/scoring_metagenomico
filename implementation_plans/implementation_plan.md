# Implementation Plan - Level 1: Metagenomic Scoring Sequential Baseline

This document presents the detailed design and implementation plan for **Level 1** of the metagenomic scoring parallel optimization project. Level 1 establishes a robust, highly modular, sequential Python baseline that serves as the gold standard for correctness, performance, and reproducibility before parallelization (multiprocessing, OpenMP, MPI, or CUDA) is introduced in later levels.

---

## Technical Concept & Mathematical Model

The goal of the system is to find an optimal weight vector $W = (W_1, W_2, W_3)$ that maximizes the Area Under the ROC Curve (AUC ROC) to distinguish between healthy and diseased samples.

### 1. Metagenomic Profiles
Each item $i \in \{1, \dots, N\}$ (representing a taxonomic unit, gene, or functional group) is characterized by three profiles:
*   $T_i \ge 0$: Taxonomic profile (reference abundance/activity)
*   $S_i \ge 0$: Ecological profile (environmental/niche behavior)
*   $F_i \ge 0$: Functional profile (metabolic/pathway capacity)

These are represented as 1D NumPy arrays of shape $(N,)$.

### 2. Weighted Feature Profile $P$
For a given weight candidate $W = (W_1, W_2, W_3)$, the unified feature profile $P \in \mathbb{R}^N$ is computed as a convex combination:
$$P_i = W_1 T_i + W_2 S_i + W_3 F_i, \quad \text{for } i = 1, \dots, N$$
Subject to constraints:
$$W_1 + W_2 + W_3 = 1, \quad W_j \ge 0 \quad (\forall j \in \{1, 2, 3\})$$

### 3. Sample Score Calculation
Let $A \in \mathbb{R}^{10 \times N}$ be the sample-by-feature matrix representing 10 physical samples (5 healthy, 5 diseased). The projected score vector $\text{Score} \in \mathbb{R}^{10}$ is calculated via matrix-vector multiplication:
$$\text{Score} = A P$$
where each element $\text{Score}_k = \sum_{j=1}^N A_{k,j} P_j$ representing the sample score for sample $k$.

### 4. Classification & Objective
The binary labels are:
$$y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]$$
where $0$ indicates a healthy sample and $1$ indicates a diseased sample. We compute the AUC ROC metric between the actual binary labels $y$ and the continuous scores $\text{Score}$:
$$\text{Objective: } \max_{W} \text{AUC}(y, A P)$$

---

## 1. Synthetic Data Generation Logic

To ensure the problem is realistic, reproducible, and has a meaningful optimization landscape, we generate synthetic data with controlled statistical properties:

1.  **Profiles $T, S, F$**: Generated using uniform positive values in $[0, 1)$ or drawn from a Dirichlet distribution.
2.  **Target Weight Vector $W_{\text{target}}$**: We define a hidden "optimal" weight vector, e.g., $W_{\text{target}} = [0.6, 0.3, 0.1]$.
3.  **Target Profile $P_{\text{target}}$**: Computed as $P_{\text{target}} = W_{\text{target},1} T + W_{\text{target},2} S + W_{\text{target},3} F$.
4.  **Matrix $A$ Generation**:
    *   For healthy samples ($y_k = 0$, rows $k=0..4$):
        $$A_{k, j} \sim \text{Uniform}(0, 1)$$
    *   For diseased samples ($y_k = 1$, rows $k=5..9$):
        $$A_{k, j} \sim \text{Uniform}(0, 1) + \beta \cdot P_{\text{target}, j}$$
        where $\beta > 0$ represents the **signal amplitude** (e.g., $\beta = 0.5$).
5.  **Separability Principle**:
    *   When the evaluated weight vector $W$ is close to $W_{\text{target}}$, the dot product of diseased samples with $P$ will be significantly higher than the dot product of healthy samples, yielding an AUC ROC close to $1.0$.
    *   If $W$ deviates significantly from $W_{\text{target}}$, the distinction degrades, yielding a lower AUC ROC.
    *   This guarantees that the landscape has a defined optimum, avoiding completely random or trivial solutions.

---

## Proposed Project Structure

We will adhere strictly to the requested folder structure:

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

## Proposed Changes by Component

### Component 1: Data Generation (`data/generate_data.py`)
Implement the command-line data generator utilizing `argparse`.
*   **Parameters**: $N$ (number of features), seed (for reproducibility), and signal scale $\beta$.
*   **Outputs**: Creates the `data/` directory (if not exists) and saves `matrix_A.npy`, `labels.npy`, `T.npy`, `S.npy`, and `F.npy` in NumPy binary format.
*   **Design**: Functions with type annotations and high-quality docstrings.

### Component 2: Utilities Module (`python/utils.py`)
Provide common routines to separate computation logic from setup/visualization.
*   `load_metagenomic_data()`: Loads the saved `.npy` files and performs shape/type validations.
*   `save_benchmark_results(file_path, data)`: Appends benchmark runs to `results/benchmark.csv`.
*   `generate_plots(y, scores, best_w, best_auc, output_dir)`: Generates two high-quality figures:
    1.  **Score Histogram**: Overlaying healthy (green/blue) vs diseased (red/orange) sample scores to visually demonstrate separability.
    2.  **ROC Curve**: Plotting the True Positive Rate vs False Positive Rate for the optimal $W$.

### Component 3: Sequential Execution Baseline (`python/sequential.py`)
Implement the core sequential random search algorithm.
*   **Search space**: Generates $K$ candidates for $W$ using:
    `np.random.dirichlet(np.ones(3), size=K)`
*   **Optimization Loop**: Evaluates each candidate sequentially.
    *   $P = W_1 T + W_2 S + W_3 F$
    *   $\text{Score} = A P$
    *   $\text{AUC} = \text{roc\_auc\_score}(y, \text{Score})$
*   **Time Measurement**: Uses `time.perf_counter()` strictly wrapping the search and evaluation loop (excluding data loading and saving).
*   **Benchmark Log**: Logs the execution metrics to `results/benchmark.csv`.

### Component 4: Documentation (`README.md`)
Write a premium, detailed markdown document explaining:
*   Project introduction and context.
*   Mathematical formulas and logic.
*   Step-by-step installation and execution instructions.
*   Future parallelization plans (multiprocessing, OpenMP, MPI, CUDA).

---

## Verification Plan

### Automated Verification
We will verify the implementation by:
1.  Running `data/generate_data.py` with different $N$ values (e.g., $N=100$, $N=10000$) and a fixed seed.
2.  Executing `python/sequential.py` with different search limits $K$ (e.g., $K=1000$, $K=10000$) and $N=10000$.
3.  Inspecting the resulting `results/benchmark.csv` to ensure time, AUC, and weight details are correctly captured.
4.  Checking that `results/plots/` contains the generated histogram and ROC curves.

### Manual Verification
*   We will check that the histogram clearly visualizes the statistical distinction between healthy (0) and diseased (1) samples.
*   We will verify that the maximum AUC ROC converges toward 1.0 as $K$ increases, proving our synthetic data generation is statistically coherent.

---

## Open Questions & Review Required

> [!NOTE]
> *   **Visual Style**: I will apply a professional dark/light high-contrast theme for the Matplotlib plots (curated palettes, no default red/green/blue, beautiful grid lines) to match a premium scientific publication style.
> *   **Data Dimension defaults**: I propose default values of $N = 10000$ and $K = 5000$ for testing the baseline execution speed.
