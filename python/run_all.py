import subprocess
import statistics
import os
import sys
import csv
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_command_and_parse(cmd, name):
    print(f"Running {name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR running {name}:\n{result.stderr}")
        return None
    return result.stdout

def read_last_result(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"No hay resultados en {csv_path}")

    row = rows[-1]

    return (
        float(row["mean_execution_time"]),
        float(row["std_execution_time"]),
        float(row["best_auc"])
    )

def run_c_runs(cmd, n_runs):
    times = []
    auc = 0.0
    for r in range(n_runs):
        print(f"  Run {r+1}/{n_runs}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return None, None, None
        
        # Parse output: K=50000, P=4, Time=12.65, AUC=1.000000
        for line in result.stdout.split('\n'):
            if line.startswith("K="):
                parts = line.split(", ")
                t_val = float(parts[2].split("=")[1])
                auc_val = float(parts[3].split("=")[1])
                times.append(t_val)
                auc = auc_val
                break
    
    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean_time, std_time, auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=50000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    K = args.K
    n_runs = args.runs
    data_dir = args.data_dir
    threads_list = [1, 2, 4, 8]
    
    results = []

    benchmark_csv = "results/benchmark.csv"

    os.makedirs("results", exist_ok=True)

    if os.path.exists(benchmark_csv):
        os.remove(benchmark_csv)

    print("=== ORQUESTADOR GLOBAL HPC ===")
    
    # 1. Python Sequential (P=1)
    
    run_command_and_parse(
        [
            sys.executable,
            "python/sequential.py",
            "-K", str(K),
            "--n_runs", str(n_runs),
            "--data_dir", data_dir,
            "--output_csv", benchmark_csv,
            "--plot_dir", "results/plots"
        ],
        "Python Sequential"
    )

    m, s, a = read_last_result(benchmark_csv)
    
    results.append({
        "implementation": "python_seq",
        "workers": 1,
        "K": K,
        "mean_time": m,
        "std_time": s,
        "auc": a,
        "speedup": 1.0,
        "efficiency": 1.0
    })

    # 2. Python Multiprocessing

    base_time_mp = None

    for p in threads_list:

        run_command_and_parse(
            [
                sys.executable,
                "python/multicore.py",
                "-K", str(K),
                "-P", str(p),
                "--n_runs", str(n_runs),
                "--data_dir", data_dir,
                "--output_csv", benchmark_csv,
                "--plot_dir", "results/plots"
            ],
            f"Python Multiprocessing P={p}"
        )

        m, s, a = read_last_result(benchmark_csv)

        if p == 1:
            base_time_mp = m

        speedup = base_time_mp / m if base_time_mp and m > 0 else 0.0
        efficiency = speedup / p if speedup > 0 else 0.0

        results.append({
            "implementation": "python_mp",
            "workers": p,
            "K": K,
            "mean_time": m,
            "std_time": s,
            "auc": a,
            "speedup": speedup,
            "efficiency": efficiency
        })

    # 3. OpenMP
    base_time_omp = None
    for p in threads_list:
        print(f"Running OpenMP P={p}:")
        cmd = ["./C_OpenMP_MPI/scoring_openmp.exe", data_dir, str(K), str(p)]
        m, s, a = run_c_runs(cmd, n_runs)
        if m is not None:
            if p == 1:
                base_time_omp = m
            speedup = base_time_omp / m if base_time_omp and m > 0 else 0.0
            efficiency = speedup / p if speedup > 0 else 0.0
            results.append({
                "implementation": "openmp", "workers": p, "K": K,
                "mean_time": m, "std_time": s, "auc": a,
                "speedup": speedup, "efficiency": efficiency
            })

    # 4. MPI
    base_time_mpi = None
    for p in threads_list:
        print(f"Running MPI P={p}:")
        cmd = ["mpiexec", "-n", str(p), "./C_OpenMP_MPI/scoring_mpi.exe", data_dir, str(K)]
        m, s, a = run_c_runs(cmd, n_runs)
        if m is not None:
            if p == 1:
                base_time_mpi = m
            speedup = base_time_mpi / m if base_time_mpi and m > 0 else 0.0
            efficiency = speedup / p if speedup > 0 else 0.0
            results.append({
                "implementation": "mpi", "workers": p, "K": K,
                "mean_time": m, "std_time": s, "auc": a,
                "speedup": speedup, "efficiency": efficiency
            })

    # Export to CSV
    os.makedirs("results", exist_ok=True)
    csv_path = "results/run_all_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["implementation", "workers", "K", "mean_time", "std_time", "auc", "speedup", "efficiency"])
        writer.writeheader()
        for r in results:
            # Rounding for cleanliness
            r["mean_time"] = round(r["mean_time"], 4)
            r["std_time"] = round(r["std_time"], 4)
            r["speedup"] = round(r["speedup"], 2)
            r["efficiency"] = round(r["efficiency"], 2)
            writer.writerow(r)
    print(f"\nGenerado CSV unificado en {csv_path}")

    # Generate Plots
    plot_dir = "results/plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Prepare data for plotting
    plot_data = {"python_seq": {"x": [], "y": []}, "python_mp": {"x": [], "y": []}, "openmp": {"x": [], "y": []}, "mpi": {"x": [], "y": []}}
    speedup_data = {"python_mp": {"x": [], "y": []}, "openmp": {"x": [], "y": []}, "mpi": {"x": [], "y": []}}
    eff_data = {"python_mp": {"x": [], "y": []}, "openmp": {"x": [], "y": []}, "mpi": {"x": [], "y": []}}

    for r in results:
        imp = r["implementation"]
        w = r["workers"]
        plot_data[imp]["x"].append(w)
        plot_data[imp]["y"].append(r["mean_time"])
        if imp != "python_seq":
            speedup_data[imp]["x"].append(w)
            speedup_data[imp]["y"].append(r["speedup"])
            eff_data[imp]["x"].append(w)
            eff_data[imp]["y"].append(r["efficiency"] * 100)

    # 1. Execution Time (Log Scale recommended due to massive difference)
    plt.figure(figsize=(10, 6))
    plt.plot(plot_data["python_seq"]["x"], plot_data["python_seq"]["y"], 'ro-', label="Python Sequential")
    plt.plot(plot_data["python_mp"]["x"], plot_data["python_mp"]["y"], 'bo-', label="Python Multiprocessing")
    plt.plot(plot_data["openmp"]["x"], plot_data["openmp"]["y"], 'go-', label="C OpenMP")
    plt.plot(plot_data["mpi"]["x"], plot_data["mpi"]["y"], 'mo-', label="C MS-MPI")
    plt.xlabel("Number of Processes / Threads (P)")
    plt.ylabel("Execution Time (seconds)")
    plt.yscale("log")
    plt.title(f"HPC Scaling: Absolute Execution Time (K={K})\nLog Scale")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xticks(threads_list)
    plt.savefig(os.path.join(plot_dir, "execution_time_vs_processes.png"), dpi=150)
    plt.close()

    # 2. Speedup (Linear)
    plt.figure(figsize=(10, 6))
    plt.plot(speedup_data["python_mp"]["x"], speedup_data["python_mp"]["y"], 'bo-', label="Python Multiprocessing")
    plt.plot(speedup_data["openmp"]["x"], speedup_data["openmp"]["y"], 'go-', label="C OpenMP")
    plt.plot(speedup_data["mpi"]["x"], speedup_data["mpi"]["y"], 'mo-', label="C MS-MPI")
    plt.plot(threads_list, threads_list, 'k--', label="Ideal Linear Speedup")
    plt.xlabel("Number of Processes / Threads (P)")
    plt.ylabel("Speedup (x) relative to own P=1")
    plt.title(f"HPC Scaling: Intrinsic Parallel Speedup (K={K})")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.xticks(threads_list)
    plt.savefig(os.path.join(plot_dir, "speedup_vs_processes.png"), dpi=150)
    plt.close()

    # 3. Efficiency
    plt.figure(figsize=(10, 6))
    plt.plot(eff_data["python_mp"]["x"], eff_data["python_mp"]["y"], 'bo-', label="Python Multiprocessing")
    plt.plot(eff_data["openmp"]["x"], eff_data["openmp"]["y"], 'go-', label="C OpenMP")
    plt.plot(eff_data["mpi"]["x"], eff_data["mpi"]["y"], 'mo-', label="C MS-MPI")
    plt.axhline(100.0, color='k', linestyle='--', label="Ideal Efficiency (100%)")
    plt.xlabel("Number of Processes / Threads (P)")
    plt.ylabel("Parallel Efficiency (%)")
    plt.title(f"HPC Scaling: Parallel Efficiency (K={K})")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.xticks(threads_list)
    plt.savefig(os.path.join(plot_dir, "efficiency_vs_processes.png"), dpi=150)
    plt.close()

    print(f"Generadas 3 gráficas combinadas en {plot_dir}/")

if __name__ == "__main__":
    main()
