import subprocess
import statistics
import os
import csv

K = 50000
runs = 3
threads_list = [1, 2, 4, 8]
data_dir = "../data"
csv_file = "../results/openmp_benchmark.csv"

print(f"--- Ejecutando Benchmark OpenMP (K={K}, runs={runs}) ---")

baseline_mean = None
results = []

for p in threads_list:
    times = []
    best_auc = 0
    best_w = ""
    for r in range(runs):
        cmd = ["scoring_openmp.exe", data_dir, str(K), str(p)]
        # Need to use shell=True or the exact path in Windows
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running command: {result.stderr}")
            continue
            
        for line in result.stdout.split('\n'):
            if line.startswith("K="):
                parts = line.split(", ")
                time_val = float(parts[2].split("=")[1])
                auc_val = float(parts[3].split("=")[1])
                w1 = parts[4].split("=")[1]
                w2 = parts[5].split("=")[1]
                w3 = parts[6].split("=")[1]
                times.append(time_val)
                best_auc = auc_val
                best_w = f"[{w1}, {w2}, {w3}]"
                break
                
    if not times:
        print(f"P={p}: FAILED")
        continue
        
    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    
    if p == 1:
        baseline_mean = mean_time
        speedup = 1.0
        efficiency = 1.0
    else:
        speedup = baseline_mean / mean_time if mean_time > 0 else 0.0
        efficiency = speedup / p
        
    print(f"P={p:<2} | Tiempo Medio: {mean_time:7.4f}s (±{std_time:6.4f}s) | Speedup: {speedup:5.2f}x | Eff: {efficiency:4.2f} | AUC: {best_auc:.6f}")
    
    results.append({
        'threads': p,
        'time': round(mean_time, 2),
        'speedup': round(speedup, 2),
        'efficiency': round(efficiency, 2),
        'auc': round(best_auc, 6)
    })

print(f"\nExportando resultados a {csv_file}...")
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['threads', 'time', 'speedup', 'efficiency', 'auc'])
    writer.writeheader()
    writer.writerows(results)
print("Exportación completada.")
