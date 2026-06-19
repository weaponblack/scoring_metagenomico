#!/bin/bash
# Wrapper script para ejecutar el orquestador global de benchmarks HPC
# Ejecuta las fases secuencial, multiprocessing, OpenMP y MPI

K=50000
N_RUNS=3
DATA_DIR="data"

echo "========================================="
echo "   Iniciando Orquestador Global HPC      "
echo "========================================="
echo "K      = $K"
echo "N_RUNS = $N_RUNS"
echo "DATA   = $DATA_DIR"
echo "========================================="

# Ejecutar el script maestro unificado en Python
python python/run_all.py --K $K --runs $N_RUNS --data_dir $DATA_DIR
