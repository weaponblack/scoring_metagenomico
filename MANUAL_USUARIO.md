# Manual de Usuario - Scoring Metagenómico

Este manual detalla las instrucciones completas para compilar, ejecutar y hacer benchmark de todas las fases del proyecto de Búsqueda de Pesos para Scoring Metagenómico. El ecosistema consta de una versión inicial en Python (Multiprocessing) y versiones de alto rendimiento en C (OpenMP y MS-MPI).

---

## 1. Requisitos Previos (Entorno Windows)

Para ejecutar correctamente todos los componentes, asegúrate de tener instalados:

*   **Python 3.8+** con `numpy` instalado.
*   **MSYS2 (MinGW-w64)**: Que incluye el compilador `gcc` nativo para Windows con soporte para OpenMP.
*   **MS-MPI (Microsoft MPI)**:
    *   *MS-MPI Runtime* (provee el comando `mpiexec`).
    *   *MS-MPI SDK* (provee los headers y librerías estáticas para enlazar programas C). Se instala por defecto en `C:\Program Files (x86)\Microsoft SDKs\MPI\`.

---

## 2. Generación del Dataset Sintético

Antes de correr cualquier búsqueda, se debe generar la matriz de datos y sus perfiles (Taxonómico `T`, Ecológico `S`, Funcional `F`). Los archivos se guardarán directamente en formato binario `numpy` ultrarrápido (`.npy`).

1. Abre tu terminal (PowerShell o CMD).
2. Entra al directorio `data/`:
   ```powershell
   cd data
   ```
3. Ejecuta el generador indicando los parámetros. Ejemplo para generar 10 muestras (5 de cada clase) y 10000 ítems:
   ```powershell
   python generate_data.py --n_samples_per_class 5 --n_items 10000
   ```
   > Los datos se guardarán automáticamente en esa misma carpeta `data/`.

---

## 3. Fase Python (Baseline y Multiprocessing)

La implementación en Python usa librerías nativas (`multiprocessing`) para demostrar el caso base e incluye la generación de visualizaciones (plots) automáticos en la carpeta `results/`.

1. Ve a la carpeta de Python:
   ```powershell
   cd python
   ```
2. Ejecuta el archivo principal usando parámetros como `K` (candidatos) y `P` (procesos). Por defecto toma el máximo de núcleos de la CPU:
   ```powershell
   python multicore.py --data_dir ../data -K 50000 -P 4
   ```
3. Alternativamente, puedes lanzar el comando **Sweep** que ejecutará y graficará automáticamente el escalamiento de todos tus núcleos:
   ```powershell
   python multicore.py --sweep --data_dir ../data -K 50000
   ```
   > Revisa la carpeta `../results/` para ver las gráficas generadas y el CSV del baseline.

---

## 4. Fase C de Alto Rendimiento (OpenMP y MS-MPI)

La fase 3 provee motores nativos en C que parsean directamente los archivos `.npy` de Python sin conversiones, computando el ROC AUC a extrema velocidad. Se proveen versiones para memoria compartida (OpenMP) y memoria distribuida (MPI).

### 4.1 Compilación de los Ejecutables en C

1. Navega a la carpeta de C:
   ```powershell
   cd C_OpenMP_MPI
   ```
2. Ejecuta `make` (si tienes MinGW make en tu PATH) o directamente `gcc`. Nuestro Makefile está preparado para MINGW:
   ```powershell
   make
   ```
   *(Nota: El Makefile está pre-configurado para enlazar MS-MPI asumiendo la instalación estándar en `C:/Program Files (x86)/Microsoft SDKs/MPI/`).*

### 4.2 Ejecución Directa de los Motores

Una vez compilados, aparecerán `scoring_openmp.exe` y `scoring_mpi.exe`. Ambos toman exactamente los mismos argumentos básicos: `[directorio_datos] [K_candidatos]`.

**Para OpenMP** (Requiere especificar número de hilos como 3er argumento):
```powershell
# Ejecutar K=50000 usando 8 hilos OpenMP
./scoring_openmp.exe ../data 50000 8
```

**Para MS-MPI** (El runtime de Windows `mpiexec` inyecta la cantidad de procesos):
```powershell
# Ejecutar K=50000 usando 8 procesos MPI
mpiexec -n 8 ./scoring_mpi.exe ../data 50000
```

> **Determinismo Garantizado**: Para cualquier `P` dado, siempre obtendrás el mismo score AUC y exactamente la misma combinación de pesos debido a los empates estrictos programados (gana siempre el candidato generado con menor índice).

### 4.3 Ejecución de Benchmarks (Wrappers Automáticos)

Para evaluar el escalamiento (Speedup y Eficiencia) de las versiones en C de forma cómoda, hemos provisto dos wrappers en Python que iteran automáticamente repitiendo la prueba varias veces (`n_runs=3`) en configuraciones de `P = 1, 2, 4, 8`.

1. Desde el directorio `C_OpenMP_MPI`:
   ```powershell
   # Para automatizar OpenMP
   python run_sweep.py
   
   # Para automatizar MS-MPI
   python run_sweep_mpi.py
   ```
2. **Exportación de Resultados**: Al finalizar, ambos scripts crearán un informe detallado en formato CSV directamente en la carpeta de resultados consolidada del proyecto:
   - `../results/openmp_benchmark.csv`
   - `../results/mpi_benchmark.csv`

---
*Con este manual, estás listo para generar datos, compilar, correr, paralelizar en memoria compartida y distribuida, y visualizar de forma automática el desempeño y escalamiento de tu herramienta genómica.*
