# Optimización Paralela de Scoring Metagenómico para Clasificación Binaria

Este proyecto de Computación de Alto Rendimiento (HPC) implementa un framework modular para la optimización de un vector de pesos metagenómico utilizado en clasificación binaria de muestras de salud (Sanas vs. Enfermas). 

Actualmente el proyecto abarca desde el **Nivel 1** (implementación secuencial en Python) hasta el **Nivel 4** (sistemas distribuidos con MS-MPI), con aceleraciones drásticas en sistemas multinúcleo gracias a C y OpenMP (**Nivel 3**). Todo el pipeline está optimizado con tipado `float32`.

---

## Modelo Matemático y Computacional

El objetivo científico es proyectar perfiles metagenómicos de alta dimensionalidad sobre una firma compacta y ponderada, para luego clasificar a los pacientes mediante su puntuación proyectada.

### 1. Perfiles y Ponderación Metagenómica
Cada metagenoma se representa mediante $N$ ítems o características (taxones, ecologías o funciones). Para cada ítem $i \in \{1, \dots, N\}$, disponemos de tres perfiles de referencia:
*   $T_i \ge 0$: Perfil taxonómico.
*   $S_i \ge 0$: Perfil ecológico.
*   $F_i \ge 0$: Perfil funcional.

Para un vector de pesos candidato $W = (W_1, W_2, W_3)$, el perfil unificado del metagenoma $P \in \mathbb{R}^N$ se define como una combinación convexa de precisión simple (`float32`):
$$P_i = W_1 T_i + W_2 S_i + W_3 F_i \quad (\forall i \in \{1, \dots, N\})$$

Sujeto a las restricciones físicas de abundancia relativa:
$$W_1 + W_2 + W_3 = 1, \quad W_j \ge 0$$

### 2. Puntuación por Muestra (Scoring)
Sea $A \in \mathbb{R}^{M \times N}$ la matriz de abundancias observadas, donde $M$ es el número total de muestras ($2 \times \text{n\_samples\_per\_class}$). La puntuación metagenómica para cada muestra se obtiene mediante el producto matriz-vector:
$$\text{Score} = A P$$
Este cálculo resulta en un vector de puntuación de forma $(M,)$ y tipo `float32`.

### 3. Función Objetivo (Maximizacion del AUC ROC)
Dado un vector de etiquetas binarias reales $y \in \mathbb{R}^M$ (donde $0.0$ indica sano y $1.0$ enfermo):
$$y = [\underbrace{0.0, \dots, 0.0}_{M/2}, \underbrace{1.0, \dots, 1.0}_{M/2}]$$
El sistema busca hallar el vector de pesos óptimo $W^*$ que maximice el área bajo la curva ROC (AUC ROC):
$$W^* = \arg\max_{W} \text{AUC}(y, A (W_1 T + W_2 S + W_3 F))$$

---

## Generación de Datos Sintéticos Controlados

Para garantizar que el espacio de optimización sea reproducible, desafiante y estadísticamente coherente, el generador implementa el siguiente enfoque:

1.  **Perfiles Base**: Genera los perfiles $T, S, F$ de manera aleatoria positiva (`float32`).
2.  **Señal Oculta**: Define internamente un vector óptimo "objetivo" real $W_{\text{target}} = [0.6, 0.3, 0.1]$ y calcula $P_{\text{target}} = 0.6 T + 0.3 S + 0.1 F$.
3.  **Matriz A de Muestras**:
    *   **Muestras Sanas ($y_k = 0$)**: Se simulan como ruido uniforme $\text{Uniform}(0, 1)$.
    *   **Muestras Enfermas ($y_k = 1$)**: Se simulan con la misma base uniforme pero agregando una perturbación proporcional a la señal objetivo:
        $$A_{k, j} \sim \text{Uniform}(0, 1) + \beta \cdot P_{\text{target}, j}$$
        Donde $\beta$ es el parámetro `signal_strength` (fuerza de la señal).
4.  **Consecuencia**: Un vector de pesos $W$ cercano a $W_{\text{target}}$ maximizará la diferencia de puntuaciones entre sanos y enfermos, logrando un AUC ROC cercano a $1.0$. Rumbos alejados degradarán el AUC significativamente. Esto evita paisajes completamente planos o triviales.

---

## Estructura del Proyecto

El código está organizado de manera estrictamente modular conforme a las especificaciones:

```text
scoring_metagenomico/
│
├── C_OpenMP_MPI/
│   ├── scoring_openmp.c       # Motor paralelo en C (Memoria Compartida - OpenMP)
│   ├── scoring_mpi.c          # Motor distribuido en C (MS-MPI)
│   ├── Makefile               # Script de compilación (GCC)
│   ├── run_sweep.py           # Benchmark automatizado para OpenMP
│   └── run_sweep_mpi.py       # Benchmark automatizado para MPI
│
├── data/
│   ├── generate_data.py       # Script CLI generador de datos sintéticos
│   ├── matrix_A.npy           # Matriz A guardada (NumPy float32)
│   ├── labels.npy             # Vector de etiquetas y (NumPy float32)
│   ├── T.npy, S.npy, F.npy    # Perfiles metagenómicos (NumPy float32)
│
├── python/
│   ├── sequential.py          # Baseline secuencial (búsqueda aleatoria)
│   ├── multicore.py           # Nivel 2: Multiprocesamiento nativo en Python
│   └── utils.py               # Funciones auxiliares de carga y evaluación
│
├── results/
│   ├── benchmark.csv          # Registro persistente de métricas (Python)
│   ├── openmp_benchmark.csv   # Registro persistente de métricas (OpenMP)
│   ├── mpi_benchmark.csv      # Registro persistente de métricas (MPI)
│   └── plots/                 # Directorio para visualizaciones
│
├── MANUAL_USUARIO.md          # Guía detallada de compilación y ejecución de todos los sistemas
├── requirements.txt           # Dependencias de Python
└── README.md                  # Documentación del proyecto (este archivo)
```

---

## Fases Implementadas

### Nivel 1: Baseline Secuencial en Python
Establece la precisión matemática y las estructuras de datos (NumPy `.npy`), calculando el AUC de forma determinista para probar correctitud y estableciendo los tiempos base (baseline).

### Nivel 2: Multiprocesamiento en Python (Multicore)
La versión paralela distribuye los $K$ candidatos de pesos entre múltiples procesos locales utilizando un `multiprocessing.Pool` (con método `spawn`). Destaca por el patrón de inicialización del IPC (`_worker_init`) para compartir la matriz de datos en memoria en los procesos hijo de forma global y eliminar el enorme *overhead* de serialización por mensajes.

### Nivel 3: Paralelismo en Memoria Compartida (C + OpenMP)
La operación matemática se porta a **C nativo**. Cuenta con un parser propio y ligero de archivos `.npy` y un algoritmo de cálculo de ROC AUC (`O(n log n)`) optimizado sin dependencias externas. 
Usando un bucle paralelo `#pragma omp parallel for` de grano grueso (Coarse-Grained) sobre el bucle externo de los $K$ candidatos, se elimina toda sobrecarga de sincronización de hilos, permitiendo acelerar las evaluaciones de manera lineal aprovechando la caché del procesador. Se garantiza 100% de determinismo matemático en los resultados.

### Nivel 4: Sistemas Distribuidos (C + MS-MPI)
Implementación orientada a granjas de servidores o clústeres. Los procesos rank MPI cargan independientemente el dataset desde el sistema de archivos local para no saturar la red (sin requerir gigantescos `MPI_Bcast` para matrices de memoria), y se dividen el espacio de búsqueda Dirichlet. Utilizando la primitiva `MPI_Gather` hacia el Rank 0, se aplica un desempate determinista riguroso de los mejores AUC locales obtenidos en paralelo, logrando resultados idénticos a los del modelo base secuencial pero distribuibles en múltiples nodos.

---

## Instrucciones de Instalación, Compilación y Ejecución

Debido a que el proyecto ha evolucionado para cubrir múltiples lenguajes y arquitecturas (Python, OpenMP, y MS-MPI en Windows), hemos consolidado y movido todos los pasos exactos a un manual independiente para mantener la legibilidad:

👉 **[Consulta el MANUAL_USUARIO.md para instrucciones completas de uso, compilación, ejecución y automatización de benchmarks.](MANUAL_USUARIO.md)**

---

## Registro de Benchmarking y Visualizaciones

Tanto la versión de Python (`multicore.py --sweep`) como las versiones en C (`run_sweep.py` y `run_sweep_mpi.py`) automatizan barridos sobre múltiples hilos ($P=1,2,4,8$) generando registros estadísticos detallados y automatizados en la carpeta `results/`:

*   **Tiempos y Eficiencia**: Tiempo medio (`mean_execution_time`), desviación estándar, factor de aceleración (`speedup`) y eficiencia paralela (`efficiency`) en relación al baseline.
*   **Optimalidad**: Mejor AUC descubierto (`best_auc`) y los pesos W hallados (`best_w1`, etc).
*   **Análisis Visual** *(Solo script Multicore de Python)*: Distribución de scores (`score_distribution.png`), Curva ROC (`roc_curve.png`), métricas de aceleración teórica vs real (`speedup_vs_processes.png`) y análisis automatizado de la Ley de Amdahl.

---

## Roadmap HPC (Futuro)

Con las capas CPU Multicore y Multinodo completadas y validadas, el framework queda arquitectónicamente listo para la fase final:

1.  **CUDA (Nivel 5 - Aceleración Masiva por GPU)**:
    *   El uso estricto y universal en el código de datos de precisión simple (`float32`) prepara el terreno de manera ideal para el hardware de GPU moderno. El producto matriz-vector $A P$ y las sumas vectoriales pueden ejecutarse concurrentemente en miles de hilos de GPU de manera matricial usando librerías como CuPy, PyCUDA o implementando kernels directos en C++ y CUDA (cublas) para lograr aceleraciones monstruosas (potencialmente superando factores de 100x respecto a la CPU secuencial en datasets gigantescos de cientos de miles de ítems).
