# Optimización Paralela de Scoring Metagenómico para Clasificación Binaria

Este proyecto de Computación de Alto Rendimiento (HPC) implementa un framework modular para la optimización de un vector de pesos metagenómico utilizado en clasificación binaria de muestras de salud (Sanas vs. Enfermas). 

Esta versión corresponde al **Nivel 1**: la implementación secuencial en Python (baseline) sin paralelismo. Todo el pipeline está optimizado con tipado `float32` y diseñado de manera desacoplada para facilitar la posterior paralelización mediante Multiprocessing, OpenMP, MPI y CUDA (GPU).

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

```
scoring_metagenomico/
│
├── data/
│   ├── generate_data.py       # Script CLI generador de datos sintéticos
│   ├── matrix_A.npy           # Matriz A guardada (NumPy float32)
│   ├── labels.npy             # Vector de etiquetas y (NumPy float32)
│   ├── T.npy                  # Perfil taxonómico T (NumPy float32)
│   ├── S.npy                  # Perfil ecológico S (NumPy float32)
│   └── F.npy                  # Perfil funcional F (NumPy float32)
│
├── python/
│   ├── sequential.py          # Baseline secuencial (búsqueda aleatoria + benchmark)
│   └── utils.py               # Funciones auxiliares de carga, evaluación y gráficos
│
├── results/
│   ├── benchmark.csv          # Registro persistente de tiempos y métricas
│   └── plots/                 # Directorio para curvas ROC e histogramas
│
├── requirements.txt           # Dependencias de Python
└── README.md                  # Documentación del proyecto (este archivo)
```

---

## Instrucciones de Instalación y Ejecución

### 1. Prerrequisitos e Instalación
Asegúrate de tener instalado Python (versión 3.9 o superior). Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

*(Las dependencias clave son `numpy`, `scikit-learn`, `matplotlib`, `pandas`, `tqdm`)*

### 2. Paso 1: Generar el Dataset Sintético
Usa el script `data/generate_data.py` para construir matrices escalables. 

**Parámetros configurables:**
*   `--n_samples_per_class`: Cantidad de muestras sanas/enfermas (ej. 50 genera 100 muestras en total).
*   `--n_items`: Dimensión de características $N$ (ej. 10000).
*   `--signal_strength`: Multiplicador $\beta$ (fuerza de la firma de enfermedad).
*   `--seed`: Semilla aleatoria de reproducibilidad.

**Ejemplo de ejecución (Set pequeño de prueba):**
```bash
python data/generate_data.py --n_samples_per_class 5 --n_items 10000 --signal_strength 0.5 --seed 42
```

**Ejemplo de ejecución (Set grande de escalabilidad):**
```bash
python data/generate_data.py --n_samples_per_class 50 --n_items 50000 --signal_strength 0.4 --seed 42
```

### 3. Paso 2: Ejecutar la Búsqueda Aleatoria y Benchmark Secuencial
El script `python/sequential.py` genera $K$ candidatos de peso usando la distribución de Dirichlet, ejecuta la optimización secuencial de forma repetida para evitar ruido estadístico, y genera los gráficos.

**Parámetros configurables:**
*   `-K`: Cantidad de vectores candidatos de pesos a evaluar.
*   `--n_runs`: Número de corridas de búsqueda consecutivas a promediar en el benchmark.
*   `--search_seed`: Semilla aleatoria para la generación de candidatos Dirichlet.
*   `--signal_strength` y `--data_seed`: Información de origen de datos para fines de auditoría del log.

**Ejemplo de ejecución:**
```bash
python python/sequential.py -K 5000 --n_runs 5 --search_seed 42
```

---

## Registro de Benchmarking y Visualizaciones

### Registro en `results/benchmark.csv`
Cada ejecución exitosa del pipeline de benchmark añade una fila al archivo CSV con las siguientes variables estadísticas:
*   `implementation`: Nombre del runtime (siempre `"sequential"` en Nivel 1).
*   `K`: Número de candidatos evaluados.
*   `n_samples`: Cantidad de muestras procesadas.
*   `n_items`: Dimensión de ítems metagenómicos.
*   `n_runs`: Cantidad de corridas de timing.
*   `mean_execution_time`: Tiempo medio neto del bucle de optimización (segundos).
*   `std_execution_time`: Desviación estándar del tiempo de búsqueda (segundos).
*   `best_auc`: Mayor AUC ROC alcanzado.
*   `best_w1`, `best_w2`, `best_w3`: Pesos óptimos descubiertos.
*   `signal_strength` / `seed`: Configuración experimental.
*   `dtype`: Precisión de memoria utilizada (`"float32"`).

### Visualizaciones en `results/plots/`
El ejecutor genera dos archivos gráficos:
1.  **`score_distribution.png`**: Histograma de las puntuaciones de muestras sanas (azul) vs. enfermas (naranja), mostrando el grado de separación logrado por el modelo optimizado.
2.  **`roc_curve.png`**: Gráfico estándar ROC para el mejor candidato de pesos, mostrando el True Positive Rate vs. False Positive Rate.

---

## Preparación para la Escalabilidad Futura (Roadmap HPC)

El código ha sido diseñado cuidadosamente para allanar el camino de las siguientes etapas de optimización:

1.  **Multiprocessing (Nivel 2 - Python)**: 
    *   La función `evaluate_candidate` en `utils.py` está totalmente aislada de variables globales y estados mutables de clase.
    *   La paralelización puede realizarse fácilmente mapeando un pool de procesos (`multiprocessing.Pool`) sobre los $K$ candidatos, distribuyendo la carga computacional en múltiples núcleos de CPU.
2.  **C/C++ con OpenMP (Nivel 3)**:
    *   La operación central es el cálculo del vector $P$ y el producto matriz-vector $A \cdot P$.
    *   En C/C++, la matriz $A$ y los vectores se almacenarán de forma contigua en memoria. Un bucle paralelo `#pragma omp parallel for` acelerará drásticamente el producto matriz-vector para conjuntos masivos.
3.  **MPI (Nivel 4 - Sistemas Distribuidos)**:
    *   Para procesar millones de candidatos $K$ o dimensiones de ítems que no quepan en un solo nodo, MPI permitirá dividir el rango de candidatos o la matriz de datos entre múltiples nodos de cómputo usando llamadas `MPI_Scatter` / `MPI_Gather`.
4.  **CUDA (Nivel 5 - Aceleración por GPU)**:
    *   El uso de datos en precisión simple (`float32`) es el estándar óptimo para hardware GPU.
    *   El producto matriz-vector $A P$ y las sumas vectoriales pueden ejecutarse concurrentemente en miles de hilos de GPU usando librerías como PyCUDA, CuPy o implementando kernels CUDA directos para lograr aceleraciones masivas (superando los 100x en datasets gigantescos).
