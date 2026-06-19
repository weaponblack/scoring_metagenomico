#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define MAGIC_PREFIX "\x93NUMPY"

// Parser minimalista para archivos .npy (float32 little-endian)
static void* load_npy_f32(const char* filepath, size_t* out_elements) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "Error opening %s\n", filepath);
        exit(1);
    }
    char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, MAGIC_PREFIX, 6) != 0) {
        fprintf(stderr, "Not a valid NPY file: %s\n", filepath);
        fclose(f);
        exit(1);
    }
    uint8_t major, minor;
    if (fread(&major, 1, 1, f) != 1 || fread(&minor, 1, 1, f) != 1) {
        fprintf(stderr, "Error reading NPY version from %s\n", filepath);
        fclose(f);
        exit(1);
    }
    
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hl;
        if (fread(&hl, 2, 1, f) != 1) {
            fprintf(stderr, "Error reading header length from %s\n", filepath);
            fclose(f);
            exit(1);
        }
        header_len = hl;
    } else {
        if (fread(&header_len, 4, 1, f) != 1) {
            fprintf(stderr, "Error reading header length from %s\n", filepath);
            fclose(f);
            exit(1);
        }
    }
    
    char* header = (char*)malloc(header_len + 1);
    if (fread(header, 1, header_len, f) != header_len) {
        fprintf(stderr, "Error reading header from %s\n", filepath);
        free(header);
        fclose(f);
        exit(1);
    }
    header[header_len] = '\0';
    
    if (!strstr(header, "'<f4'") && !strstr(header, "\"<f4\"")) {
        fprintf(stderr, "Error: %s is not float32 (<f4).\n", filepath);
        free(header);
        fclose(f);
        exit(1);
    }
    free(header);
    
    long data_start = ftell(f);
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, data_start, SEEK_SET);
    
    long data_bytes = file_size - data_start;
    size_t elements = data_bytes / sizeof(float);
    
    float* data = (float*)malloc(data_bytes);
    if (fread(data, sizeof(float), elements, f) != elements) {
        fprintf(stderr, "Error reading data from %s\n", filepath);
        free(data);
        fclose(f);
        exit(1);
    }
    fclose(f);
    
    if (out_elements) *out_elements = elements;
    return data;
}

typedef struct {
    float score;
    float label;
} Pair;

int compare_pairs(const void* a, const void* b) {
    float diff = ((Pair*)b)->score - ((Pair*)a)->score;
    if (diff > 0) return 1;
    if (diff < 0) return -1;
    return 0;
}

double compute_roc_auc(float* scores, const float* labels, size_t n, Pair* pairs) {
    size_t total_positives = 0;
    size_t total_negatives = 0;
    
    for (size_t i = 0; i < n; i++) {
        pairs[i].score = scores[i];
        pairs[i].label = labels[i];
        if (labels[i] > 0.5f) total_positives++;
        else total_negatives++;
    }
    
    if (total_positives == 0 || total_negatives == 0) return 0.0;
    
    qsort(pairs, n, sizeof(Pair), compare_pairs);
    
    double auc = 0.0;
    size_t tp = 0, fp = 0;
    size_t prev_tp = 0, prev_fp = 0;
    float prev_score = pairs[0].score + 1.0f;
    
    for (size_t i = 0; i < n; i++) {
        if (pairs[i].score != prev_score) {
            double trapezoid = ((double)fp - prev_fp) * ((double)tp + prev_tp) / 2.0;
            auc += trapezoid;
            prev_tp = tp;
            prev_fp = fp;
            prev_score = pairs[i].score;
        }
        if (pairs[i].label > 0.5f) tp++;
        else fp++;
    }
    
    double trapezoid = ((double)fp - prev_fp) * ((double)tp + prev_tp) / 2.0;
    auc += trapezoid;
    
    return auc / ((double)total_positives * total_negatives);
}

double rand_uniform() {
    double r = (double)rand() / ((double)RAND_MAX + 1.0);
    if (r == 0.0) r = 1e-10; // Evitar log(0)
    return r;
}

void generate_dirichlet_candidates(float* candidates, size_t K, int seed) {
    srand(seed);
    for (size_t i = 0; i < K; i++) {
        double x1 = -log(1.0 - rand_uniform());
        double x2 = -log(1.0 - rand_uniform());
        double x3 = -log(1.0 - rand_uniform());
        double sum = x1 + x2 + x3;
        candidates[i*3 + 0] = (float)(x1 / sum);
        candidates[i*3 + 1] = (float)(x2 / sum);
        candidates[i*3 + 2] = (float)(x3 / sum);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    if (argc < 3) {
        if (rank == 0) {
            printf("Usage: mpiexec -n <nprocs> %s <data_dir> <K>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    const char* data_dir = argv[1];
    size_t K = (size_t)atol(argv[2]);
    int seed = 42; // Fijo para reproducibilidad entre ejecuciones
    
    char path_A[512], path_y[512], path_T[512], path_S[512], path_F[512];
    snprintf(path_A, sizeof(path_A), "%s/matrix_A.npy", data_dir);
    snprintf(path_y, sizeof(path_y), "%s/labels.npy", data_dir);
    snprintf(path_T, sizeof(path_T), "%s/T.npy", data_dir);
    snprintf(path_S, sizeof(path_S), "%s/S.npy", data_dir);
    snprintf(path_F, sizeof(path_F), "%s/F.npy", data_dir);
    
    size_t n_A, n_y, n_T, n_S, n_F;
    float* A = (float*)load_npy_f32(path_A, &n_A);
    float* y = (float*)load_npy_f32(path_y, &n_y);
    float* T = (float*)load_npy_f32(path_T, &n_T);
    float* S = (float*)load_npy_f32(path_S, &n_S);
    float* F = (float*)load_npy_f32(path_F, &n_F);
    
    size_t n_samples = n_y;
    size_t n_items = n_T;
    
    if (n_A != n_samples * n_items || n_S != n_items || n_F != n_items) {
        if (rank == 0) fprintf(stderr, "Dimension mismatch in loaded files.\n");
        MPI_Finalize();
        return 1;
    }
    
    // Todos los procesos generan todos los candidatos para mantener determinismo
    float* candidates = (float*)malloc(K * 3 * sizeof(float));
    generate_dirichlet_candidates(candidates, K, seed);
    
    // Distribuir el trabajo
    size_t chunk_size = K / nprocs;
    size_t start_k = rank * chunk_size;
    size_t end_k = (rank == nprocs - 1) ? K : start_k + chunk_size;
    
    if (rank == 0) {
        printf("=== Metagenomic Scoring - MPI ===\n");
        printf("Dataset: n_samples=%zu, n_items=%zu\n", n_samples, n_items);
        printf("Candidates K=%zu, Processes=%d\n", K, nprocs);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();
    
    float* local_P = (float*)malloc(n_items * sizeof(float));
    float* local_scores = (float*)malloc(n_samples * sizeof(float));
    Pair* local_pairs = (Pair*)malloc(n_samples * sizeof(Pair));
    
    double local_best_auc = -1.0;
    float local_best_w[3] = {0};
    size_t local_best_k = K + 1;
    
    for (size_t k = start_k; k < end_k; k++) {
        const float* w = &candidates[k * 3];
        
        for (size_t j = 0; j < n_items; j++) {
            local_P[j] = w[0]*T[j] + w[1]*S[j] + w[2]*F[j];
        }
        
        for (size_t i = 0; i < n_samples; i++) {
            float score = 0.0f;
            const float* row_A = &A[i * n_items];
            for (size_t j = 0; j < n_items; j++) {
                score += row_A[j] * local_P[j];
            }
            local_scores[i] = score;
        }
        
        double auc = compute_roc_auc(local_scores, y, n_samples, local_pairs);
        
        if (auc > local_best_auc || (auc == local_best_auc && k < local_best_k)) {
            local_best_auc = auc;
            local_best_w[0] = w[0];
            local_best_w[1] = w[1];
            local_best_w[2] = w[2];
            local_best_k = k;
        }
    }
    
    free(local_P);
    free(local_scores);
    free(local_pairs);
    
    // Sincronizar para terminar el timer global
    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double search_time = t_end - t_start;
    
    // Preparar buffer para la reducción: [auc, w1, w2, w3, double(local_best_k)]
    // Usamos double para todo en el Gather para facilitar
    double local_res[5] = { local_best_auc, (double)local_best_w[0], (double)local_best_w[1], (double)local_best_w[2], (double)local_best_k };
    double* all_res = NULL;
    
    if (rank == 0) {
        all_res = (double*)malloc(nprocs * 5 * sizeof(double));
    }
    
    MPI_Gather(local_res, 5, MPI_DOUBLE, all_res, 5, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        double global_best_auc = -1.0;
        float global_best_w[3] = {0};
        size_t global_best_k = K + 1;
        
        for (int p = 0; p < nprocs; p++) {
            double p_auc = all_res[p * 5 + 0];
            float p_w1 = (float)all_res[p * 5 + 1];
            float p_w2 = (float)all_res[p * 5 + 2];
            float p_w3 = (float)all_res[p * 5 + 3];
            size_t p_k = (size_t)all_res[p * 5 + 4];
            
            if (p_auc > global_best_auc || (p_auc == global_best_auc && p_k < global_best_k)) {
                global_best_auc = p_auc;
                global_best_w[0] = p_w1;
                global_best_w[1] = p_w2;
                global_best_w[2] = p_w3;
                global_best_k = p_k;
            }
        }
        
        printf("Search time: %.4f s\n", search_time);
        printf("Best AUC:    %.6f\n", global_best_auc);
        printf("Best W:      [%f, %f, %f]\n", global_best_w[0], global_best_w[1], global_best_w[2]);
        printf("--- Benchmark ---\n");
        printf("K=%zu, P=%d, Time=%.4f, AUC=%.6f, W1=%f, W2=%f, W3=%f\n", 
               K, nprocs, search_time, global_best_auc, 
               global_best_w[0], global_best_w[1], global_best_w[2]);
               
        free(all_res);
    }
           
    free(A); free(y); free(T); free(S); free(F);
    free(candidates);
    
    MPI_Finalize();
    return 0;
}
