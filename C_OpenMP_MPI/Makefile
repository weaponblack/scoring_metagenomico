CC       = gcc
CFLAGS   = -O3 -Wall -Wextra -std=c99
OMP_FLAG = -fopenmp
LDFLAGS  = -lm

MSMPI_INC = "C:/Program Files (x86)/Microsoft SDKs/MPI/Include"
MSMPI_LIB = "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64"

all: scoring_openmp scoring_mpi

scoring_openmp: scoring_openmp.c
	$(CC) $(CFLAGS) $(OMP_FLAG) -o $@ $< $(LDFLAGS)

scoring_mpi: scoring_mpi.c
	$(CC) $(CFLAGS) -I$(MSMPI_INC) -o $@ $< -L$(MSMPI_LIB) -lmsmpi $(LDFLAGS)

clean:
	rm -f scoring_openmp.exe scoring_openmp scoring_mpi.exe scoring_mpi
