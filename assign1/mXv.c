//Sequential Matrix-Vector Multiplication (matrix_vector_seq.c)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void matrix_vector_multiply(double *matrix, double *vector, double *result, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < cols; ++j) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <matrix_size> <vector_size>\n", argv[0]);
        return 1;
    }
    int matrix_size = atoi(argv[1]);
    int vector_size = atoi(argv[2]);
    double *matrix = (double *)malloc(matrix_size * matrix_size * sizeof(double));
    double *vector = (double *)malloc(vector_size * sizeof(double));
    double *result = (double *)malloc(matrix_size * sizeof(double));
    srand(time(NULL)); 
    for (int i = 0; i < matrix_size * matrix_size; ++i) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < vector_size; ++i) {
        vector[i] = (double)rand() / RAND_MAX;
    }
    matrix_vector_multiply(matrix, vector, result, matrix_size, matrix_size);
    printf("Result:\n");
    for (int i = 0; i < matrix_size; ++i) {
        printf("%lf ", result[i]);
    }
    printf("\n");
    free(matrix);
    free(vector);
    free(result);
    return 0;
}
 

//OpenMP Naive Matrix-Vector Multiplication (matrix_vector_openmp_naive.c)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
void matrix_vector_multiply(double *matrix, double *vector, double *result, int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < cols; ++j) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <matrix_size> <vector_size>\n", argv[0]);
        return 1;
    }
    int matrix_size = atoi(argv[1]);
    int vector_size = atoi(argv[2]);
    double *matrix = (double *)malloc(matrix_size * matrix_size * sizeof(double));
    double *vector = (double *)malloc(vector_size * sizeof(double));
    double *result = (double *)malloc(matrix_size * sizeof(double));
    srand(time(NULL));

    for (int i = 0; i < matrix_size * matrix_size; ++i) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < vector_size; ++i) {
        vector[i] = (double)rand() / RAND_MAX;
}
matrix_vector_multiply(matrix, vector, result, matrix_size, matrix_size);
printf("Result:\n");
for (int i = 0; i < matrix_size; ++i) {
    printf("%lf ", result[i]);
}
printf("\n");
free(matrix);
free(vector);
free(result);
return 0;
}
 

//MPI Naive Matrix-Vector Multiplication (matrix_vector_mpi_naive.c)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
void matrix_vector_multiply(double *matrix, double *vector, double *result, int rows, int cols, int my_rank, int num_procs) {
    int chunk_size = rows / num_procs;
    int start_row = my_rank * chunk_size;
    int end_row = (my_rank == num_procs - 1) ? rows : start_row + chunk_size;
    for (int i = start_row; i < end_row; ++i) {
        result[i] = 0.0;
        for (int j = 0; j < cols; ++j) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
    if (my_rank != 0) {
        MPI_Send(result + start_row, end_row - start_row, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        for (int proc = 1; proc < num_procs; ++proc) {
            MPI_Recv(result + proc * chunk_size, chunk_size, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}
int main(int argc, char *argv[]) {
    int my_rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (argc != 3) {
        if (my_rank == 0) {
            printf("Usage: %s <matrix_size> <vector_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    int matrix_size = atoi(argv[1]);
    int vector_size = atoi(argv[2]);
    double *matrix = (double *)malloc(matrix_size * matrix_size * sizeof(double));
    double *vector = (double *)malloc(vector_size * sizeof(double));
    double *result = (double *)malloc(matrix_size * sizeof(double));
    srand(time(NULL));
    if (my_rank == 0) {
        for (int i = 0; i < matrix_size * matrix_size; ++i) {
            matrix[i] = (double)rand() / RAND_MAX;
        }
        for (int i = 0; i < vector_size; ++i) {
            vector[i] = (double)rand() / RAND_MAX;
        }
    }
    MPI_Bcast(matrix, matrix_size * matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, vector_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    matrix_vector_multiply(matrix, vector, result, matrix_size, matrix_size, my_rank, num_procs);
    if (my_rank == 0) {
        printf("Result:\n");
        for (int i = 0; i < matrix_size; ++i) {
            printf("%lf ", result[i]);
        }
        printf("\n");
    }
    free(matrix);
    free(vector);
    free(result);
    MPI_Finalize();
    return 0;
}
 

//OpenMP Matrix-Vector Multiplication with Tiling (matrix_vector_openmp_tiling.c)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
void matrix_vector_multiply(double *matrix, double *vector, double *result, int rows, int cols) {
    const int tile_size = 16; // Define tile size
    #pragma omp parallel for
    for (int i = 0; i < rows; i += tile_size) {
        for (int j = 0; j < cols; ++j) {
            for (int ii = i; ii < i + tile_size && ii < rows; ++ii) {
                result[ii] += matrix[ii * cols + j] * vector[j];
            }
        }
    }
}
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <matrix_size> <vector_size>\n", argv[0]);
        return 1;
    }
    int matrix_size = atoi(argv[1]);
    int vector_size = atoi(argv[2]);
    double *matrix = (double *)malloc(matrix_size * matrix_size * sizeof(double));
    double *vector = (double *)malloc(vector_size * sizeof(double));
    double *result = (double *)calloc(matrix_size, sizeof(double)); // Initialize result to zero
    srand(time(NULL));
    for (int i = 0; i < matrix_size * matrix_size; ++i) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
    for (int i = 0; i < vector_size; ++i) {
        vector[i] = (double)rand() / RAND_MAX;
    }

    matrix_vector_multiply(matrix, vector, result, matrix_size, matrix_size);
    printf("Result:\n");
    for (int i = 0; i < matrix_size; ++i) {
        printf("%lf ", result[i]);
    }
    printf("\n");
    free(matrix);
    free(vector);
    free(result);
    return 0;
}
 

//MPI Matrix-Vector Multiplication with Tiling (matrix_vector_mpi_tiling.c)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
void matrix_vector_multiply(double *matrix, double *vector, double *result, int rows, int cols, int my_rank, int num_procs) {
    const int tile_size = 16; // Define tile size
    int chunk_size = rows / num_procs;
    int start_row = my_rank * chunk_size;
    int end_row = (my_rank == num_procs - 1) ? rows : start_row + chunk_size;

    for (int i = start_row; i < end_row; i += tile_size) {
        for (int j = 0; j < cols; ++j) {
            for (int ii = i; ii < i + tile_size && ii < end_row; ++ii) {
                result[ii] += matrix[ii * cols + j] * vector[j];
            }
        }
    }
    if (my_rank != 0) {
        MPI_Send(result + start_row, end_row - start_row, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        for (int proc = 1; proc < num_procs; ++proc) {
            MPI_Recv(result + proc * chunk_size, chunk_size, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}
int main(int argc, char *argv[]) {
    int my_rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (argc != 3) {
        if (my_rank == 0) {
            printf("Usage: %s <matrix_size> <vector_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    int matrix_size = atoi(argv[1]);
    int vector_size = atoi(argv[2]);
    double *matrix = (double *)malloc(matrix_size * matrix_size * sizeof(double));
    double *vector = (double *)malloc(vector_size * sizeof(double));
    double *result = (double *)calloc(matrix_size, sizeof(double)); // Initialize result to zero
    srand(time(NULL));
    if (my_rank == 0) {
        for (int i = 0; i < matrix_size * matrix_size; ++i) {
            matrix[i] = (double)rand() / RAND_MAX;
        }
        for (int i = 0; i < vector_size; ++i) {
            vector[i] = (double)rand() / RAND_MAX;
        }
    }
    MPI_Bcast(matrix, matrix_size * matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, vector_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    matrix_vector_multiply(matrix, vector, result, matrix_size, matrix_size, my_rank, num_procs);
    if (my_rank == 0) {
        printf("Result:\n");
        for (int i = 0; i < matrix_size; ++i) {
            printf("%lf ", result[i]);
        }
        printf("\n");
    }
    free(matrix);
    free(vector);
    free(result);
    MPI_Finalize();
    return 0;
}
 
//Commands
//To accomplish steps 8-10
#!/bin/bash
# Define the programs to run
PROGRAMS=("matrix_vector_seq" "matrix_vector_openmp_naive" "matrix_vector_mpi_naive" "matrix_vector_openmp_tiling" "matrix_vector_mpi_tiling")
# Define the input sizes
INPUT_SIZES=(64 128 256 512 1024 2048 4096 8192 16384 32768)
# Define the number of repetitions for averaging
REPEATS=10
# Output file name
OUTPUT_FILE="execution_times.csv"
# Remove existing output file
rm -f $OUTPUT_FILE
# Print CSV header
echo "test S.no, file, input size, time taken" >> $OUTPUT_FILE
# Loop over each program
for program in "${PROGRAMS[]}"; do
    # Compile the program
    make $program
    # Loop over each input size
    for size in "${INPUT_SIZES[@]}"; do
        total_time=0
        # Run the program multiple times to get average time
        for (( i=1; i<=$REPEATS; i++ )); do
            # Execute the program and capture execution time
            execution_time=$(./$program $size $size | tail -1)
            total_time=$(echo "$total_time + $execution_time" | bc)
        done
        # Calculate average time
        average_time=$(echo "scale=6; $total_time / $REPEATS" | bc)
        # Print to CSV file
        echo "$program, $size, $average_time secs" >> $OUTPUT_FILE
    done
done
# Inform user about completion
echo "Execution times recorded in $OUTPUT_FILE"
