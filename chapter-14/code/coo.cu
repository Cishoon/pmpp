#include <stdio.h>
#include <cuda_runtime.h>

__global__ void spmv_coo_kernel(int nnz, int *rowIdx, int *colIdx, float *values, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        int row = rowIdx[i];
        int col = colIdx[i];
        float value = values[i];
        atomicAdd(&y[row], value * x[col]);
    }
}

void spmv_coo(int nnz, int rows, int cols, int *h_rowIdx, int *h_colIdx, float *h_values, float *h_x) {
    int *d_rowIdx, *d_colIdx;
    float *d_values, *d_x, *d_y;

    cudaMalloc((void**)&d_rowIdx, nnz * sizeof(int));
    cudaMalloc((void**)&d_colIdx, nnz * sizeof(int));
    cudaMalloc((void**)&d_values, nnz * sizeof(float));
    cudaMalloc((void**)&d_x, cols * sizeof(float));
    cudaMalloc((void**)&d_y, rows * sizeof(float));

    cudaMemcpy(d_rowIdx, h_rowIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, rows * sizeof(float));  // init y as 0

    int blockSize = 256;
    int gridSize = (nnz + blockSize - 1) / blockSize;
    spmv_coo_kernel<<<gridSize, blockSize>>>(nnz, d_rowIdx, d_colIdx, d_values, d_x, d_y);

    float *h_y = new float[rows];
    cudaMemcpy(h_y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result vector y:\n");
    for (int i = 0; i < rows; i++) {
        printf("%f ", h_y[i]);
    }
    printf("\n");

    delete[] h_y;
    cudaFree(d_rowIdx);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    int h_rowIdx[] = {0, 0, 1, 2};
    int h_colIdx[] = {0, 1, 2, 2};
    float h_values[] = {10.0, 20.0, 30.0, 40.0};
    float h_x[] = {1.0, 2.0, 3.0};  // Input vector

    int nnz = 4; // Number of non-zero elements
    int rows = 3, cols = 3;

    spmv_coo(nnz, rows, cols, h_rowIdx, h_colIdx, h_values, h_x);

    return 0;
}
