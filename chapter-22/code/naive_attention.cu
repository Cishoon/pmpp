#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cfloat>

// ============================================================
// 朴素 Attention 参考实现
// O = softmax(Q @ K^T / sqrt(d)) @ V
// ============================================================

#define TILE 16

// 步骤 1: S = Q @ K^T / sqrt(d)
__global__ void ScaledDotProductKernel(
    const float* Q, const float* K, float* S,
    int N, int d
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < d; k++) {
            sum += Q[row * d + k] * K[col * d + k];
        }
        S[row * N + col] = sum / sqrtf((float)d);
    }
}

// 步骤 2: P = softmax(S) 按行
__global__ void SoftmaxKernel(const float* S, float* P, int N) {
    int row = blockIdx.x;
    // 求 max
    float m = -FLT_MAX;
    for (int j = 0; j < N; j++) {
        m = fmaxf(m, S[row * N + j]);
    }
    // 求 sum(exp)
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        sum += expf(S[row * N + j] - m);
    }
    // 写 P
    for (int j = 0; j < N; j++) {
        P[row * N + j] = expf(S[row * N + j] - m) / sum;
    }
}

// 步骤 3: O = P @ V
__global__ void PVMultiplyKernel(
    const float* P, const float* V, float* O,
    int N, int d
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < d) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += P[row * N + j] * V[j * d + col];
        }
        O[row * d + col] = sum;
    }
}

void naive_attention(float* Q, float* K, float* V, float* O, int N, int d) {
    float *d_Q, *d_K, *d_V, *d_S, *d_P, *d_O;
    size_t qkv_size = N * d * sizeof(float);
    size_t s_size = N * N * sizeof(float);

    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_S, s_size);
    cudaMalloc(&d_P, s_size);
    cudaMalloc(&d_O, qkv_size);

    cudaMemcpy(d_Q, Q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, qkv_size, cudaMemcpyHostToDevice);

    dim3 block1(TILE, TILE);
    dim3 grid1((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    ScaledDotProductKernel<<<grid1, block1>>>(d_Q, d_K, d_S, N, d);

    SoftmaxKernel<<<N, 1>>>(d_S, d_P, N);

    dim3 block3(TILE, TILE);
    dim3 grid3((d + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    PVMultiplyKernel<<<grid3, block3>>>(d_P, d_V, d_O, N, d);

    cudaMemcpy(O, d_O, qkv_size, cudaMemcpyDeviceToHost);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_S); cudaFree(d_P); cudaFree(d_O);
}

int main() {
    int N = 256, d = 64;
    float *Q = (float*)malloc(N * d * sizeof(float));
    float *K = (float*)malloc(N * d * sizeof(float));
    float *V = (float*)malloc(N * d * sizeof(float));
    float *O = (float*)malloc(N * d * sizeof(float));

    srand(42);
    for (int i = 0; i < N * d; i++) {
        Q[i] = (float)rand() / RAND_MAX - 0.5f;
        K[i] = (float)rand() / RAND_MAX - 0.5f;
        V[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    naive_attention(Q, K, V, O, N, d);

    printf("Naive Attention 完成: N=%d, d=%d\n", N, d);
    printf("O[0][0..3] = %.4f %.4f %.4f %.4f\n", O[0], O[1], O[2], O[3]);

    free(Q); free(K); free(V); free(O);
    return 0;
}
