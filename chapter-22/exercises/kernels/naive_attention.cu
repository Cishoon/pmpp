#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>

#define TILE 16
#define BLOCK_SIZE 256
#define cdiv(a, b) (((a) + (b) - 1) / (b))

__global__ void ScaledDotProductKernel(const float* d_Q, const float* d_K, float* d_S, int N, int d) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < d; i++) {
            // Q[row, i] * K[col, i]  (K[col] 即 K^T 的第 col 列)
            sum += d_Q[row * d + i] * d_K[col * d + i];
        }
        d_S[row * N + col] = sum / sqrtf((float)d);
    }
}

__global__ void SoftmaxKernel(const float* d_S, float* d_P, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N) {
        // 求max
        float m = -FLT_MAX; // FLT_MAX 位于cfloat头文件中
        for(int i = 0; i < N; i++) {
            m = fmax(m, d_S[row * N + i]);
        }
        
        // 逐行求指数和
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += expf(d_S[row * N + i] - m);
        }
        
        // 写回P
        for (int i = 0; i < N; i++) {
            d_P[row * N + i] = expf(d_S[row * N + i] - m) / sum;
        }
    }
}

__global__ void PVMultiplyKernel(const float* d_P, const float* d_V, float* d_O, int N, int d) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (row < N && col < d) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += d_P[row * N + i] * d_V[i * d + col];
        }
        d_O[row * d + col] = sum;
    }
}

void launch_naive_attention(
    const float* d_Q, const float* d_K, const float* d_V, float* d_O,
    int N, int d
) {
    float* d_S, *d_P;
    cudaMalloc(&d_S, sizeof(float) * N * N);
    cudaMalloc(&d_P, sizeof(float) * N * N);
    
    // 1. S = Q K^T / sqrt(d)
    dim3 block1(TILE, TILE);
    dim3 grid1(cdiv(N, TILE), cdiv(N, TILE)); // 想输出矩阵是 NxN的，每一个线程计算一个输出元素。
    ScaledDotProductKernel<<<grid1, block1>>>(
        d_Q, d_K, d_S, N, d
    );
    // 2. P = softmax(S); per row
    dim3 block2(1);
    dim3 grid2(N);
    SoftmaxKernel<<<grid2, block2>>>(
        d_S, d_P, N
    );
    
    // 3. O = P V
    dim3 block3(TILE, TILE);
    dim3 grid3(cdiv(N, TILE), cdiv(d, TILE));
    PVMultiplyKernel<<<grid3, block3>>>(
        d_P, d_V, d_O, N, d
    );
    
    cudaFree(d_S);
    cudaFree(d_P);
}
