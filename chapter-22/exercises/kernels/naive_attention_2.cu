#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>

#define TILE 16
#define BLOCK_SIZE 256
#define cdiv(a, b) (((a) + (b) - 1) / (b))

__global__ void ScaledDotProductKernel(const float* d_Q, const float* d_K, float* d_S, int N, int d) {
    __shared__ float s_Q[TILE][TILE];
    __shared__ float s_K[TILE][TILE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = tx + blockIdx.x * TILE;
    int col = ty + blockIdx.y * TILE;
    
    float sum = 0.0f;
    
    for (int i = 0; i < cdiv(d, TILE); i++) {
        // 第一步，视角切换，block内每个线程管理一个s_Q和s_K的线程
        if (row < N && i * TILE + ty < d) {
            // s_Q[tx][ty] = Q[row][i*TILE + ty]
            s_Q[tx][ty] = d_Q[row * d + i * TILE + ty];
        } else {
            s_Q[tx][ty] = 0.0f;
        }
        if (col < N && i * TILE + tx < d) {
            // s_K[tx][ty] = K^T[i*TILE + tx][col] = K[col][i*TILE + tx]
            s_K[tx][ty] = d_K[col * d + i * TILE + tx];
        } else {
            s_K[tx][ty] = 0.0f;
        }
        __syncthreads(); // 同步，所有线程现在的状态都一样，才可以重新分工。
        // 视角切换。一个线程用于计算 S[row][col] 的值
        for (int k = 0; k < TILE; k++) {
            sum += s_Q[tx][k] * s_K[k][ty];       
        }
        __syncthreads(); // 同步，线程马上要重新分工去读数据。
    }
    
    if (row < N && col < N) {
        d_S[row * N + col] = sum / sqrtf((float)d);
    }
}

__global__ void SoftmaxKernel(const float* d_S, float* d_P, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float sdata[BLOCK_SIZE];

    // 1. 每个线程遍历多个元素，求局部 max
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, d_S[row * N + i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    // tree reduction 求全局 max
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    float m = sdata[0];

    // 2. 每个线程遍历多个元素，求局部 exp sum
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        local_sum += expf(d_S[row * N + i] - m);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // tree reduction 求全局 sum
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float sum = sdata[0];

    // 3. 写回
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        d_P[row * N + i] = expf(d_S[row * N + i] - m) / sum;
    }
}

__global__ void PVMultiplyKernel(const float* d_P, const float* d_V, float* d_O, int N, int d) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = tx + blockIdx.x * blockDim.x;
    int col = ty + blockIdx.y * blockDim.y;
    
    __shared__ float s_P[TILE][TILE];
    __shared__ float s_V[TILE][TILE];
    
    float sum = 0.0f;
    for (int i = 0; i < cdiv(N, TILE); i++) {
        if (row < N && i * TILE + ty < N) {
            s_P[tx][ty] = d_P[row * N + i * TILE + ty];
        } else {
            s_P[tx][ty] = 0.0f;
        }
        if (i * TILE + tx < N && col < d) {
            s_V[tx][ty] = d_V[(i * TILE + tx) * d + col];
        } else {
            s_V[tx][ty] = 0.0f;
        }
        __syncthreads();
        
        for (int k = 0; k < TILE; k++) {
            sum += s_P[tx][k] * s_V[k][ty];
        }
        __syncthreads();
    }
    
    if (row < N && col < d) {
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
    // 2. P = softmax(S);
    dim3 block2(BLOCK_SIZE);
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
