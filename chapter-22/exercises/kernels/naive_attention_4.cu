#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>

#define TILE 16
#define BLOCK_SIZE 256
#define cdiv(a, b) (((a) + (b) - 1) / (b))
#define COARSE 4

__global__ void ScaledDotProductKernel(const float* d_Q, const float* d_K, float* d_S, int N, int d) {
    int row = threadIdx.x + blockIdx.x * TILE;
    int col = threadIdx.y + blockIdx.y * TILE * COARSE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    __shared__ float s_Q[TILE][TILE];
    __shared__ float s_K[TILE][TILE * COARSE];
    
    float sum[COARSE] = {};
    
    for (int i = 0; i < cdiv(d, TILE); i++) {
        // s_Q[tx][ty] = d_Q[row][i * TILE + ty]
        if (row < N && i * TILE + ty < d)
            s_Q[tx][ty] = d_Q[row * d + i * TILE + ty];
        else
            s_Q[tx][ty] = 0.0f;
        
        // s_K[tx][ty] = d_K^T[i * TILE + tx][col]
        #pragma unroll
        for (int c = 0; c < COARSE; c++) {
            int cur_col = col + c * TILE;
            if (cur_col < N && i * TILE + tx < d)
                s_K[tx][ty + c * TILE] = d_K[cur_col * d + i * TILE + tx];
            else
                s_K[tx][ty + c * TILE] = 0.0f;
        }
        __syncthreads();
        
        for (int k = 0; k < TILE; k++) {
            for (int c = 0; c < COARSE; c++) {
                sum[c] += s_Q[tx][k] * s_K[k][ty + c * TILE];
            }
        }
        __syncthreads();
    }
    
    float scale = 1.0f / sqrtf((float)d);
    #pragma unroll
    for (int c = 0; c < COARSE; c++) {
        int cur_col = col + c * TILE;
        if (row < N && cur_col < N) 
            d_S[row * N + cur_col] = sum[c] * scale;
    }
}

__global__ void SoftmaxKernel(const float* d_S, float* d_P, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float sdata[BLOCK_SIZE];
    
    // 1. 求局部最大值
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        local_max = fmax(local_max, d_S[row * N + i]);
    }
    sdata[tid] = local_max;
    __syncthreads();
    
    // 归约，求整行的最大值
    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    float m = sdata[0];
    
    // 2. 自然指数求和
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        local_sum += expf(d_S[row * N + i] - m);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    
    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float sum = sdata[0];
    
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
    dim3 grid1(cdiv(N, TILE), cdiv(N, TILE * COARSE));
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
