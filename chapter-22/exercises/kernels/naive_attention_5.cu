#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>

#define TILE 16
#define BLOCK_SIZE 256
#define cdiv(a, b) (((a) + (b) - 1) / (b))
#define COARSE 4

__global__ void ScaleDotProductKernel(const float* d_Q, const float* d_K, float* d_S, int N, int d) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int base_row = blockIdx.y * TILE;
    int base_col = blockIdx.x * TILE * COARSE;
    
    __shared__ float s_Q[TILE][TILE];
    __shared__ float s_K[TILE][TILE];
    float sum[COARSE] = {};
    
    for (int i = 0; i < cdiv(d, TILE); i++) {
        int q_row = base_row + ty;
        int q_col = i * TILE + tx;
        if (q_row < N && q_col < d)
            s_Q[ty][tx] = d_Q[q_row * d + q_col];
        else
            s_Q[ty][tx] = 0.0f;
        
        #pragma unroll
        for (int c = 0; c < COARSE; c++) {
            int k_row = base_col + c * TILE + ty;
            int k_col = i * TILE + tx;
            // s_K[ty][tx] = d_K^T[i * TILE + ty][base_col + c * TILE + tx]
            // s_K[ty][tx] = d_K[base_col + tx][i * TILE + ty]
            // s_K[tx][ty] = d_K[base_col + ty][i * TILE + tx]
            if (k_row < N && k_col < d)
                s_K[tx][ty] = d_K[k_row * d + k_col];
            else
                s_K[tx][ty] = 0.0f;
            __syncthreads();
            
            for (int k = 0; k < TILE; k++) {
                sum[c] += s_Q[ty][k] * s_K[k][tx];
            }
            __syncthreads();
        }
    }
    
    #pragma unroll
    for (int c = 0; c < COARSE; c++) {
        int row = base_row + ty;
        int col = base_col + c * TILE + tx;
        if (row < N && col < N)
            d_S[row * N + col] = sum[c] / sqrtf(float(d));
    }
}

__global__ void SoftmaxKernel(const float* d_S, float* d_P, int N) {
    int row = blockIdx.x;
    int tx = threadIdx.x;
    
    __shared__ float sdata[BLOCK_SIZE];
    
    float local_max = -FLT_MAX;
    for (int i = tx; i < N; i += BLOCK_SIZE) {
        local_max = fmax(local_max, d_S[row * N + i]);
    }
    sdata[tx] = local_max;
    __syncthreads();
    
    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride >>= 1) {
        if (tx < stride) { 
            sdata[tx] = fmax(sdata[tx], sdata[tx + stride]);
        }
        __syncthreads();
    }
    float m = sdata[0];
    
    float local_sum = 0.0f;
    for (int i = tx; i < N; i += BLOCK_SIZE) {
        local_sum += expf(d_S[row * N + i] - m);
    }
    sdata[tx] = local_sum;
    __syncthreads();
    
    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride >>= 1) {
        if (tx < stride) {
            sdata[tx] += sdata[tx + stride];
        }
        __syncthreads();
    }
    float sum = sdata[0];
    
    float scale = 1.0f / sum;
    for (int i = tx; i < N; i += BLOCK_SIZE) {
        d_P[row * N + i] = expf(d_S[row * N + i] - m) * scale;
    }
}

__global__ void PVMultiplyKernel(const float* d_P, const float* d_V, float* d_O, int N, int d) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int base_row = blockIdx.y * TILE;
    int base_col = blockIdx.x * TILE * COARSE;
    
    __shared__ float s_P[TILE][TILE];
    __shared__ float s_V[TILE][TILE];
    float sum[COARSE] = {};
    
    for (int i = 0; i < cdiv(N, TILE); i++) {
        int p_row = base_row + ty;
        int p_col = i * TILE + tx;
        if (p_row < N && p_col < N) 
            s_P[ty][tx] = d_P[p_row * N + p_col];
        else
            s_P[ty][tx] = 0.0f;
        
        #pragma unroll
        for (int c = 0; c < COARSE; c++) {
            int v_row = i * TILE + ty;
            int v_col = base_col + c * TILE + tx;
            if (v_row < N && v_col < d) 
                s_V[ty][tx] = d_V[v_row * d + v_col];
            else
                s_V[ty][tx] = 0.0f;
            __syncthreads();
            
            for (int k = 0; k < TILE; k++) {
                sum[c] += s_P[ty][k] * s_V[k][tx];
            }
            __syncthreads();
        }
    }
    
    #pragma unroll
    for (int c = 0; c < COARSE; c++) {
        int row = base_row + ty;
        int col = base_col + c * TILE + tx;
        if (row < N && col < d) 
            d_O[row * d + col] = sum[c];
    }
}

void launch_naive_attention(
    const float* d_Q, const float* d_K, const float* d_V, float* d_O,
    int N, int d
) {
    float* d_P, *d_S;
    cudaMalloc(&d_P, sizeof(float) * N * N);
    cudaMalloc(&d_S, sizeof(float) * N * N);
    
    dim3 block1(TILE, TILE);
    dim3 grid1(cdiv(N, TILE * COARSE), cdiv(N, TILE));
    ScaleDotProductKernel<<<grid1, block1>>>(
        d_Q, d_K, d_S, N, d
    );
    
    dim3 block2(BLOCK_SIZE);
    dim3 grid2(N);
    SoftmaxKernel<<<grid2, block2>>>(
        d_S, d_P, N
    );
    
    dim3 block3(TILE, TILE);
    dim3 grid3(cdiv(d, TILE * COARSE), cdiv(N, TILE));
    PVMultiplyKernel<<<grid3, block3>>>(
        d_P, d_V, d_O, N, d
    );
    
    cudaFree(d_S);
    cudaFree(d_P);
}
