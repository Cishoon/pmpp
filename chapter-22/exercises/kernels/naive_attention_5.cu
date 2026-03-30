#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>

#define TILE 16
#define BLOCK_SIZE 256
#define cdiv(a, b) (((a) + (b) - 1) / (b))
#define COARSE 4

__global__ void ScaledDotProductKernel(const float* d_Q, const float* d_K, float* d_S, int N, int d) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = threadIdx.y + blockIdx.y * TILE;
    int col = threadIdx.x + blockIdx.x * TILE * COARSE;
    
    __shared__ float s_Q[TILE][TILE + 1];
    __shared__ float s_K[TILE][TILE + 1];
    
    float sum[COARSE] = {};
    
    for (int i = 0; i < cdiv(d, TILE); i++) {
        if (row < N && i * TILE + tx < d)
            s_Q[ty][tx] = d_Q[row * d + i * TILE + tx];
        else
            s_Q[ty][tx] = 0.0f;
        
        #pragma unroll
        for (int c = 0; c < COARSE; c++) {
            // 只看 块号，K^T
            // k_row = i * TLIE
            // k_col = col + c * TILE = tx + blockIdx.x * TILE * COARSE + c * TILE; 
            // 只看块号，所以tx去掉
            // k_col = blockIdx.x * TILE * COARSE + c * TILE; 
            // s_K[ty][tx] = d_K^T[k_row + ty][k_col + tx]
            // 接下来转置回来
            // s_K[ty][tx] = d_K[k_col + tx][k_row + ty]
            // 到这就是3.3的写法，但是还是ty在行序上，我们直接替换 tx 和 ty，不会改变取的是哪个BLOCK，只是让这个BLOCK内部发生了一次转置
            // s_K[ty][tx] = d_K[k_col + ty][k_row + tx]
            // 后续做矩阵乘法的时候，再取 s_K^T 就又转置回去了。
            int k_col = blockIdx.x * TILE * COARSE + c * TILE;
            int k_row = i * TILE;
            if (k_col + ty < N && k_row + tx < d)
                s_K[ty][tx] = d_K[(k_col + ty) * d + k_row + tx];
            else
                s_K[ty][tx] = 0.0f;
            __syncthreads();
            
            for (int k = 0; k < TILE; k++) {
                sum[c] += s_Q[ty][k] * s_K[tx][k];
            }
            __syncthreads();
        }
    }
    
    float scale = 1.0f / sqrtf((float)d);
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
    int row = threadIdx.y + blockIdx.y * TILE;
    int col = threadIdx.x + blockIdx.x * TILE * COARSE;
    
    __shared__ float s_P[TILE][TILE];
    __shared__ float s_V[TILE][TILE];
    
    float sum[COARSE] = {};
    for (int i = 0; i < cdiv(N, TILE); i++) {
        if (row < N && i * TILE + tx < N)
            s_P[ty][tx] = d_P[row * N + i * TILE + tx];
        else 
            s_P[ty][tx] = 0.0f;
        
        #pragma unroll
        for (int c = 0; c < COARSE; c++) {
            int cur_row = ty + i * TILE;
            int cur_col = col + c * TILE;
            if (cur_row < N && cur_col < d)
                s_V[ty][tx] = d_V[cur_row * d + cur_col];
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
        int cur_col = col + c * TILE;
        if (row < N && cur_col < d)
            d_O[row * d + cur_col] = sum[c];
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
    dim3 grid1(cdiv(N, TILE * COARSE), cdiv(N, TILE));
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
    dim3 grid3(cdiv(d, TILE * COARSE), cdiv(N, TILE));
    PVMultiplyKernel<<<grid3, block3>>>(
        d_P, d_V, d_O, N, d
    );
    
    cudaFree(d_S);
    cudaFree(d_P);
}
