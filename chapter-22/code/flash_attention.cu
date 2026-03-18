#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cfloat>

// ============================================================
// FlashAttention 参考实现
// 分块 + 在线 softmax，避免 O(N²) 额外内存
// ============================================================

#define B_r 16
#define B_c 16
#define D_MAX 64

__global__ void FlashAttentionKernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d
) {
    int q_block = blockIdx.x;
    int q_start = q_block * B_r;
    int row_in_block = threadIdx.x;
    int dim_idx = threadIdx.y;
    int global_row = q_start + row_in_block;

    if (global_row >= N || dim_idx >= d) return;

    __shared__ float Q_tile[B_r][D_MAX];
    __shared__ float K_tile[B_c][D_MAX];
    __shared__ float V_tile[B_c][D_MAX];

    // 加载 Q_tile
    Q_tile[row_in_block][dim_idx] = Q[global_row * d + dim_idx];
    __syncthreads();

    float m = -FLT_MAX;
    float l = 0.0f;
    float o = 0.0f;

    int num_kv_tiles = (N + B_c - 1) / B_c;

    for (int j = 0; j < num_kv_tiles; j++) {
        int kv_start = j * B_c;

        // 协作加载 K_tile, V_tile
        if (row_in_block < B_c && (kv_start + row_in_block) < N) {
            K_tile[row_in_block][dim_idx] = K[(kv_start + row_in_block) * d + dim_idx];
            V_tile[row_in_block][dim_idx] = V[(kv_start + row_in_block) * d + dim_idx];
        } else if (row_in_block < B_c) {
            K_tile[row_in_block][dim_idx] = 0.0f;
            V_tile[row_in_block][dim_idx] = 0.0f;
        }
        __syncthreads();

        // 计算 S[row_in_block][c] 并做在线 softmax
        // 先求本块的 max 和 exp-sum
        float m_j = -FLT_MAX;
        float l_j = 0.0f;

        // 每个线程需要计算完整的 s 向量（跨维度归约）
        // 这里用共享内存做点积
        __shared__ float S_tile[B_r][B_c];

        // 计算 S_tile[row_in_block][c]
        for (int c = 0; c < B_c && (kv_start + c) < N; c++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++) {
                dot += Q_tile[row_in_block][k] * K_tile[c][k];
            }
            S_tile[row_in_block][c] = dot / sqrtf((float)d);
        }
        // 超出范围的填 -inf
        for (int c = N - kv_start; c < B_c; c++) {
            if (c >= 0) S_tile[row_in_block][c] = -FLT_MAX;
        }
        __syncthreads();

        // 求本块 max
        for (int c = 0; c < B_c; c++) {
            m_j = fmaxf(m_j, S_tile[row_in_block][c]);
        }

        // 求本块 exp-sum
        for (int c = 0; c < B_c; c++) {
            l_j += expf(S_tile[row_in_block][c] - m_j);
        }

        // 在线 softmax 更新
        float m_new = fmaxf(m, m_j);
        float l_new = l * expf(m - m_new) + l_j * expf(m_j - m_new);

        // 更新 o
        // o = o * (l * exp(m - m_new) / l_new) + sum_c(exp(S[r][c] - m_new) / l_new * V[c][dim])
        o = o * (l * expf(m - m_new) / l_new);
        for (int c = 0; c < B_c && (kv_start + c) < N; c++) {
            o += expf(S_tile[row_in_block][c] - m_new) / l_new * V_tile[c][dim_idx];
        }

        m = m_new;
        l = l_new;
        __syncthreads();
    }

    O[global_row * d + dim_idx] = o;
}

void flash_attention(float* Q, float* K, float* V, float* O, int N, int d) {
    float *d_Q, *d_K, *d_V, *d_O;
    size_t qkv_size = N * d * sizeof(float);

    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_O, qkv_size);

    cudaMemcpy(d_Q, Q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, qkv_size, cudaMemcpyHostToDevice);

    dim3 block(B_r, d);
    dim3 grid((N + B_r - 1) / B_r);
    FlashAttentionKernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, N, d);

    cudaMemcpy(O, d_O, qkv_size, cudaMemcpyDeviceToHost);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
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

    flash_attention(Q, K, V, O, N, d);

    printf("FlashAttention 完成: N=%d, d=%d\n", N, d);
    printf("O[0][0..3] = %.4f %.4f %.4f %.4f\n", O[0], O[1], O[2], O[3]);

    free(Q); free(K); free(V); free(O);
    return 0;
}
