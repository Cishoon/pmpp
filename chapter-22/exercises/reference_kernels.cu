#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

// ============================================================
// 参考实现：用于生成 ground truth（不要修改此文件）
// 基于 code/flash_attention.cu 的分块 + 在线 softmax 实现
// ============================================================

#define REF_B_r 16
#define REF_B_c 16
#define REF_D_MAX 64

__global__ void ReferenceAttentionKernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d, int causal
) {
    int q_block = blockIdx.x;
    int q_start = q_block * REF_B_r;
    int row_in_block = threadIdx.x;
    int dim_idx = threadIdx.y;
    int global_row = q_start + row_in_block;

    if (global_row >= N || dim_idx >= d) return;

    __shared__ float Q_tile[REF_B_r][REF_D_MAX];
    __shared__ float K_tile[REF_B_c][REF_D_MAX];
    __shared__ float V_tile[REF_B_c][REF_D_MAX];
    __shared__ float S_tile[REF_B_r][REF_B_c];

    Q_tile[row_in_block][dim_idx] = Q[global_row * d + dim_idx];
    __syncthreads();

    float m = -FLT_MAX;
    float l = 0.0f;
    float o = 0.0f;

    int num_kv_tiles = (N + REF_B_c - 1) / REF_B_c;

    for (int j = 0; j < num_kv_tiles; j++) {
        int kv_start = j * REF_B_c;

        // 因果掩码：整个块在掩码外则跳过
        if (causal && kv_start > global_row) break;

        // 加载 K_tile, V_tile
        if (row_in_block < REF_B_c && (kv_start + row_in_block) < N) {
            K_tile[row_in_block][dim_idx] = K[(kv_start + row_in_block) * d + dim_idx];
            V_tile[row_in_block][dim_idx] = V[(kv_start + row_in_block) * d + dim_idx];
        } else if (row_in_block < REF_B_c) {
            K_tile[row_in_block][dim_idx] = 0.0f;
            V_tile[row_in_block][dim_idx] = 0.0f;
        }
        __syncthreads();

        // 计算 S_tile
        for (int c = 0; c < REF_B_c; c++) {
            int global_col = kv_start + c;
            if (global_col < N) {
                float dot = 0.0f;
                for (int k = 0; k < d; k++) {
                    dot += Q_tile[row_in_block][k] * K_tile[c][k];
                }
                float s_val = dot / sqrtf((float)d);
                // 因果掩码
                if (causal && global_col > global_row) {
                    s_val = -FLT_MAX;
                }
                S_tile[row_in_block][c] = s_val;
            } else {
                S_tile[row_in_block][c] = -FLT_MAX;
            }
        }
        __syncthreads();

        // 本块 max
        float m_j = -FLT_MAX;
        for (int c = 0; c < REF_B_c; c++) {
            m_j = fmaxf(m_j, S_tile[row_in_block][c]);
        }

        // 本块 exp-sum
        float l_j = 0.0f;
        for (int c = 0; c < REF_B_c; c++) {
            l_j += expf(S_tile[row_in_block][c] - m_j);
        }

        // 在线 softmax 更新
        float m_new = fmaxf(m, m_j);
        float l_new = l * expf(m - m_new) + l_j * expf(m_j - m_new);

        o = o * (l * expf(m - m_new) / l_new);
        for (int c = 0; c < REF_B_c && (kv_start + c) < N; c++) {
            if (!causal || (kv_start + c) <= global_row) {
                o += expf(S_tile[row_in_block][c] - m_new) / l_new * V_tile[c][dim_idx];
            }
        }

        m = m_new;
        l = l_new;
        __syncthreads();
    }

    O[global_row * d + dim_idx] = o;
}

// 供 test_harness 调用的接口
void launch_reference_attention(
    const float* d_Q, const float* d_K, const float* d_V, float* d_O,
    int N, int d, bool causal
) {
    dim3 block(REF_B_r, d);
    dim3 grid((N + REF_B_r - 1) / REF_B_r);
    ReferenceAttentionKernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, N, d, causal ? 1 : 0);
    cudaDeviceSynchronize();
}
