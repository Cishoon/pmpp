#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>

// ============================================================
// 代码题 D: FlashAttention 因果掩码版
//
// 在分块版（代码题 C）的基础上添加因果掩码：
//   S[i][j] 在 j > i 时设为 -inf
//
// 分块级别的优化：
//   对于第 i 个 Q 块（行范围 [q_start, q_start+B_r)）
//   和第 j 个 K/V 块（列范围 [kv_start, kv_start+B_c)）：
//
//   - 若 kv_start > q_start + B_r - 1：整个块被掩码，跳过（break）
//   - 若 kv_start + B_c - 1 <= q_start：整个块不受掩码影响，正常计算
//   - 否则（对角块）：逐元素检查 global_row >= global_col
//
// 约束：不要修改函数签名，d ≤ D_MAX
// ============================================================

#define B_r 16
#define B_c 16
#define D_MAX 64

// TODO: 实现 FlashAttentionCausalKernel
__global__ void FlashAttentionCausalKernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d
) {
    // TODO: 你的代码
    //
    // 提示框架：
    //
    // int q_block = blockIdx.x;
    // int q_start = q_block * B_r;
    // int row_in_block = threadIdx.x;
    // int dim_idx = threadIdx.y;
    // int global_row = q_start + row_in_block;
    //
    // // 加载 Q_tile（同代码题 C）...
    //
    // float m = -FLT_MAX;
    // float l = 0.0f;
    // float o = 0.0f;
    //
    // // 因果掩码：最多遍历到包含 global_row 的 K/V 块
    // int max_kv_tile = min((q_start + B_r - 1) / B_c + 1,
    //                       (N + B_c - 1) / B_c);
    //
    // for (int j = 0; j < max_kv_tile; j++) {
    //     int kv_start = j * B_c;
    //     // 加载 K_tile, V_tile ...
    //     // 计算 S[row_in_block][c]
    //     // 应用因果掩码：
    //     //   int global_col = kv_start + c;
    //     //   if (global_col > global_row) S_val = -FLT_MAX;
    //     // 在线 softmax 更新 ...
    //     // 更新 o ...
    // }
    //
    // if (global_row < N)
    //     O[global_row * d + dim_idx] = o;
}

// launch 函数（is_causal 参数在此内核中始终为 true）
void launch_flash_attention_causal(
    const float* d_Q, const float* d_K, const float* d_V, float* d_O,
    int N, int d
) {
    // TODO: 你的代码
    // grid: (ceil(N/B_r)),  block: (B_r, d)
}
