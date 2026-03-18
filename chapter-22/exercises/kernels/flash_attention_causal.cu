#include <torch/extension.h>
#include <cmath>
#include <cfloat>

// FlashAttention 因果掩码版
//
// 输入:
//   Q: (N, d) 查询矩阵
//   K: (N, d) 键矩阵
//   V: (N, d) 值矩阵
//
// 输出:
//   O: (N, d) = softmax(causal_mask(Q @ K^T / sqrt(d))) @ V
//
// 因果掩码规则：
//   对于 S[i][j]，若 j > i，则 S[i][j] = -inf（该位置不参与 softmax）
//
// 与非因果版的区别：
//   在分块处理时，对于第 i 个 Q 块和第 j 个 K/V 块：
//   - 若 j * B_c > (i+1) * B_r - 1：整个块在掩码之外，跳过
//   - 若 (j+1) * B_c - 1 <= i * B_r：整个块在掩码之内，正常计算
//   - 否则：需要逐元素检查 global_row >= global_col
//
// 优化要点：
//   跳过完全被掩码的块可以节省约一半的计算量
//
// 共享内存布局同分块版

#define B_r 16
#define B_c 16
#define D_MAX 64

// TODO: 实现 FlashAttentionCausalKernel
//
// 在分块版的基础上添加：
//   1. 内层循环的提前终止：当 K/V 块完全在掩码外时 break
//   2. 对角块的逐元素掩码：S[r][c] = (global_row >= global_col) ? S[r][c] : -FLT_MAX
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
    // float m = -FLT_MAX;
    // float l = 0.0f;
    // float o = 0.0f;
    //
    // // 加载 Q_tile ...
    //
    // // 因果掩码：只需遍历到 j 使得 j * B_c <= global_row 的最后一个块
    // int max_kv_tile = min((q_start + B_r - 1) / B_c + 1, (N + B_c - 1) / B_c);
    //
    // for (int j = 0; j < max_kv_tile; j++) {
    //     int kv_start = j * B_c;
    //
    //     // 加载 K_tile, V_tile ...
    //
    //     // 计算 S[row_in_block][col]
    //     // 应用因果掩码：
    //     // for (int c = 0; c < B_c; c++) {
    //     //     int global_col = kv_start + c;
    //     //     if (global_col > global_row)
    //     //         S_val = -FLT_MAX;
    //     // }
    //
    //     // 在线 softmax 更新 ...
    //     // 更新 o ...
    // }
    //
    // if (global_row < N)
    //     O[global_row * d + dim_idx] = o;
}

// TODO: 实现 host 函数
// 函数签名: torch::Tensor flashAttentionCausal(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
torch::Tensor flashAttentionCausal(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: 你的代码
    return torch::zeros_like(Q);
}
