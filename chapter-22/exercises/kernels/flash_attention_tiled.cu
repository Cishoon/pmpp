#include <torch/extension.h>
#include <cmath>
#include <cfloat>

// FlashAttention 分块版（多 block，Q 也分块）
//
// 输入:
//   Q: (N, d) 查询矩阵
//   K: (N, d) 键矩阵
//   V: (N, d) 值矩阵
//
// 输出:
//   O: (N, d) = softmax(Q @ K^T / sqrt(d)) @ V
//
// 与单 block 版的区别：
//   - Q 也按 B_r 分块，每个 block 处理 B_r 行 Q
//   - 共享内存中同时存放 Q_tile(B_r × d), K_tile(B_c × d), V_tile(B_c × d)
//   - 需要计算 S_tile(B_r × B_c) 的小矩阵乘法
//
// 算法（每个 block 处理第 i 个 Q 块）：
//
//   1. 加载 Q_i (B_r × d) 到共享内存
//   2. 初始化 m[B_r] = -inf, l[B_r] = 0, O_local[B_r][d] = 0
//
//   3. for j in 0..T_c:
//        a. 加载 K_j, V_j 到共享内存
//        b. 计算 S_ij = Q_i @ K_j^T / sqrt(d)，形状 (B_r, B_c)
//        c. 对 S_ij 的每一行：
//           - m_j = max(S_ij[row])
//           - m_new = max(m[row], m_j)
//           - l_new = l[row] * exp(m[row] - m_new) + sum(exp(S_ij[row] - m_new))
//           - O_local[row] = O_local[row] * (l[row] * exp(m[row] - m_new) / l_new)
//                          + (exp(S_ij[row] - m_new) / l_new) @ V_j
//           - m[row] = m_new, l[row] = l_new
//
//   4. 将 O_local 写回 O
//
// 线程分工建议：
//   - 2D block: (B_r, d) 或 (B_r, B_c)
//   - 每个线程负责 O 的一个元素
//
// 共享内存布局：
//   Q_tile: B_r × d
//   K_tile: B_c × d
//   V_tile: B_c × d
//   S_tile: B_r × B_c（可选，也可用寄存器）

#define B_r 16   // Q 分块大小
#define B_c 16   // K/V 分块大小
#define D_MAX 64 // 最大 head 维度

// TODO: 实现 FlashAttentionTiledKernel
//
// blockIdx.x 对应第几个 Q 块
// 2D block: blockDim.x = B_r, blockDim.y = d（或其他合理分工）
__global__ void FlashAttentionTiledKernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d
) {
    // TODO: 你的代码
    //
    // 提示框架：
    //
    // int q_block = blockIdx.x;  // 第几个 Q 块
    // int q_start = q_block * B_r;
    // int row_in_block = threadIdx.x;  // 块内第几行
    // int dim_idx = threadIdx.y;       // 第几个维度
    //
    // __shared__ float Q_tile[B_r][D_MAX];
    // __shared__ float K_tile[B_c][D_MAX];
    // __shared__ float V_tile[B_c][D_MAX];
    //
    // // 加载 Q_tile
    //
    // float m = -FLT_MAX;
    // float l = 0.0f;
    // float o = 0.0f;
    //
    // int num_kv_tiles = (N + B_c - 1) / B_c;
    //
    // for (int j = 0; j < num_kv_tiles; j++) {
    //     // 1. 协作加载 K_tile, V_tile
    //     // 2. 计算 S[row_in_block][col] for col in [0, B_c)
    //     //    S[r][c] = sum_k(Q_tile[r][k] * K_tile[c][k]) / sqrt(d)
    //     // 3. 在线 softmax 更新 m, l
    //     // 4. 更新 o（注意 rescale 旧的累加值）
    // }
    //
    // // 写回 O
    // int global_row = q_start + row_in_block;
    // if (global_row < N)
    //     O[global_row * d + dim_idx] = o;
}

// TODO: 实现 host 函数
// 函数签名: torch::Tensor flashAttentionTiled(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
//
// 步骤：
//   1. 获取 N, d
//   2. 分配 O(N, d)
//   3. 计算 grid: (ceil(N/B_r),)
//   4. 计算 block: (B_r, d)
//   5. 共享内存大小: (B_r*d + 2*B_c*d) * sizeof(float)
//   6. 启动内核并返回 O
torch::Tensor flashAttentionTiled(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: 你的代码
    return torch::zeros_like(Q);
}
