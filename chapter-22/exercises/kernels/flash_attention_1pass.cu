#include <torch/extension.h>
#include <cmath>
#include <cfloat>

// FlashAttention 单 block 版（1-pass 在线 softmax）
//
// 输入:
//   Q: (N, d) 查询矩阵
//   K: (N, d) 键矩阵
//   V: (N, d) 值矩阵
//
// 输出:
//   O: (N, d) = softmax(Q @ K^T / sqrt(d)) @ V
//
// 核心思想：
//   对于 Q 的每一行 q（由一个 thread block 处理），
//   将 K 和 V 按 B_c 大小分块，逐块处理：
//
//   初始化: m = -inf, l = 0, o = 0（长度为 d 的向量）
//
//   对每个 K/V 块 j:
//     1. 从 HBM 加载 K_j (B_c × d) 到共享内存
//     2. 计算 s_j = q @ K_j^T / sqrt(d)，得到长度为 B_c 的向量
//     3. 求 m_j = max(s_j)
//     4. 更新: m_new = max(m, m_j)
//     5. 更新: l_new = l * exp(m - m_new) + sum(exp(s_j - m_new))
//     6. 更新: o = o * (l * exp(m - m_new) / l_new)
//              + (exp(s_j - m_new) / l_new) @ V_j
//     7. m = m_new, l = l_new
//
//   将 o 写回 O[row]
//
// 限制：
//   - 每个 block 处理 Q 的一行
//   - d 必须足够小以放入寄存器/共享内存
//   - B_c 为 K/V 的分块大小
//
// 共享内存布局：
//   K_tile: B_c × d
//   V_tile: B_c × d

#define B_c 32   // K/V 分块大小
#define D_MAX 64 // 最大 head 维度

// TODO: 实现 FlashAttention1PassKernel
//
// 每个 block 处理 Q 的一行（blockIdx.x = row）
// block 内的线程协作加载 K/V 块到共享内存，
// 然后每个线程独立计算该行的一部分输出维度
//
// 建议的线程分工：
//   - blockDim.x = d（每个线程负责输出的一个维度）
//   - 线程协作加载 K_tile 和 V_tile
//   - 每个线程维护自己的 o[threadIdx.x], m, l
//   - 内层循环中，所有线程需要共享 s_j 向量（通过共享内存）
__global__ void FlashAttention1PassKernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d
) {
    // TODO: 你的代码
    //
    // 提示框架：
    //
    // int row = blockIdx.x;
    // int tx = threadIdx.x;  // 对应输出维度 tx
    //
    // __shared__ float K_tile[B_c][D_MAX];
    // __shared__ float V_tile[B_c][D_MAX];
    // __shared__ float s[B_c];  // 当前块的注意力分数
    //
    // float q_val[?];  // 缓存当前行的 Q 值（可选）
    // float m = -FLT_MAX;
    // float l = 0.0f;
    // float o = 0.0f;
    //
    // int num_tiles = (N + B_c - 1) / B_c;
    //
    // for (int tile = 0; tile < num_tiles; tile++) {
    //     // 1. 协作加载 K_tile 和 V_tile
    //     // 2. __syncthreads()
    //     // 3. 计算 s[j] = dot(q, K_tile[j]) / sqrt(d)（需要归约）
    //     // 4. 求 m_new = max(m, max(s))
    //     // 5. 求 l_new
    //     // 6. 更新 o
    //     // 7. __syncthreads()
    // }
    //
    // O[row * d + tx] = o;
}

// TODO: 实现 host 函数
// 函数签名: torch::Tensor flashAttention1Pass(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
//
// 步骤：
//   1. 获取 N, d
//   2. 分配 O(N, d)
//   3. 启动内核: grid(N), block(d)
//   4. 返回 O
torch::Tensor flashAttention1Pass(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: 你的代码
    return torch::zeros_like(Q);
}
