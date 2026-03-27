#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>

// ============================================================
// 代码题 B: FlashAttention 单 block（1-pass 在线 softmax）
//
// 每个 thread block 处理 Q 的一行。
// 将 K 和 V 按 B_c 分块，逐块加载到共享内存，
// 使用在线 softmax 动态维护 m（最大值）和 l（exp 之和），
// 在寄存器中累加输出 o，最后一次写回 HBM。
//
// 算法（对于 Q 的第 row 行）：
//   初始化: m = -inf, l = 0, o[dim] = 0
//   for each K/V tile j:
//     1. 加载 K_j(B_c × d), V_j(B_c × d) 到共享内存
//     2. 计算 s[c] = dot(Q[row], K_j[c]) / sqrt(d), c ∈ [0, B_c)
//     3. m_tile = max(s[0..B_c-1])
//     4. m_new  = max(m, m_tile)
//     5. l_new  = l * exp(m - m_new) + sum_c(exp(s[c] - m_new))
//     6. o[dim] = o[dim] * (l * exp(m - m_new) / l_new)
//                 + sum_c(exp(s[c] - m_new) / l_new * V_j[c][dim])
//     7. m = m_new, l = l_new
//   写回 O[row][dim] = o[dim]
//
// 线程分工：
//   blockDim.x = d，每个线程负责输出的一个维度
//   所有线程协作加载 K/V tile，并通过共享内存共享 s[] 向量
//
// 共享内存布局：
//   K_tile[B_c][D_MAX], V_tile[B_c][D_MAX], s[B_c]
//
// 约束：不要修改函数签名，d ≤ D_MAX
// ============================================================

#define B_c 32
#define D_MAX 64

// TODO: 实现 FlashAttention1PassKernel
__global__ void FlashAttention1PassKernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d
) {
    // TODO: 你的代码
    //
    // 提示框架：
    //
    // int row = blockIdx.x;
    // int tx = threadIdx.x;  // 对应输出维度
    // if (row >= N || tx >= d) return;
    //
    // __shared__ float K_tile[B_c][D_MAX];
    // __shared__ float V_tile[B_c][D_MAX];
    // __shared__ float s[B_c];
    //
    // // 缓存 Q[row] 到寄存器
    // float q_reg = Q[row * d + tx];
    //
    // float m = -FLT_MAX;
    // float l = 0.0f;
    // float o = 0.0f;
    //
    // int num_tiles = (N + B_c - 1) / B_c;
    // for (int tile = 0; tile < num_tiles; tile++) {
    //     int kv_start = tile * B_c;
    //     // 1. 协作加载 K_tile, V_tile（注意边界）
    //     // 2. __syncthreads()
    //     // 3. 计算 s[c]（需要跨线程归约求点积，或让每个线程算完整点积）
    //     //    一种简单做法：用共享内存做归约
    //     //    另一种：如果 d 不大，每个线程遍历所有 d 维度算点积
    //     // 4. 在线 softmax 更新 m, l, o
    //     // 5. __syncthreads()
    // }
    //
    // O[row * d + tx] = o;
}

// launch 函数
// d_Q, d_K, d_V: device 输入 (N, d)
// d_O: device 输出 (N, d)
void launch_flash_attention_1pass(
    const float* d_Q, const float* d_K, const float* d_V, float* d_O,
    int N, int d
) {
    // TODO: 你的代码
    // grid: (N),  block: (d)
}
