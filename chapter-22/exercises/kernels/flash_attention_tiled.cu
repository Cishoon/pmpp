#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>

// ============================================================
// 代码题 C: FlashAttention 分块版（Q 也分块，多 block）
//
// 与单 block 版的区别：Q 也按 B_r 分块，每个 block 处理 B_r 行。
// 共享内存同时存放 Q_tile, K_tile, V_tile。
//
// 算法（每个 block 处理第 i 个 Q 块）：
//   1. 加载 Q_i(B_r × d) 到共享内存
//   2. 初始化 m = -inf, l = 0, o = 0（每个线程对应一个 (row, dim)）
//   3. for j in 0..T_c:
//        a. 加载 K_j(B_c × d), V_j(B_c × d) 到共享内存
//        b. 计算 S[row_in_block][c] = dot(Q_tile[row], K_tile[c]) / sqrt(d)
//        c. 在线 softmax 更新 m, l
//        d. 更新 o（rescale 旧值 + 加入新贡献）
//   4. 写回 O[global_row][dim] = o
//
// 线程分工：
//   2D block: blockDim.x = B_r（行）, blockDim.y = d（维度）
//   每个线程负责 O 的一个元素 O[q_start + threadIdx.x][threadIdx.y]
//
// 共享内存布局：
//   Q_tile[B_r][D_MAX]
//   K_tile[B_c][D_MAX]
//   V_tile[B_c][D_MAX]
//
// 约束：不要修改函数签名，d ≤ D_MAX
// ============================================================

#define B_r 16
#define B_c 16
#define D_MAX 64

// TODO: 实现 FlashAttentionTiledKernel
__global__ void FlashAttentionTiledKernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d
) {
    // TODO: 你的代码
    //
    // 提示框架：
    //
    // int q_block = blockIdx.x;
    // int q_start = q_block * B_r;
    // int row_in_block = threadIdx.x;  // [0, B_r)
    // int dim_idx = threadIdx.y;       // [0, d)
    // int global_row = q_start + row_in_block;
    //
    // __shared__ float Q_tile[B_r][D_MAX];
    // __shared__ float K_tile[B_c][D_MAX];
    // __shared__ float V_tile[B_c][D_MAX];
    //
    // // 加载 Q_tile
    // if (global_row < N)
    //     Q_tile[row_in_block][dim_idx] = Q[global_row * d + dim_idx];
    // else
    //     Q_tile[row_in_block][dim_idx] = 0.0f;
    // __syncthreads();
    //
    // float m = -FLT_MAX;
    // float l = 0.0f;
    // float o = 0.0f;
    //
    // int num_kv_tiles = (N + B_c - 1) / B_c;
    // for (int j = 0; j < num_kv_tiles; j++) {
    //     int kv_start = j * B_c;
    //     // 1. 协作加载 K_tile, V_tile（注意边界填 0）
    //     // 2. __syncthreads()
    //     // 3. 对每个 c ∈ [0, B_c)，计算 S_val = dot(Q_tile[row], K_tile[c]) / sqrt(d)
    //     //    注意：这里需要遍历 d 维度求点积，但当前线程只负责一个 dim
    //     //    方案：用共享内存存 S_tile[B_r][B_c]，让所有 dim 线程协作归约
    //     //    或者：每个线程独立遍历 d 维度（因为 Q_tile 和 K_tile 都在共享内存中）
    //     // 4. 在线 softmax 更新 m, l
    //     // 5. 更新 o
    //     // 6. __syncthreads()
    // }
    //
    // if (global_row < N)
    //     O[global_row * d + dim_idx] = o;
}

// launch 函数
void launch_flash_attention_tiled(
    const float* d_Q, const float* d_K, const float* d_V, float* d_O,
    int N, int d
) {
    // TODO: 你的代码
    // grid: (ceil(N/B_r)),  block: (B_r, d)
}
