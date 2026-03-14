#include <torch/extension.h>

// 分块矩阵转置
// 输入 A (m×n)，输出 B (n×m)，即 B[j][i] = A[i][j]
//
// 为什么需要 shared memory？
//   朴素版本：thread(row,col) 读 A[row*n+col]，写 B[col*m+row]
//   读是 coalesced（同一 warp 的线程读连续地址），但写不是（列方向跳跃）
//   用 shared memory 做中转，可以让读和写都 coalesced
//
// 思路：
//   每个线程块负责一个 TILE×TILE 的输入分块
//   Step 1: 协作从 A 中 coalesced 读入 tile 到 shared memory（行优先读）
//   Step 2: __syncthreads()
//   Step 3: 从 shared memory 转置读出，coalesced 写入 B
//           注意写入 B 时，线程的 x/y 角色要对调
//
// 关键点：shared memory 的读写下标如何对调才能实现转置？

#define TILE_WIDTH 16

__global__ void TiledTransposeKernel(const float* A, float* B, int m, int n) {
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    if (row < m && col < n) {
        tile[threadIdx.y][threadIdx.x] = A[row * n + col];
    }
    __syncthreads();
    
    int B_row = col;
    int B_col = row;
    if (B_row < n && B_col < m) {
        B[B_row * m + B_col] = tile[threadIdx.y][threadIdx.x];
    }
}

torch::Tensor tiledTranspose(torch::Tensor A) {
    int m = A.size(0), n = A.size(1);
    auto B = torch::zeros({n, m}, A.options());

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((n + TILE_WIDTH - 1) / TILE_WIDTH,
              (m + TILE_WIDTH - 1) / TILE_WIDTH);

    TiledTransposeKernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), m, n);

    return B;
}
