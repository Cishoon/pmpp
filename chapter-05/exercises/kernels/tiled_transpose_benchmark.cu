#include <torch/extension.h>

#define TILE_WIDTH 16

// ============================================================
// 题目 A：朴素转置
// 每个线程负责一个元素，直接读 A[row][col] 写 B[col][row]
// 不使用 shared memory
// ============================================================
__global__ void NaiveTransposeKernel(const float* A, float* B, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < n && row < m)
        // B[col][row] = A[row][col]; 这样是不可以吗？
        B[col * m + row] = A[row * n + col];
}

// ============================================================
// 题目 B：分块转置（用 shared memory 消除 uncoalesced write）
// 和 tiled_transpose.cu 一样的思路，在这里独立再实现一遍
// ============================================================
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

// ---- host wrappers（不需要修改）----

torch::Tensor naiveTranspose(torch::Tensor A) {
    int m = A.size(0), n = A.size(1);
    auto B = torch::zeros({n, m}, A.options());
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((n + TILE_WIDTH - 1) / TILE_WIDTH,
              (m + TILE_WIDTH - 1) / TILE_WIDTH);
    NaiveTransposeKernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), m, n);
    return B;
}

torch::Tensor tiledTransposeBench(torch::Tensor A) {
    int m = A.size(0), n = A.size(1);
    auto B = torch::zeros({n, m}, A.options());
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((n + TILE_WIDTH - 1) / TILE_WIDTH,
              (m + TILE_WIDTH - 1) / TILE_WIDTH);
    TiledTransposeKernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), m, n);
    return B;
}
