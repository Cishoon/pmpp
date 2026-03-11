#include <torch/extension.h>

// 分块矩阵乘法内核
// 计算 P = M * N，其中 M 是 (m x n)，N 是 (n x o)，P 是 (m x o)
// 使用共享内存分块（tiling）来减少全局内存访问
//
// 提示：
//   - 使用 __shared__ 声明两个分块缓冲区 Mds 和 Nds，大小为 TILE_WIDTH x TILE_WIDTH
//   - 外层循环遍历所有 phase（分块阶段）
//   - 每个 phase 中：先协作加载数据到共享内存，__syncthreads()，再计算点积，__syncthreads()
//   - 注意边界检查（矩阵尺寸不一定是 TILE_WIDTH 的整数倍）

#define TILE_WIDTH 16

__global__ void TiledMatMulKernel(const float* M, const float* N, float* P,
                                   int m, int n, int o) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    float PValue = 0.0f;
    for (int ph = 0; ph < (n + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        int my = y;
        int mx = ph * TILE_WIDTH + threadIdx.x;
        if (mx < n && my < m) {
            Mds[threadIdx.y][threadIdx.x] = M[my * n + mx];
        } else {
            Mds[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        int ny = ph * TILE_WIDTH + threadIdx.y;
        int nx = x;
        if (nx < o && ny < n) {
            Nds[threadIdx.y][threadIdx.x] = N[ny * o + nx];
        } else {
            Nds[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; k++) {
            PValue += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (x < o && y < m) {
        P[y * o + x] = PValue;
    }
}

torch::Tensor tiledMatMul(torch::Tensor M, torch::Tensor N) {
    int m = M.size(0), n = M.size(1), o = N.size(1);
    auto P = torch::zeros({m, o}, M.options());

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((o + TILE_WIDTH - 1) / TILE_WIDTH,
              (m + TILE_WIDTH - 1) / TILE_WIDTH);

    TiledMatMulKernel<<<grid, block>>>(
        M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(),
        m, n, o);

    return P;
}
