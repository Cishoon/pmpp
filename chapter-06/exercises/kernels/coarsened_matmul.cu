#include <torch/extension.h>

// 线程粗化分块矩阵乘法
// 每个线程负责计算 P 中 COARSE_FACTOR 个连续列的元素
//
// 核心思路：
//   标准分块乘法中，每个线程块只计算 P 的一个 TILE×TILE 块。
//   线程粗化后，每个线程块负责 P 的 TILE×(TILE*COARSE_FACTOR) 的区域，
//   每个线程负责其中一行的 COARSE_FACTOR 个元素。
//
// 好处：
//   M 的分块只需加载一次，就能被 COARSE_FACTOR 个不同的 N 分块复用，
//   减少了 M 的全局内存加载次数。
//
// 提示：
//   - 用数组 float PValue[COARSE_FACTOR] 保存每个线程负责的多个输出值
//   - colStart = blockIdx.x * TILE_WIDTH * COARSE_FACTOR + threadIdx.x
//   - 每个 coarse 步骤 c 对应的列：col = colStart + c * TILE_WIDTH
//   - 外层 phase 循环内，M 的分块只加载一次；N 的分块对每个 c 分别加载
//   - 注意 __syncthreads() 的位置

#define TILE_WIDTH 16
#define COARSE_FACTOR 4

__global__ void CoarsenedMatMulKernel(const float* M, const float* N, float* P,
                                       int m, int n, int o) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int colStart = blockIdx.x * TILE_WIDTH * COARSE_FACTOR + threadIdx.x;

    // TODO: 声明 PValue 数组，初始化为 0
    float PValue[COARSE_FACTOR] = {0.0f};

    for (int ph = 0; ph < (n + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {

        // TODO: 加载 M 的分块（每个 phase 只加载一次，被所有 c 复用）
        int M_col = ph * TILE_WIDTH + threadIdx.x;
        int M_row = row;
        if (M_row < m && M_col < n) 
            Mds[threadIdx.y][threadIdx.x] = M[M_row * n + M_col];
        else 
            Mds[threadIdx.y][threadIdx.x] = 0.0f;

        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = colStart + c * TILE_WIDTH;

            // TODO: 加载 N 的第 c 个分块
            int N_row = ph * TILE_WIDTH + threadIdx.y;
            int N_col = col;
            if (N_col < o && N_row < n) 
                Nds[threadIdx.y][threadIdx.x] = N[N_row * o + N_col];
            else
                Nds[threadIdx.y][threadIdx.x] = 0.0f;
            
            // TODO: __syncthreads()
            __syncthreads();

            // TODO: 点积累加到 PValue[c]
            for (int k = 0; k < TILE_WIDTH; k++) {
                PValue[c] += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
            }

            __syncthreads();
        }
    }

    // TODO: 将 PValue[c] 写回 P（注意边界检查）
    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = colStart + c * TILE_WIDTH;
        if (row < m && col < o) P[row * o + col] = PValue[c];
    }
}

torch::Tensor coarsenedMatMul(torch::Tensor M, torch::Tensor N) {
    int m = M.size(0), n = M.size(1), o = N.size(1);
    auto P = torch::zeros({m, o}, M.options());

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((o + TILE_WIDTH * COARSE_FACTOR - 1) / (TILE_WIDTH * COARSE_FACTOR),
              (m + TILE_WIDTH - 1) / TILE_WIDTH);

    CoarsenedMatMulKernel<<<grid, block>>>(
        M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(),
        m, n, o);

    return P;
}
