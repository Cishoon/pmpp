#include <torch/extension.h>

// 习题 1：列主序 N 的分块矩阵乘法（Corner Turning）
// 计算 P = M * N_T，其中 N 以列主序存储（即传入的是 N 的转置，形状为 o×n）
//
// 背景：
//   标准分块乘法中 N 以行主序存储，加载 N 的分块时访问模式是 coalesced 的。
//   但如果 N 以列主序存储（等价于传入转置后的 N），需要调整索引才能保持 coalesced 访问。
//   这种技巧叫 Corner Turning：coalesced 地把数据加载进共享内存，再从共享内存中正确读取。
//
// 提示：
//   - M 的加载方式与标准分块乘法完全相同
//   - N_T 的形状是 (o, n)，加载第 ph 个分块时：
//     行索引是 col（P 的列），列索引是 ph * TILE_WIDTH + threadIdx.y
//     即：N_T[col * n + ph * TILE_WIDTH + threadIdx.y]
//   - Nds 加载完后，点积计算方式与标准分块乘法相同

#define TILE_WIDTH 16

__global__ void ColMajorMatMulKernel(const float* M, const float* N_T, float* P,
                                      int m, int n, int o) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float PValue = 0.0f;
    for (int ph = 0; ph < (n + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {

        // TODO: 加载 M 的分块（与标准分块乘法相同）
        // if (row < m && (ph * TILE_WIDTH + threadIdx.x) < n)
        //     Mds[threadIdx.y][threadIdx.x] = M[...];
        // else
        //     Mds[threadIdx.y][threadIdx.x] = 0.0f;
        if (row < m && ph * TILE_WIDTH + threadIdx.x < n) {
            Mds[threadIdx.y][threadIdx.x] = M[row * n + ph * TILE_WIDTH + threadIdx.x];
        } else {
            Mds[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // TODO: 加载 N_T 的分块（N_T 形状为 o×n，coalesced 地按列读取）
        // 提示：N_T[col * n + ph * TILE_WIDTH + threadIdx.y]
        // if ((ph * TILE_WIDTH + threadIdx.y) < n && col < o)
        //     Nds[threadIdx.y][threadIdx.x] = N_T[...];
        // else
        //     Nds[threadIdx.y][threadIdx.x] = 0.0f;

        if (ph * TILE_WIDTH + threadIdx.y < n && col < o) {
            Nds[threadIdx.y][threadIdx.x] = N_T[col * n + ph * TILE_WIDTH + threadIdx.y];
        } else {
            Nds[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // TODO: 点积累加（与标准分块乘法相同）
        for (int k = 0; k < TILE_WIDTH; k++)
            PValue += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];

        __syncthreads();
    }

    // TODO: 写回结果（注意边界检查）
    if (row < m && col < o) P[row * o + col] = PValue;
}

torch::Tensor colMajorMatMul(torch::Tensor M, torch::Tensor N_T) {
    // N_T 是 N 的转置，形状为 (o, n)
    int m = M.size(0), n = M.size(1), o = N_T.size(0);
    auto P = torch::zeros({m, o}, M.options());

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((o + TILE_WIDTH - 1) / TILE_WIDTH,
              (m + TILE_WIDTH - 1) / TILE_WIDTH);

    ColMajorMatMulKernel<<<grid, block>>>(
        M.data_ptr<float>(), N_T.data_ptr<float>(), P.data_ptr<float>(),
        m, n, o);

    return P;
}
