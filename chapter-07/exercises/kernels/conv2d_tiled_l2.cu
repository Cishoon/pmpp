#include <torch/extension.h>

// 分块 2D 卷积 + L2 缓存利用
//
// 与标准分块版（conv2d_tiled.cu）的区别：
//   标准分块版：每个块的 IN_TILE 比 OUT_TILE 大，边缘线程加载 halo 数据
//   L2 缓存版：每个块只加载 TILE_SIZE × TILE_SIZE（等于输出大小），
//              卷积时如果需要 halo 数据，直接从全局内存读取，
//              依赖相邻块已经把这些数据带入 L2 缓存。
//
// 好处：每个块的线程数更少（TILE²），共享内存用量更小
// 代价：halo 部分仍需访问全局内存（但通常已在 L2 中）
//
// 提示：
//   - 线程坐标直接对应输出坐标（不需要减 FILTER_RADIUS）
//       row = blockIdx.y * TILE_SIZE + threadIdx.y
//       col = blockIdx.x * TILE_SIZE + threadIdx.x
//   - 先加载 TILE_SIZE × TILE_SIZE 的输入块到共享内存
//   - 卷积时：
//       sy = threadIdx.y - FILTER_RADIUS + fRow
//       sx = threadIdx.x - FILTER_RADIUS + fCol
//       如果 sy/sx 在 [0, TILE_SIZE) 内 → 从共享内存读
//       否则 → 从全局内存读（注意边界检查）

#define FILTER_RADIUS 2
#define TILE_SIZE 16

__constant__ float constFilter[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];

__global__ void Conv2dTiledL2Kernel(const float* N, float* P, int height, int width) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    __shared__ float N_s[TILE_SIZE][TILE_SIZE];
    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    const int &r = FILTER_RADIUS;
    float p_value = 0.0f;
    for (int i = -r; i <= r; i++) {
        for (int j = -r; j <= r; j++) {
            int c_row = i + r;
            int c_col = j + r;
            
            int n_row = threadIdx.y + i;
            int n_col = threadIdx.x + j;
            
            if (n_row >= 0 && n_row < TILE_SIZE && n_col >= 0 && n_col < TILE_SIZE)
                p_value += N_s[n_row][n_col] * constFilter[c_row * (2 * r + 1) + c_col];
            else {
                int n_row = row + i;
                int n_col = col + j;
                if (n_row >= 0 && n_row < height && n_col >= 0 && n_col < width)
                    p_value += N[n_row * width + n_col] * constFilter[c_row * (2 * r + 1) + c_col];
            } 
        }
    }
    if (row < height && col < width)
        P[row * width + col] = p_value;
}

torch::Tensor conv2dTiledL2(torch::Tensor N, torch::Tensor F_kernel) {
    int height = N.size(0), width = N.size(1);
    cudaMemcpyToSymbol(constFilter, F_kernel.data_ptr<float>(),
                       F_kernel.numel() * sizeof(float));

    auto P = torch::zeros({height, width}, N.options());
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width  + TILE_SIZE - 1) / TILE_SIZE,
              (height + TILE_SIZE - 1) / TILE_SIZE);

    Conv2dTiledL2Kernel<<<grid, block>>>(
        N.data_ptr<float>(), P.data_ptr<float>(), height, width);

    return P;
}
