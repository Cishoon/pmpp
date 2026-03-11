#include <torch/extension.h>

// 分块 2D 卷积（使用常量内存存储滤波器）
//
// 与朴素版的区别：
//   1. 滤波器存在常量内存（__constant__），所有线程广播读取，无需重复从全局内存加载
//   2. 输入分块加载到共享内存，减少全局内存访问
//
// 关键尺寸关系：
//   IN_TILE_SIZE  = OUT_TILE_SIZE + 2 * FILTER_RADIUS
//   每个块有 IN_TILE_SIZE × IN_TILE_SIZE 个线程
//   但只有中间 OUT_TILE_SIZE × OUT_TILE_SIZE 个线程写输出
//
// 提示：
//   - 线程的全局坐标（含 halo）：
//       row = blockIdx.y * OUT_TILE_SIZE + threadIdx.y - FILTER_RADIUS
//       col = blockIdx.x * OUT_TILE_SIZE + threadIdx.x - FILTER_RADIUS
//   - 先协作加载输入分块到 N_s（含边界外补 0）
//   - __syncthreads()
//   - 只有 tileRow/tileCol 在 [0, OUT_TILE_SIZE) 内的线程才计算输出
//   - 卷积时直接用 N_s[threadIdx.y + fRow - r][threadIdx.x + fCol - r]

#define FILTER_RADIUS 2
#define OUT_TILE_SIZE 8
#define IN_TILE_SIZE (OUT_TILE_SIZE + 2 * FILTER_RADIUS)

__constant__ float constFilter[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];

__global__ void Conv2dTiledKernel(const float* N, float* P, int height, int width) {
    const int& r = FILTER_RADIUS;
    
    int row = blockIdx.y * OUT_TILE_SIZE + threadIdx.y - r;
    int col = blockIdx.x * OUT_TILE_SIZE + threadIdx.x - r;
    __shared__ float N_s[IN_TILE_SIZE][IN_TILE_SIZE];
    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    __syncthreads();
    
    int tileRow = threadIdx.y - r;
    int tileCol = threadIdx.x - r;
    if (row >= 0 && row < height && col >= 0 && col < width) {
        if (tileRow >= 0 && tileRow < OUT_TILE_SIZE && tileCol >= 0 && tileCol < OUT_TILE_SIZE) {
            float PValue = 0.0f;
            for (int i = 0; i < 2 * r + 1; i++) {
                for (int j = 0; j < 2 * r + 1; j++) {
                    int fIndex = i * (2 * r + 1) + j;
                    PValue += constFilter[fIndex] * N_s[tileRow + i][tileCol + j];
                }
            }        
            
            P[row * width + col] = PValue;
        }   
    }

}

torch::Tensor conv2dTiled(torch::Tensor N, torch::Tensor F_kernel) {
    int height = N.size(0), width = N.size(1);
    // 将滤波器拷贝到常量内存
    cudaMemcpyToSymbol(constFilter, F_kernel.data_ptr<float>(),
                       F_kernel.numel() * sizeof(float));

    auto P = torch::zeros({height, width}, N.options());

    dim3 block(IN_TILE_SIZE, IN_TILE_SIZE);
    dim3 grid((width  + OUT_TILE_SIZE - 1) / OUT_TILE_SIZE,
              (height + OUT_TILE_SIZE - 1) / OUT_TILE_SIZE);

    Conv2dTiledKernel<<<grid, block>>>(
        N.data_ptr<float>(), P.data_ptr<float>(), height, width);

    return P;
}
