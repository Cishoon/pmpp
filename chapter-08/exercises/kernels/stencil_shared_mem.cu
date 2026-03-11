#include <torch/extension.h>

// 共享内存 3D 七点模板计算
//
// 与朴素版的区别：
//   将输入分块（含 halo）加载到共享内存，减少全局内存访问
//
// 关键尺寸关系：
//   IN_TILE_DIM = OUT_TILE_DIM + 2（每个方向各 1 层 halo）
//   每个块有 IN_TILE_DIM³ 个线程
//   但只有中间 OUT_TILE_DIM³ 个线程写输出
//
// 提示：
//   - 线程的全局坐标（含 halo 偏移）：
//       i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1
//       j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1
//       k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1
//   - 先协作加载输入分块到 in_s[threadIdx.z][threadIdx.y][threadIdx.x]
//   - __syncthreads()
//   - 只有 threadIdx 在 [1, IN_TILE_DIM-2] 范围内的线程才计算输出
//   - 同时检查全局坐标在 [1, N-2] 范围内

#define OUT_TILE_DIM 8
#define IN_TILE_DIM (OUT_TILE_DIM + 2)

__global__ void Stencil3dSharedMemKernel(const float* in, float* out, int N,
                                          float c0, float c1, float c2,
                                          float c3, float c4, float c5, float c6) {
    // TODO: 计算全局坐标 (i, j, k)，注意 halo 偏移
    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    // TODO: 声明共享内存 in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM]
    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    // TODO: 协作加载数据到共享内存（边界外补 0）
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
    } else {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    // TODO: 内部线程计算七点模板并写入输出
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        if (threadIdx.z > 0 && threadIdx.z < IN_TILE_DIM - 1 && threadIdx.y > 0 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x > 0 && threadIdx.x < IN_TILE_DIM - 1) {
            out[i * N * N + j * N + k] = c0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x]
                + c1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] + c2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1]
                + c3 * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] + c4 * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x]
                + c5 * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] + c6 * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
        }
    }
}

torch::Tensor stencil3dSharedMem(torch::Tensor in, int N,
                                  float c0, float c1, float c2,
                                  float c3, float c4, float c5, float c6) {
    auto out = torch::zeros_like(in);

    dim3 block(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    dim3 grid((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
              (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
              (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    Stencil3dSharedMemKernel<<<grid, block>>>(
        in.data_ptr<float>(), out.data_ptr<float>(), N,
        c0, c1, c2, c3, c4, c5, c6);

    return out;
}
