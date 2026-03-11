#include <torch/extension.h>

// 线程粗化 3D 七点模板计算
//
// 与共享内存版的区别：
//   使用 2D 线程块（IN_TILE_DIM × IN_TILE_DIM），每个块沿 z 轴处理
//   OUT_TILE_DIM 个连续输出平面。使用三个 2D 共享内存数组
//   （prev、curr、next）滑动窗口式地处理。
//
// 关键思路：
//   - 2D 线程块处理 x-y 平面（含 halo）
//   - 沿 z 轴循环，每次迭代处理一个输出平面
//   - 三个共享内存平面：inPrev_s、inCurr_s、inNext_s
//   - 每次迭代结束后滑动：prev←curr, curr←next
//
// 提示：
//   - iStart = blockIdx.z * OUT_TILE_DIM
//   - j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1
//   - k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1
//   - 初始化：加载 iStart-1 平面到 inPrev_s，iStart 平面到 inCurr_s
//   - 循环 i 从 iStart 到 iStart+OUT_TILE_DIM-1：
//       加载 i+1 平面到 inNext_s → __syncthreads() → 计算 → __syncthreads() → 滑动

#define OUT_TILE_DIM 30
#define IN_TILE_DIM (OUT_TILE_DIM + 2)

__global__ void Stencil3dCoarseningKernel(const float* in, float* out, int N,
                                           float c0, float c1, float c2,
                                           float c3, float c4, float c5, float c6) {
    // TODO: 计算 iStart, j, k
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    // TODO: 声明三个共享内存平面 inPrev_s, inCurr_s, inNext_s
    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];
    
    inPrev_s[threadIdx.y][threadIdx.x] = 0.0f;
    inCurr_s[threadIdx.y][threadIdx.x] = 0.0f;

    if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1) * N * N + j * N + k];
    }
    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart * N * N + j * N + k];
    }

    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        // 加载 next 平面
        inNext_s[threadIdx.y][threadIdx.x] = 0.0f;
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();

        // 计算
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1 &&
            threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
            threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
            out[i * N * N + j * N + k] = c0 * inCurr_s[threadIdx.y][threadIdx.x]
                + c1 * inCurr_s[threadIdx.y][threadIdx.x - 1] + c2 * inCurr_s[threadIdx.y][threadIdx.x + 1]
                + c3 * inCurr_s[threadIdx.y - 1][threadIdx.x] + c4 * inCurr_s[threadIdx.y + 1][threadIdx.x]
                + c5 * inPrev_s[threadIdx.y][threadIdx.x] + c6 * inNext_s[threadIdx.y][threadIdx.x];
        }
        __syncthreads();

        // 滑动
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
}

torch::Tensor stencil3dCoarsening(torch::Tensor in, int N,
                                   float c0, float c1, float c2,
                                   float c3, float c4, float c5, float c6) {
    auto out = torch::zeros_like(in);

    dim3 block(IN_TILE_DIM, IN_TILE_DIM, 1);
    dim3 grid((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
              (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
              (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    Stencil3dCoarseningKernel<<<grid, block>>>(
        in.data_ptr<float>(), out.data_ptr<float>(), N,
        c0, c1, c2, c3, c4, c5, c6);

    return out;
}
