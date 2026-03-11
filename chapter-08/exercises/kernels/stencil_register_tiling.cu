#include <torch/extension.h>

// 寄存器分块 3D 七点模板计算
//
// 与线程粗化版的区别：
//   将 inPrev 和 inNext 存储在寄存器中而非共享内存，
//   只保留一个 2D 共享内存数组 inCurr_s 用于 x-y 平面的邻居访问。
//   这大幅减少了共享内存用量（从 3 个平面降到 1 个）。
//
// 关键思路：
//   - inPrev（寄存器）：当前线程在 z-1 平面的值
//   - inCurr_s（共享内存）：当前 z 平面的完整分块（需要 x-y 邻居）
//   - inCurr（寄存器）：当前线程在 z 平面的值（用于 c0 系数）
//   - inNext（寄存器）：当前线程在 z+1 平面的值
//
// 提示：
//   - 初始化：加载 iStart-1 到 inPrev，iStart 到 inCurr 和 inCurr_s
//   - 循环中：
//       加载 i+1 到 inNext → __syncthreads() → 计算（用寄存器和共享内存）
//       → __syncthreads() → 滑动：inPrev=inCurr, inCurr=inNext, inCurr_s=inNext

#define OUT_TILE_DIM 30
#define IN_TILE_DIM (OUT_TILE_DIM + 2)

__global__ void Stencil3dRegisterTilingKernel(const float* in, float* out, int N,
                                               float c0, float c1, float c2,
                                               float c3, float c4, float c5, float c6) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    // 寄存器变量——就是普通的局部变量，编译器会自动分配到寄存器
    float inPrev = 0.0f;
    float inCurr = 0.0f;
    float inNext = 0.0f;

    // 共享内存只保留一个平面（用于读 x-y 方向的邻居）
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];

    // 初始化：加载 iStart-1 到寄存器 inPrev
    if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev = in[(iStart - 1) * N * N + j * N + k];
    }

    // 加载 iStart 到寄存器 inCurr 和共享内存 inCurr_s
    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr = in[iStart * N * N + j * N + k];
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    }

    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        // 加载 i+1 到寄存器 inNext
        inNext = 0.0f;
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();

        // 计算：z 方向邻居从寄存器读，x-y 方向邻居从共享内存读
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1 &&
            threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
            threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
            out[i * N * N + j * N + k] = c0 * inCurr
                + c1 * inCurr_s[threadIdx.y][threadIdx.x - 1]
                + c2 * inCurr_s[threadIdx.y][threadIdx.x + 1]
                + c3 * inCurr_s[threadIdx.y - 1][threadIdx.x]
                + c4 * inCurr_s[threadIdx.y + 1][threadIdx.x]
                + c5 * inPrev
                + c6 * inNext;
        }
        __syncthreads();

        // 滑动：寄存器直接赋值，共享内存用 inNext 更新
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;
    }
}

torch::Tensor stencil3dRegisterTiling(torch::Tensor in, int N,
                                       float c0, float c1, float c2,
                                       float c3, float c4, float c5, float c6) {
    auto out = torch::zeros_like(in);

    dim3 block(IN_TILE_DIM, IN_TILE_DIM, 1);
    dim3 grid((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
              (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
              (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    Stencil3dRegisterTilingKernel<<<grid, block>>>(
        in.data_ptr<float>(), out.data_ptr<float>(), N,
        c0, c1, c2, c3, c4, c5, c6);

    return out;
}
