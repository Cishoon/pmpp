#include <torch/extension.h>

// 使用常量内存的朴素 2D 卷积
// 与 conv2d_naive.cu 的区别：滤波器存在 __constant__ 内存中
//
// 为什么用常量内存？
//   所有线程使用相同的滤波器，常量内存有专用缓存，
//   广播读取时比全局内存快得多。
//
// 提示：
//   - 用 cudaMemcpyToSymbol 把滤波器拷贝到 constFilter（host wrapper 里已完成）
//   - kernel 里直接用 constFilter[fRow*(2*r+1)+fCol] 读取，不需要传 F 指针
//   - 其余逻辑与朴素版完全相同

#define FILTER_RADIUS 3
#define BLOCK_SIZE 16

__constant__ float constFilter[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];

__global__ void Conv2dConstMemKernel(const float* N, float* P, int r, int height, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float p_value = 0.0f;
    for (int i = -r; i <= r; i++) {
        for (int j = -r; j <= r; j++) {
            int n_row = row + i, n_col = col + j;
            int f_row = i + r, f_col = j + r;
            if (n_row >= 0 && n_row < height && n_col >= 0 && n_col < width)
                p_value += N[n_row * width + n_col] * constFilter[f_row * (2 * r + 1) + f_col];
        }
    }
    if (row < height && col < width)
        P[row * width + col] = p_value;
}

torch::Tensor conv2dConstMem(torch::Tensor N, torch::Tensor F_kernel) {
    int height = N.size(0), width = N.size(1);
    int r = F_kernel.size(0) / 2;
    // 拷贝滤波器到常量内存
    cudaMemcpyToSymbol(constFilter, F_kernel.data_ptr<float>(),
                       F_kernel.numel() * sizeof(float));

    auto P = torch::zeros({height, width}, N.options());
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width  + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    Conv2dConstMemKernel<<<grid, block>>>(
        N.data_ptr<float>(), P.data_ptr<float>(), r, height, width);

    return P;
}
