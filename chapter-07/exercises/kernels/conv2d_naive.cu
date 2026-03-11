#include <torch/extension.h>

// 朴素 2D 卷积
// 输入 N: (height, width)，滤波器 F: (2r+1, 2r+1)，输出 P: (height, width)
//
// 每个线程负责一个输出元素 P[outRow][outCol]
// 对滤波器的每个位置 (fRow, fCol)，计算对应输入位置并累加
// 边界外的输入视为 0（ghost cell）
//
// 提示：
//   - outRow = blockIdx.y * blockDim.y + threadIdx.y
//   - outCol = blockIdx.x * blockDim.x + threadIdx.x
//   - 输入位置：inRow = outRow - r + fRow，inCol = outCol - r + fCol
//   - 注意边界检查

#define BLOCK_SIZE 16

__global__ void Conv2dNaiveKernel(const float* N, const float* F, float* P,
                                   int r, int height, int width) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int baseOffset = row * width + col;
    float Pvalue = 0.0f;
    for (int i = -r; i <= r; i++) {
        for (int j = -r; j <= r; j++) {
            if (row + i < height && col + j < width && row + i >= 0 && col + j >= 0) 
                Pvalue += N[baseOffset + i * width + j] * F[(i + r) * (2 * r + 1) + (j + r)];
        }
    }
    if (row < height && col < width)
        P[row * width + col] = Pvalue;
}

torch::Tensor conv2dNaive(torch::Tensor N, torch::Tensor F_kernel) {
    int height = N.size(0), width = N.size(1);
    int filter_size = F_kernel.size(0);
    int r = filter_size / 2;
    auto P = torch::zeros({height, width}, N.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width  + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    Conv2dNaiveKernel<<<grid, block>>>(
        N.data_ptr<float>(), F_kernel.data_ptr<float>(), P.data_ptr<float>(),
        r, height, width);

    return P;
}
