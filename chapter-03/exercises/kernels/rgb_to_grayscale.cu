// 附加题: RGB 转灰度内核
// 输入 Pin: height x width x 3 的 unsigned char 数组 (RGB)
// 输出 Pout: height x width x 1 的 unsigned char 数组 (灰度)
//
// 灰度公式: gray = 0.21 * R + 0.71 * G + 0.07 * B
//
// 提示：
// - 用 2D 线程网格，blockIdx.x/threadIdx.x 对应列，blockIdx.y/threadIdx.y 对应行
// - RGB 数据是交错存储的：Pin[offset*3+0]=R, Pin[offset*3+1]=G, Pin[offset*3+2]=B

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

__global__ void rgbToGrayscaleKernel(unsigned char* Pin, unsigned char* Pout, int width, int height) {
    // TODO: 计算 col 和 row
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    // int base = width * height;
    // TODO: 边界检查，计算灰度值并写入 Pout
    if (col < width && row < height) {
        // float sum = 0;
        // for (int channel = 0; channel < 3; channel++) {
        //     sum += Pin[base * channel + row * width + col];
        // }
        int offset = (row * width + col) * 3;
        float R = Pin[offset];
        float G = Pin[offset + 1]; 
        float B = Pin[offset + 2];
        Pout[row * width + col] = 0.21 * R + 0.71 * G + 0.07 * B;
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor rgb_to_gray(torch::Tensor img) {
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);

    const auto height = img.size(0);
    const auto width = img.size(1);

    dim3 dimBlock(32, 32);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    rgbToGrayscaleKernel<<<dimGrid, dimBlock, 0, torch::cuda::getCurrentCUDAStream()>>>(
        img.data_ptr<unsigned char>(), result.data_ptr<unsigned char>(), width, height);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
