// 附加题: 高斯模糊（均值模糊）内核
// 输入 Pin: channels x height x width 的 unsigned char 数组
// 输出 Pout: 同尺寸，每个像素取周围 (2*blur_size+1)^2 邻域的均值
//
// 提示：
// - 用 3D 线程块：threadIdx.x 对应列，threadIdx.y 对应行，threadIdx.z 对应通道
// - 对每个像素，遍历 [-blur_size, blur_size] 的邻域，注意边界检查
// - 每个通道独立处理，通道偏移 = channel * height * width

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

__global__ void blur_kernel(unsigned char* Pin, unsigned char* Pout, int width, int height, int blur_size) {
    // TODO: 计算 col, row, channel 和 baseOffset
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int channel = blockDim.z * blockIdx.z + threadIdx.z;
    int baseOffset = height * width * channel;
    // TODO: 边界检查，遍历邻域计算均值，写入 Pout
    if (row < height && col < width) {
        float sum = 0;
        int count = 0;
        for (int i = -blur_size; i <= blur_size; i++) {
            for (int j = -blur_size; j <= blur_size; j++) {
                int dx = i + row; 
                int dy = j + col;
                if (dx >= 0 && dx < height && dy >= 0 && dy < width) {
                    sum += Pin[baseOffset + dx * width + dy];    
                    count++;
                }
            }
        }
        sum /= count;
        Pout[baseOffset + row * width + col] = sum;
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor gaussian_blur(torch::Tensor img, int blurSize) {
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);

    const auto channels = img.size(0);
    const auto height = img.size(1);
    const auto width = img.size(2);

    dim3 dimBlock(16, 16, channels);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    auto result = torch::empty_like(img);

    blur_kernel<<<dimGrid, dimBlock, 0, torch::cuda::getCurrentCUDAStream()>>>(
        img.data_ptr<unsigned char>(), result.data_ptr<unsigned char>(), width, height, blurSize);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
