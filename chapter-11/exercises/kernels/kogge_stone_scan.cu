#include <torch/extension.h>

// Kogge-Stone 并行前缀扫描（inclusive scan）
//
// 输入 input: 1D 浮点张量（长度 <= 1024，单 block 处理）
// 输出: 同长度张量，每个位置存储前缀和
//
// 思路：
//   - 将输入加载到 extern shared memory（超出 N 的位置填 0）
//   - stride 从 1 开始，每次翻倍，直到 >= blockDim.x
//   - 每次迭代中，threadIdx.x >= stride 的线程将当前值与 stride 位置前的值相加
//   - 需要两次 __syncthreads() 防止 write-after-read 竞争
//   - 最后将结果写回输出
//
// host 端需要：
//   - 创建输出张量（与输入同形状）
//   - 配置 grid(1), block(N)，动态共享内存大小 N*sizeof(float)
//   - 启动内核并返回结果

#define BLOCK_SIZE 1024

__global__ void KoggeStoneInclusiveScanKernel(float* X, float* Y, unsigned int N) {
    int i = threadIdx.x;
    
    if (i < N) Y[i] = X[i];
    __syncthreads();
    
    float tmp;
    for (int stride = 1; stride <= N / 2; stride *= 2) {
        if (i + stride < N) tmp = Y[i] + Y[i + stride];
        __syncthreads();
        if (i + stride < N) Y[i + stride] = tmp;
        __syncthreads();
    }
}

// 函数签名: torch::Tensor kogge_stone_scan(torch::Tensor input)
torch::Tensor kogge_stone_scan(torch::Tensor input) {
    int length = input.size(0);
    auto output = torch::empty_like(input);
    
    dim3 block(BLOCK_SIZE);
    dim3 grid(1);
    
    KoggeStoneInclusiveScanKernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), length
    );
    
    return output;
}