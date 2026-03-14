#include <torch/extension.h>

// 朴素归约内核（Fig. 10.6 - 发散型）
//
// 输入 input: 1D 浮点张量
// 输出: 标量，所有元素之和
//
// 思路：
//   - 每个线程映射到 i = 2 * threadIdx.x
//   - stride 从 1 开始，每次迭代翻倍
//   - 条件 threadIdx.x % stride == 0 时，执行 input[i] += input[i + stride]
//   - 每次迭代后需要 __syncthreads()
//   - 线程 0 将 input[0] 写入 output
//
// 限制：仅支持 2*blockDim.x 个元素（单 block）
//
// host 端需要：
//   - 创建输出张量（标量）
//   - 克隆输入（因为内核会原地修改）
//   - 配置 grid(1), block(length/2)
//   - 启动内核并返回结果

#define BLOCK_DIM 1024

// TODO: 实现 SimpleReductionKernel
__global__ void SimpleReductionKernel(float* input, float* output, int length) {
    int i = 2 * threadIdx.x;
    
    for (int stride = 1; stride <= BLOCK_DIM; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        output[0] = input[0];
    }
}

// TODO: 实现 host 函数
// 函数签名: torch::Tensor reduceSimple(torch::Tensor input)
torch::Tensor reduceSimple(torch::Tensor input) {
    int length = input.size(0);
    auto input_clone = input.clone();
    auto output = torch::zeros({1}, input.options().dtype(torch::kFloat32));
    
    dim3 block(length / 2);
    dim3 grid(1);
    
    SimpleReductionKernel<<<grid, block>>>(
        input_clone.data_ptr<float>(), output.data_ptr<float>(), length
    );
    return output;
}
