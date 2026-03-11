#include <torch/extension.h>

// 收敛型归约内核（Fig. 10.9）
//
// 与朴素版的区别：
//   stride 从 blockDim.x 开始，每次迭代减半
//   条件改为 threadIdx.x < stride
//   这样活跃线程集中在低编号 warp，减少控制分歧
//
// 思路：
//   - i = threadIdx.x
//   - stride 从 blockDim.x 开始，每次除以 2
//   - 条件 threadIdx.x < stride 时，执行 input[i] += input[i + stride]
//   - 每次迭代后 __syncthreads()
//   - 线程 0 将 input[0] 写入 output
//
// 限制：仅支持 2*blockDim.x 个元素（单 block）
//
// host 端需要：
//   - 克隆输入（内核原地修改）
//   - grid(1), block(length/2)

#define BLOCK_DIM 1024

// TODO: 实现 ConvergentReductionKernel
__global__ void ConvergentReductionKernel(float* input, float* output) {
    int i = threadIdx.x;
    
    for (int stride = BLOCK_DIM; stride >= 1; stride /= 2) {
        if (i < stride) input[i] += input[i + stride];
        __syncthreads();
    }
    if (i == 0)
        output[0] = input[0];
}

// TODO: 实现 host 函数
// 函数签名: torch::Tensor reduceConvergent(torch::Tensor input)
torch::Tensor reduceConvergent(torch::Tensor input) {
    auto input_clone = input.clone();
    int length = input.size(0);
    auto output = torch::zeros({1}, input.options().dtype(torch::kFloat32));
    
    dim3 block(length / 2);
    dim3 grid(1);
    
    ConvergentReductionKernel<<<grid, block>>>(
        input_clone.data_ptr<float>(), output.data_ptr<float>()
    );
    
    return output;
}
