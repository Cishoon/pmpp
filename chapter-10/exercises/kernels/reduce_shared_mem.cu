#include <torch/extension.h>

// 共享内存归约内核（Fig. 10.11）
//
// 与收敛型的区别：
//   使用共享内存存储中间结果，避免全局内存的反复读写
//   第一步就将两个元素加载到共享内存并求和
//
// 思路：
//   - 分配共享内存 input_s[BLOCK_DIM]
//   - t = threadIdx.x
//   - 初始加载：input_s[t] = input[t] + input[t + BLOCK_DIM]
//   - stride 从 blockDim.x/2 开始（因为第一次迭代已在加载时完成）
//   - 每次迭代前 __syncthreads()
//   - 条件 t < stride 时，input_s[t] += input_s[t + stride]
//   - 线程 0 将 input_s[0] 写入 output
//
// 限制：仅支持 2*BLOCK_DIM 个元素（单 block）
//
// host 端需要：
//   - 不需要克隆输入（使用共享内存，不修改全局输入）
//   - grid(1), block(BLOCK_DIM)

#define BLOCK_DIM 1024

// TODO: 实现 SharedMemReductionKernel
__global__ void SharedMemReductionKernel(float* input, float* output) {
    int i = threadIdx.x;
    
    __shared__ float input_s[BLOCK_DIM];
    input_s[i] = input[i] + input[i + BLOCK_DIM];
    __syncthreads();
    
    for (int stride = BLOCK_DIM / 2; stride >= 1; stride >>= 1) {
        if (i < stride) input_s[i] += input_s[i + stride];
        __syncthreads();
    }
    
    if (i == 0) {
        output[0] = input_s[0];
    }
}

// TODO: 实现 host 函数
// 函数签名: torch::Tensor reduceSharedMem(torch::Tensor input)
torch::Tensor reduceSharedMem(torch::Tensor input) {
    int length = input.size(0);
    auto output = torch::zeros({1}, input.options().dtype(torch::kFloat32));
    
    dim3 block(BLOCK_DIM);
    dim3 grid(1);
    
    SharedMemReductionKernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>()
    );
    return output;
}
