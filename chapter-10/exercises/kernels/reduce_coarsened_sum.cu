#include <torch/extension.h>

// 线程粗化求和归约内核（Fig. 10.15 - 支持任意长度）
//
// 与共享内存版的区别：
//   每个线程先将 COARSE_FACTOR*2 个元素累加，再进行树形归约
//   支持多 block，通过 atomicAdd 汇总各 block 结果
//   支持任意长度输入（需要边界检查）
//
// 思路：
//   - segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x
//   - i = segment + threadIdx.x, t = threadIdx.x
//   - 先检查 i < length，将 input[i] 赋给 sum
//   - 粗化循环：for tile in [1, COARSE_FACTOR*2)，
//     若 i + tile*BLOCK_DIM < length 则累加
//   - 将 sum 存入 input_s[t]
//   - 树形归约：stride 从 blockDim.x/2 开始
//   - 线程 0 用 atomicAdd 将 input_s[0] 加到 output
//
// host 端需要：
//   - 计算 elementsPerBlock = 2 * COARSE_FACTOR * BLOCK_DIM
//   - grid = ceil(length / elementsPerBlock)
//   - 输出初始化为 0（标量张量）

#define BLOCK_DIM 1024
#define COARSE_FACTOR 2

// TODO: 实现 CoarsenedSumReductionKernel
// __global__ void CoarsenedSumReductionKernel(float* input, float* output, int length) {
//     int i = blockIdx.x * BLOCK_DIM * COARSE_FACTOR + threadIdx.x;
//     __shared__ float input_s[BLOCK_DIM];
//     float sum = 0;
    
//     for (int c = 0; c < COARSE_FACTOR; c++) {
//         if (i + c * BLOCK_DIM < length / 2)
//             input_s[threadIdx.x] = input[i + c * BLOCK_DIM] + input[i + c * BLOCK_DIM + length / 2];
//         else 
//             input_s[threadIdx.x] = 0.0f;
//         __syncthreads();
        
//         for (int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2) {
//             if (threadIdx.x < stride) {
//                 input_s[threadIdx.x] += input_s[threadIdx.x + stride];
//             }
//             __syncthreads();
//         }
        
//         if (threadIdx.x == 0)
//             sum += input_s[0];
//         __syncthreads();
//     }
    
//     if (threadIdx.x == 0) {
//         atomicAdd(&output[0], sum);
//     }
// }

__global__ void CoarsenedSumReductionKernel(float* input, float* output, int length) {
    int segment = blockIdx.x * BLOCK_DIM * COARSE_FACTOR * 2;
    int i = segment + threadIdx.x;
    __shared__ float input_s[BLOCK_DIM];
    float sum = 0;

    // 标准做法：把下面的 for 循环 + 树形归约改成：
    // 1) 用一个循环把 COARSE_FACTOR*2 个 tile 的元素全部累加到寄存器 sum 里：
    //    for (int tile = 0; tile < COARSE_FACTOR * 2; tile++) {
    //        int idx = i + tile * BLOCK_DIM;
    //        if (idx < length) sum += input[idx];
    //    }
    // 2) 写入共享内存：input_s[threadIdx.x] = sum;
    // 3) __syncthreads();
    // 4) 只做一次树形归约（就是下面内层的 for stride 循环）
    // 5) 线程 0 atomicAdd
    // 这样整个内核只需要一次树形归约，而不是 COARSE_FACTOR 次
    for (int tile = 0; tile < COARSE_FACTOR * 2; tile++) {
        int idx = i + tile * BLOCK_DIM;
        if (idx < length) sum += input[idx];
    }
    input_s[threadIdx.x] = sum;
    __syncthreads();
    
    for (int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
        } 
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(&output[0], input_s[0]);
    }
}


// TODO: 实现 host 函数
torch::Tensor reduceCoarsenedSum(torch::Tensor input) {
    int length = input.size(0);
    auto output = torch::zeros({1}, input.options().dtype(torch::kFloat32));
    
    dim3 block(BLOCK_DIM);
    int elementsPerBlock = 2 * COARSE_FACTOR * BLOCK_DIM;
    dim3 grid((length + elementsPerBlock - 1) / elementsPerBlock);
    
    CoarsenedSumReductionKernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), length
    );
    
    return output;
}
