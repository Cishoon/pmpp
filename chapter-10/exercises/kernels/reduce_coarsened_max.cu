#include <torch/extension.h>
#include <cfloat>

// 线程粗化求最大值归约内核（习题 4 扩展）
//
// 与粗化求和版的区别：
//   - 累加操作 (+= ) 替换为 fmax
//   - atomicAdd 替换为自定义的 atomicMax（浮点数版本）
//   - 初始值使用 -FLT_MAX 而非 0
//
// 思路：
//   - segment, i, t 的计算与求和版相同
//   - 粗化阶段：max_val = input[i]，循环中用 fmax 取最大
//   - 树形归约：input_s[t] = fmax(input_s[t], input_s[t + stride])
//   - 线程 0 需要将 block 结果与全局 output 取最大
//     由于 CUDA 没有浮点 atomicMax，可以用以下技巧：
//     atomicExch(output, fmax(*output, input_s[0]))
//     注意：这种方式在高竞争下不完全安全，
//     更严谨的做法是用 atomicCAS 实现浮点 atomicMax
//   - 支持任意长度输入，需要边界检查
//
// host 端需要：
//   - 输出初始化为 -FLT_MAX 的标量张量
//   - grid/block 配置与求和版相同

#define BLOCK_DIM 1024
#define COARSE_FACTOR 2

__device__ void atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

// TODO: 实现 CoarsenedMaxReductionKernel
__global__ void CoarsenedMaxReductionKernel(float* input, float* output, int length) {
    int i = blockIdx.x * BLOCK_DIM * COARSE_FACTOR * 2 + threadIdx.x;
    __shared__ float input_s[BLOCK_DIM];
    float max_val = -FLT_MAX;
    
    for (int tile = 0; tile < COARSE_FACTOR * 2; tile++) {
        int idx = i + tile * BLOCK_DIM;
        if (idx < length) {
            max_val = fmax(max_val, input[idx]);
        }
    }
    input_s[threadIdx.x] = max_val;
    __syncthreads();
    
    for (int stride = BLOCK_DIM / 2; stride >= 1; stride >>= 1) {
        if (threadIdx.x < stride) {
            input_s[threadIdx.x] = fmax(input_s[threadIdx.x], input_s[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicMaxFloat(output, input_s[0]);
    }
}

// TODO: 实现 host 函数
// 函数签名: torch::Tensor reduceCoarsenedMax(torch::Tensor input)
torch::Tensor reduceCoarsenedMax(torch::Tensor input) {
    auto output = torch::zeros({1}, input.options().dtype(torch::kFloat32)) - FLT_MAX;
    int length = input.size(0);
    
    dim3 block(BLOCK_DIM);
    int elementsPerBlock = 2 * COARSE_FACTOR * BLOCK_DIM;
    dim3 grid((length + elementsPerBlock - 1) / elementsPerBlock);
    CoarsenedMaxReductionKernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), length
    );
    return output;
}
