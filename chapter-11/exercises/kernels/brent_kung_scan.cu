#include <torch/extension.h>

// Brent-Kung 并行前缀扫描（inclusive scan）
//
// 与 Kogge-Stone 的区别：
//   分为两个阶段：归约树（up-sweep）和反向树（down-sweep）
//   工作效率更高，总操作数约 2N - 2 - log2(N)
//   每个线程处理两个元素，blockDim.x = N/2
//
// 思路：
//   - 每个线程加载两个元素到共享内存：XY[tid] 和 XY[tid + blockDim.x]
//   - 归约树阶段（stride 从 1 到 blockDim.x）：
//     * index = (threadIdx.x + 1) * 2 * stride - 1
//     * 若 index < SECTION_SIZE，则 XY[index] += XY[index - stride]
//   - 反向树阶段（stride 从 SECTION_SIZE/4 到 1）：
//     * index = (threadIdx.x + 1) * stride * 2 - 1
//     * 若 index + stride < SECTION_SIZE，则 XY[index + stride] += XY[index]
//   - 将结果写回输出
//
// host 端需要：
//   - 创建输出张量
//   - 配置 grid(1), block(N/2)，动态共享内存大小 N*sizeof(float)（N 需向上取到 2 的幂）
//   - 启动内核并返回结果

#define SECTION_SIZE 2048
using u32 = unsigned int;
using i32 = int;
__global__ void BrentKungInclusiveScanKernel(float* X, float* Y, unsigned int N) {
    extern __shared__ float s[];
    u32 tid = threadIdx.x;
    u32 segment = blockDim.x * 2;
    
    u32 i0 = 2 * tid, i1 = 2 * tid + 1;
    s[i0] = (i0 < N) ? X[i0] : 0.0f;
    s[i1] = (i1 < N) ? X[i1] : 0.0f;
    __syncthreads();
    
    for (u32 stride = 1; stride < segment; stride *= 2) {
        u32 idx = (tid + 1) * (stride * 2) - 1;
        if (idx < segment) {
            s[idx] += s[idx - stride];
        }
        __syncthreads();
    }
    
    for (u32 stride = segment / 4; stride >= 1; stride /= 2) {
        u32 idx = (tid + 1) * (stride * 2) + stride - 1;
        if (idx < segment) {
            s[idx] += s[idx - stride];
        }
        __syncthreads();
    }
    if (i0 < N) Y[i0] = s[i0];
    if (i1 < N) Y[i1] = s[i1];
    
}

torch::Tensor brent_kung_scan(torch::Tensor input) {
    int N = input.size(0);
    auto output = torch::empty_like(input);
    
    int n_pow2 = 1;
    if (N > 0) {
        n_pow2 = 1 << (32 - __builtin_clz(N - 1));
    }
    
    dim3 grid(1);    
    dim3 block(n_pow2 / 2);
    int shared_size = n_pow2 * sizeof(float);
    BrentKungInclusiveScanKernel<<<grid, block, shared_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N
    );
    
    return output;
}
