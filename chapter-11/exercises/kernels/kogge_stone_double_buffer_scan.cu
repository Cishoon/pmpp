#include <torch/extension.h>

// Kogge-Stone 双缓冲前缀扫描（inclusive scan）
//
// 与基础版的区别：
//   使用双缓冲（两块共享内存交替读写）代替两次 __syncthreads()
//   每次迭代只需一次 __syncthreads()，消除 write-after-read 竞争
//
// 思路：
//   - 分配 2*N 大小的 extern shared memory，分为 src_buffer 和 trg_buffer
//   - 将输入加载到 src_buffer
//   - stride 从 1 开始翻倍，每次迭代：
//     * threadIdx.x >= stride 时，trg_buffer[tid] = src_buffer[tid] + src_buffer[tid - stride]
//     * 否则直接复制 trg_buffer[tid] = src_buffer[tid]
//     * 交换 src_buffer 和 trg_buffer 指针
//   - 最终结果在 src_buffer 中，写回输出
//
// host 端需要：
//   - 创建输出张量
//   - 配置 grid(1), block(N)，动态共享内存大小 2*N*sizeof(float)
//   - 启动内核并返回结果

#define BLOCK_SIZE 1024

__global__ void KoggeStoneDoubleBufferScanKernel(float* X, float* Y, unsigned int N) {
    int i = threadIdx.x;
    
    extern __shared__ float buffer[];
    float* src = buffer;
    float* tar = &buffer[BLOCK_SIZE];
    
    if (i < N) {
        src[i] = X[i];
    }
    __syncthreads();
    
    for (int stride = 1; stride < N; stride *= 2) {
        if (i < N) {
            if (i >= stride) tar[i] = src[i] + src[i - stride];
            else tar[i] = src[i];
        }
        __syncthreads();
        float* tmp;
        tmp = src;
        src = tar;
        tar = tmp;
    }
    
    if (i < N) {
        Y[i] = src[i];
    }
}

// 函数签名: 
torch::Tensor kogge_stone_double_buffer_scan(torch::Tensor input) {
    int length = input.size(0);
    auto output = torch::empty_like(input);
    
    dim3 grid(1);
    dim3 block(BLOCK_SIZE);
    size_t buffer_size = BLOCK_SIZE * sizeof(float) * 2;
    
    KoggeStoneDoubleBufferScanKernel<<<grid, block, buffer_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), length
    );
    
    return output;
}
