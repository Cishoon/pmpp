#include <torch/extension.h>

// 分层 Kogge-Stone 前缀扫描（支持任意长度）
//
// 与单 block 版本的区别：
//   通过三阶段分层策略支持任意长度输入
//   Phase 1: 每个 block 独立做 Kogge-Stone 扫描，保存 block 末尾的和到辅助数组 S
//   Phase 2: 对辅助数组 S 做 Kogge-Stone 扫描
//   Phase 3: 将 S 中的前缀和加回到对应 block 的每个元素
//
// 思路：
//   Phase 1 内核：
//     - global_idx = blockIdx.x * blockDim.x + threadIdx.x
//     - 加载到共享内存，执行标准 Kogge-Stone 扫描
//     - 写回 Y[global_idx]
//     - 最后一个线程将 block 的总和写入 S[blockIdx.x]
//   Phase 2 内核：
//     - 对 S 数组执行 Kogge-Stone 扫描（S 长度 = num_blocks）
//   Phase 3 内核：
//     - blockIdx.x > 0 的 block，每个元素加上 S[blockIdx.x - 1]
//
// host 端需要：
//   - 计算 num_blocks = ceil(N / BLOCK_SIZE)
//   - 分配输出张量和辅助张量 S（长度 num_blocks）
//   - 依次启动三个内核
//   - 返回输出张量

#define BLOCK_SIZE 1024

__global__ void HierarchicalScanPhase1(float* g_X, float* g_Y, float* g_S, unsigned int N) {
    // TODO
    int g_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    // 每一块一个共享内存，共享内存里放一块（ BLOCK_SIZE 个数据），需要 BLOCK_SIZE* sizeof(float)
    extern __shared__ float buf[]; 
    
    // 读取数据
    buf[tid] = g_idx < N ? g_X[g_idx] : 0.0f;
    __syncthreads();
    
    // kogge stone 前缀和
    for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        float tmp = 0.0f;
        if (tid >= stride) tmp = buf[tid] + buf[tid - stride];
        __syncthreads();
        if (tid >= stride) buf[tid] = tmp;
        __syncthreads();
    }
    
    // 给 S 赋值为 buf[BLOCK_SIZE - 1]
    if (tid == 0) g_S[blockIdx.x] = buf[BLOCK_SIZE - 1];    
    if (g_idx < N) g_Y[g_idx] = buf[tid];
}

__global__ void HierarchicalScanPhase2(float* g_S, unsigned int num_blocks) {
    int tid = threadIdx.x;
    extern __shared__ float buf[];
    
    // 读数据
    // if (tid < num_blocks) buf[tid] = g_S[tid]
    // else buf[tid] = 0.0f;
    buf[tid] = tid < num_blocks ? g_S[tid] : 0.0f;
    __syncthreads();
    
    for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        float tmp = 0.0f;
        if (tid >= stride) tmp = buf[tid] + buf[tid - stride];
        __syncthreads();
        if (tid >= stride) buf[tid] = tmp;
        __syncthreads();
    }
    
    if (tid < num_blocks) g_S[tid] = buf[tid];
}

__global__ void HierarchicalScanPhase3(float* g_Y, float* g_S, unsigned int N) {
    // 把S[0] 加到 Y[1 * BLOCK_SIZE, 2 * BLOCK_SIZE - 1] 上
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float offset = blockIdx.x > 0 ? g_S[blockIdx.x - 1] : 0.0f;
    
    if (g_idx < N) g_Y[g_idx] += offset;
}

torch::Tensor hierarchical_scan(torch::Tensor input) {
    int N = input.size(0);
    auto output = torch::empty_like(input);
    if (N == 0) return output;
    
    // 一个BLOCK放 BLOCK_SIZE 个线程，可以处理 BLOCK_SIZE 长度的数组
    // 所以总共要处理 ceil(N / BLOCK_SIZE) 块
    
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // 要求 S 不分块。所以最多分BLOCK_SIZE块
    assert(num_blocks <= BLOCK_SIZE);
    
    dim3 block(BLOCK_SIZE);
    dim3 grid(num_blocks);
    
    // 整体思路和线程粗化版本的一样，但是每一块的总和得单独开辟一个global空间了
    auto s = torch::zeros({1}, input.options().dtype(torch::kFloat32));
    HierarchicalScanPhase1<<<grid, block, BLOCK_SIZE * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), s.data_ptr<float>(), N
    );
    
    dim3 grid2(1);
    HierarchicalScanPhase2<<<grid2, block, BLOCK_SIZE * sizeof(float)>>>(
        s.data_ptr<float>(), num_blocks
    );
    
    HierarchicalScanPhase3<<<grid, block>>>(
        output.data_ptr<float>(), s.data_ptr<float>(), N
    );
    
    return output;
}
