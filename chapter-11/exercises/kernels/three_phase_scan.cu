#include <torch/extension.h>

// 三阶段前缀扫描（Three-phase scan / 线程粗化扫描）
//
// 与 Kogge-Stone 的区别：
//   每个线程负责 COARSE_FACTOR 个连续元素，减少了并行扫描阶段的线程数
//   分三个阶段完成：局部扫描 → 段尾扫描 → 分发累加
//   工作效率更高，减少了 Kogge-Stone 中的冗余加法
//
// 思路：
//   - 分配 extern shared memory，前 N 个 float 为 buffer，后面为 sections_ends
//   - 阶段 1（局部扫描）：
//     * 每个线程加载 COARSE_FACTOR 个元素到 buffer
//     * 对自己负责的 COARSE_FACTOR 个元素做顺序前缀和（idx 从 1 开始，buffer[idx] += buffer[idx-1]）
//     * 将每段末尾值存入 sections_ends[threadIdx.x]
//   - 阶段 2（段尾扫描）：
//     * 对 sections_ends 数组执行 Kogge-Stone 扫描（长度 = num_sections = N/COARSE_FACTOR）
//     * stride 从 1 开始翻倍，需要两次 __syncthreads() 防止竞争
//   - 阶段 3（分发累加）：
//     * 每个线程遍历自己的 COARSE_FACTOR 个元素
//     * 若 section > 0，则 buffer[idx] += sections_ends[section - 1]
//     * 写回输出
//
// 限制：仅支持单 block（N <= BLOCK_SIZE * COARSE_FACTOR）
//
// host 端需要：
//   - 创建输出张量
//   - 计算 num_threads = ceil(N / COARSE_FACTOR)
//   - 配置 grid(1), block(num_threads)
//   - 动态共享内存大小 = (N + num_threads) * sizeof(float)
//   - 启动内核并返回结果

#define COARSE_FACTOR 4

__global__ void ThreePhaseScanKernel(float* X, float* Y, unsigned int N) {
    // 前面N个用来缓存前缀和的中间结果
    // 后面 num_threads（blockDim.x）存放每一个 线程 处理部分的最后一个数
    extern __shared__ float shared_data[];
    float* buf = shared_data;
    float* sections_ends = &shared_data[N];
    
    const int CF = COARSE_FACTOR;
    // 载入数据，每个线程载入4个连续数据
    int tid = threadIdx.x;
    int startIdx = tid * CF;
    for (int i = 0; i < CF; i++) {
        if (startIdx + i < N) { 
            if (i >= 1) buf[startIdx + i] = X[startIdx + i] + buf[startIdx + i - 1];
            else buf[startIdx + i] = X[startIdx + i];
        }
    }
    // 直接四个一组，计算好了前缀和。
    if (startIdx < N) {
        int last_valid_idx = (startIdx + CF - 1 < N) ? (startIdx + CF - 1) : (N - 1);
        sections_ends[tid] = buf[last_valid_idx];
    }
    __syncthreads();
    
    // 对sections_ends做前缀和
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float tmp = 0.0f;
        if (tid >= stride) {
            tmp = sections_ends[tid - stride];
        }
        __syncthreads();
        
        if (tid >= stride) {
            sections_ends[tid] += tmp;
        }
        __syncthreads();
    }
    
    // 把前缀和分发下去，加到每个独立的模块里。
    // sections_end[0] 加到 buf[4,5,6,7]
    if (tid > 0) {
        float prev_sum = sections_ends[tid - 1]; // 前一个块的最终前缀和
        for (int i = 0; i < CF; i++) {
            if (startIdx + i < N) {
                buf[startIdx + i] += prev_sum;
            }
        }
    }
    
    // 输出到Y
    for (int i = 0; i < CF; i++) {
        if (startIdx + i < N) Y[startIdx + i] = buf[startIdx + i];
    }
}

torch::Tensor three_phase_scan(torch::Tensor input) {
    int N = input.numel();
    auto output = torch::empty_like(input);
    if (N == 0) return output;
    
    int num_threads = (N + COARSE_FACTOR - 1) / COARSE_FACTOR;
    
    dim3 grid(1);
    dim3 block(num_threads);
    int shmem_size = (N + num_threads) * sizeof (float);
    
    ThreePhaseScanKernel<<<grid, block, shmem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N
    );
    
    return output;
}
