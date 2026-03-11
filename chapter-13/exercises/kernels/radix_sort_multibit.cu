#include <torch/extension.h>
#include <cub/cub.cuh>

#define BLOCK_SIZE 256
#define RADIX_BITS 4
#define cdiv(a, b) (((a) + (b) - 1) / (b))

// 1. 局部扫描与直方图构建
__global__ void localScanKernelMultibitRadix(const int* d_input, int* d_localOffsets,
                                              int* d_blockHist, int N, int pass, int r) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int numBuckets = 1 << r; // 比如 r=4, 则有 16 个桶

    // 分配共享内存
    extern __shared__ int shared_mem[];
    int* s_hist = shared_mem;                       // 大小: numBuckets
    int* s_digits = (int*)&shared_mem[numBuckets];  // 大小: BLOCK_SIZE

    // 初始化共享内存直方图
    if (tid < numBuckets) {
        s_hist[tid] = 0;
    }
    __syncthreads();

    int digit = -1;
    // 读取数据并提取当前 bit 位的 digit
    if (gid < N) {
        int val = d_input[gid];
        digit = (val >> (pass * r)) & (numBuckets - 1);
        s_digits[tid] = digit;
        
        // 原子加法统计 block 级直方图
        atomicAdd(&s_hist[digit], 1);
    } else {
        s_digits[tid] = -1; // 越界线程设为无效值
    }
    __syncthreads();

    // 计算 localOffset（必须通过遍历前驱线程来保证排序的稳定性）
    if (gid < N) {
        int offset = 0;
        // 遍历同一 Block 内排在我前面的所有线程
        for (int i = 0; i < tid; ++i) {
            if (s_digits[i] == digit) {
                offset++;
            }
        }
        d_localOffsets[gid] = offset;
    }
    __syncthreads();

    // 将 Block 直方图写回全局内存。
    // 【关键优化】：采用 bucket * gridDim.x + blockIdx.x 的布局
    if (tid < numBuckets) {
        d_blockHist[tid * gridDim.x + blockIdx.x] = s_hist[tid];
    }
}

// 2. 数据散布 (Scatter)
__global__ void scatterKernelMultibitRadix(const int* d_input, int* d_output,
                                            const int* d_localOffsets, const int* d_globalOffsets,
                                            int N, int pass, int r) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int numBuckets = 1 << r;

    if (gid < N) {
        int val = d_input[gid];
        int digit = (val >> (pass * r)) & (numBuckets - 1);
        int localOffset = d_localOffsets[gid];

        // 去全局偏移表中查询当前 Block 的当前桶的起始绝对坐标
        int globalOffset = d_globalOffsets[digit * gridDim.x + blockIdx.x];

        // 目标位置 = 全局基址 + 局部相对偏移
        int dest = globalOffset + localOffset;
        d_output[dest] = val;
    }
}

// 3. Host 端调度函数
torch::Tensor radix_sort_multibit(torch::Tensor input) {
    int N = input.size(0);
    int numBlocks = cdiv(N, BLOCK_SIZE);
    int numBuckets = 1 << RADIX_BITS;
    int numPasses = cdiv(32, RADIX_BITS); // 处理 32 位整数所需轮数，4-bit 需要 8 轮

    auto current_in = input.clone();
    auto current_out = torch::empty_like(input);
    auto options = torch::dtype(torch::kInt32).device(input.device());

    auto d_localOffsets = torch::empty_like(input);
    // 数组大小为 桶数 * Block数
    auto d_blockHist = torch::empty({numBuckets * numBlocks}, options);
    auto d_globalOffsets = torch::empty({numBuckets * numBlocks}, options);

    int* ptr_in = current_in.data_ptr<int>();
    int* ptr_out = current_out.data_ptr<int>();
    int* ptr_localOffsets = d_localOffsets.data_ptr<int>();
    int* ptr_blockHist = d_blockHist.data_ptr<int>();
    int* ptr_globalOffsets = d_globalOffsets.data_ptr<int>();

    // 获取 CUB Scan 所需的临时存储空间
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, ptr_blockHist, ptr_globalOffsets, numBuckets * numBlocks);
    auto d_temp_storage = torch::empty({(long)temp_storage_bytes}, torch::dtype(torch::kUInt8).device(input.device()));
    void* d_temp = d_temp_storage.data_ptr();

    // 动态共享内存大小：16 个直方图计数器 + 256 个 digit 缓存
    size_t shared_mem_size = (numBuckets + BLOCK_SIZE) * sizeof(int);

    for (int pass = 0; pass < numPasses; ++pass) {
        // 步骤 1：局部扫描
        localScanKernelMultibitRadix<<<numBlocks, BLOCK_SIZE, shared_mem_size>>>(
            ptr_in, ptr_localOffsets, ptr_blockHist, N, pass, RADIX_BITS
        );

        // 步骤 2：一次性完成所有桶、所有 Block 的全局前缀和！
        // 因为我们的内存排布是 [Bucket0_Block0 ... Bucket0_BlockN] [Bucket1_Block0 ... Bucket1_BlockN]
        // 一个 Flat Exclusive Scan 会自动把 Bucket 0 的总和滚入 Bucket 1 的起始位置！
        cub::DeviceScan::ExclusiveSum(
            d_temp, temp_storage_bytes, ptr_blockHist, ptr_globalOffsets, numBuckets * numBlocks
        );

        // 步骤 3：散布写入
        scatterKernelMultibitRadix<<<numBlocks, BLOCK_SIZE>>>(
            ptr_in, ptr_out, ptr_localOffsets, ptr_globalOffsets, N, pass, RADIX_BITS
        );

        // 交换指针
        std::swap(ptr_in, ptr_out);
    }

    return current_in;
}