// 内存合并基数排序
//
// 与朴素版的区别：
//   使用共享内存进行 block 内局部扫描，scatter 时按 block 偏移写入，实现内存合并访问
//
// 思路：
//   - 每轮处理 1 个 bit 位，使用两个内核：
//     1. localScanKernel:
//        - 每个 block 将自己负责的元素的 bit 值加载到共享内存
//        - 在共享内存中执行 Brent-Kung exclusive scan（先 up-sweep 再 down-sweep）
//        - 将局部扫描结果写入 d_localScan（每个元素之前有多少个 1）
//        - 将每个 block 中 1 的总数写入 d_blockOneCount
//     2. scatterKernelCoalesced:
//        - 根据 bit 值、局部前缀和、block 级别的零偏移和一偏移计算目标位置
//        - bit==0: dest = blockZeroOffset + tid - localPrefix
//        - bit==1: dest = totalZeros + blockOneOffset + localPrefix
//   - host 端在两个内核之间计算 block 级别的偏移量（exclusive scan on block counts）
//
// host 端需要：
//   - 分配 d_output, d_localScan, d_blockOneCount, d_blockZeroOffsets, d_blockOneOffsets
//   - 循环 32 轮，每轮：
//     启动 localScanKernel → 拷贝 blockOneCount 到 host → 计算 blockZero/OneOffsets → 拷回 device → 启动 scatter
//   - 返回排序结果
#include <torch/extension.h>
#include <cub/cub.cuh>  // 引入 CUB 库以使用 DeviceScan

#define BLOCK_SIZE 256
#define NUM_BITS 32
#define cdiv(a, b) (((a) + (b) - 1) / (b))

// 1. 局部扫描内核 (不变，去掉了 host 端拷贝注释)
__global__ void localScanKernel(int* d_input, int* d_localScan, int* d_blockOneCount, int N, int iter) {
    __shared__ int temp[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    if (gid < N) {
        temp[tid] = (d_input[gid] >> iter) & 1;
    } else {
        temp[tid] = 0;
    }
    __syncthreads();
    
    // Brent-Kung exclusive scan
    int offset = 1;
    for (int d = BLOCK_SIZE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }
    
    __syncthreads();
    if (tid == 0) {
        if (blockIdx.x < gridDim.x) d_blockOneCount[blockIdx.x] = temp[BLOCK_SIZE - 1];
        temp[BLOCK_SIZE - 1] = 0;
    }
    
    for (int d = 1; d < BLOCK_SIZE; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    
    if (gid < N) {
        d_localScan[gid] = temp[tid];
    }
}

// 2. 新增：在 GPU 端直接计算每个 Block 里 0 的个数
__global__ void computeBlockCountsKernel(int* d_blockOneCount, int* d_blockZeroCount, int N, int numBlocks, int BLOCK_SIZE_VAL) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid < numBlocks) {
        int elements_in_block = (bid == numBlocks - 1) ? (N - bid * BLOCK_SIZE_VAL) : BLOCK_SIZE_VAL;
        d_blockZeroCount[bid] = elements_in_block - d_blockOneCount[bid];
    }
}

// 3. 修改：Scatter 内核，直接在 GPU 内部计算 totalZeros
__global__ void scatterKernelCoalesced(int* d_input, int* d_output, int* d_localScan,
                                       int* d_blockZeroOffsets, int* d_blockOneOffsets,
                                       int* d_blockZeroCount, int numBlocks, int N, int iter) {
    __shared__ int smem[BLOCK_SIZE];
                                        
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // 所有的线程都可以直接算出 totalZeros，无须 CPU 传入
    // totalZeros = 最后一个 block 的 0 的前缀和 + 最后一个 block 本身包含的 0 的数量
    int totalZeros = d_blockZeroOffsets[numBlocks - 1] + d_blockZeroCount[numBlocks - 1];
    int zerosInThisBlock = d_blockZeroCount[blockIdx.x];
    
    
    // 阶段1：读取全局内存，scatter到shared memory
    if (gid < N) {
        int val = d_input[gid];
        int bit = (val >> iter) & 1;
        int localPrefixOnes = d_localScan[gid];
        
        int smem_dest;
        if (bit == 0) {
            smem_dest = tid - localPrefixOnes; // 当前线程在 block 内的 0 的相对位置
        } else {
            smem_dest = zerosInThisBlock + localPrefixOnes; // 当前线程在 block 内的 1 的相对位置
        }
        smem[smem_dest] = val;
    }
    __syncthreads();
    
    int elements_in_this_block = (blockIdx.x == numBlocks - 1) ? (N - blockIdx.x * blockDim.x) : blockDim.x;
    // 阶段2：从 shared memory 顺序读取，合并写入全局内存
    if (tid < elements_in_this_block) {
        int sorted_val = smem[tid];
        int global_dest;
        
        if (tid < zerosInThisBlock) {
            global_dest = d_blockZeroOffsets[blockIdx.x] + tid; // 写入全局内存的 0 的位置
        } else {
            global_dest = totalZeros + d_blockOneOffsets[blockIdx.x] + (tid - zerosInThisBlock); // 写入全局内存的 1 的位置
        }
        d_output[global_dest] = sorted_val;
    }
}

// 主机端调用函数
torch::Tensor radix_sort_coalesced(torch::Tensor input) {
    int N = input.size(0);
    int numBlocks = cdiv(N, BLOCK_SIZE);
    
    auto current_in = input.clone();
    auto current_out = torch::empty_like(input);
    
    auto options = torch::dtype(torch::kInt32).device(input.device());
    
    auto d_localScan = torch::empty_like(input);
    auto d_blockOneCount = torch::empty({numBlocks}, options);
    auto d_blockZeroCount = torch::empty({numBlocks}, options); // 新增：存 0 的数量
    auto d_blockOneOffsets = torch::empty({numBlocks}, options);
    auto d_blockZeroOffsets = torch::empty({numBlocks}, options);
    
    int* ptr_in = current_in.data_ptr<int>();
    int* ptr_out = current_out.data_ptr<int>();
    int* ptr_localScan = d_localScan.data_ptr<int>();
    int* ptr_boc = d_blockOneCount.data_ptr<int>();
    int* ptr_bzc = d_blockZeroCount.data_ptr<int>();
    int* ptr_boo = d_blockOneOffsets.data_ptr<int>();
    int* ptr_bzo = d_blockZeroOffsets.data_ptr<int>();
    
    // 初始化 CUB 所需的临时存储空间大小
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, ptr_boc, ptr_boo, numBlocks);
    auto d_temp_storage = torch::empty({(long)temp_storage_bytes}, torch::dtype(torch::kUInt8).device(input.device()));
    void* d_temp = d_temp_storage.data_ptr();
    
    int computeBlocks = cdiv(numBlocks, 256);

    for (int iter = 0; iter < NUM_BITS; ++iter) {
        // 步骤 1：局部扫描，得到 localScan 和 blockOneCount
        localScanKernel<<<numBlocks, BLOCK_SIZE>>>(ptr_in, ptr_localScan, ptr_boc, N, iter);
        
        // 步骤 2：并行计算 blockZeroCount
        computeBlockCountsKernel<<<computeBlocks, 256>>>(ptr_boc, ptr_bzc, N, numBlocks, BLOCK_SIZE);
        
        // 步骤 3：调用 CUB 直接在显存里做 Exclusive Scan，彻底摆脱 CPU
        cub::DeviceScan::ExclusiveSum(d_temp, temp_storage_bytes, ptr_boc, ptr_boo, numBlocks);
        cub::DeviceScan::ExclusiveSum(d_temp, temp_storage_bytes, ptr_bzc, ptr_bzo, numBlocks);
        
        // 步骤 4：Scatter 写入（将 ptr_bzc 传进去算 totalZeros）
        scatterKernelCoalesced<<<numBlocks, BLOCK_SIZE>>>(
            ptr_in, ptr_out, ptr_localScan, ptr_bzo, ptr_boo, ptr_bzc, numBlocks, N, iter
        );
        
        // 交换指针
        std::swap(ptr_in, ptr_out);
    }
    
    return current_in;
}