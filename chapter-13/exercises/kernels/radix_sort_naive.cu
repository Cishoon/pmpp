#include <torch/extension.h>
#include <cub/cub.cuh>

// 1. 提取当前 bit 位到全局数组
__global__ void extractBitsKernel(int* d_input, int* d_bits, int N, int iter) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        d_bits[gid] = (d_input[gid] >> iter) & 1;
    }
}

// 2. Naive 的 Scatter，直接根据全局前缀和计算坐标
__global__ void naiveScatterKernel(int* d_input, int* d_output, int* d_bits, 
                                   int* d_prefixOnes, int totalZeros, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        int bit = d_bits[gid];
        int prefixOnes = d_prefixOnes[gid];
        int dest;
        
        if (bit == 0) {
            // 当前索引 gid 减去前面的 1 的数量，就是前面的 0 的数量
            dest = gid - prefixOnes; 
        } else {
            dest = totalZeros + prefixOnes;
        }
        d_output[dest] = d_input[gid];
    }
}

// 获取最后一个元素的值来计算总数 (用于求 totalOnes)
__global__ void getTotalOnesKernel(int* d_bits, int* d_prefixOnes, int* d_totalOnes, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 最后一个元素的前缀和 + 最后一个元素本身的 bit
        d_totalOnes[0] = d_prefixOnes[N - 1] + d_bits[N - 1];
    }
}

// 主机端调用
torch::Tensor radix_sort_naive(torch::Tensor input) {
    int N = input.size(0);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    auto current_in = input.clone();
    auto current_out = torch::empty_like(input);
    auto options = torch::dtype(torch::kInt32).device(input.device());
    
    // Naive 必须开辟和输入一样大的 N 长度数组来存 bit 和 前缀和
    auto d_bits = torch::empty({N}, options);
    auto d_prefixOnes = torch::empty({N}, options);
    auto d_totalOnes = torch::empty({1}, options);
    
    int* ptr_in = current_in.data_ptr<int>();
    int* ptr_out = current_out.data_ptr<int>();
    int* ptr_bits = d_bits.data_ptr<int>();
    int* ptr_prefixOnes = d_prefixOnes.data_ptr<int>();
    int* ptr_totalOnes = d_totalOnes.data_ptr<int>();
    
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, ptr_bits, ptr_prefixOnes, N);
    auto d_temp_storage = torch::empty({(long)temp_storage_bytes}, torch::dtype(torch::kUInt8).device(input.device()));
    void* d_temp = d_temp_storage.data_ptr();
    
    for (int iter = 0; iter < 32; ++iter) {
        // 步骤 1: 将所有元素的当前位提取到 d_bits
        extractBitsKernel<<<blocks, threads>>>(ptr_in, ptr_bits, N, iter);
        
        // 步骤 2: 对整个长度为 N 的 d_bits 数组做前缀和
        cub::DeviceScan::ExclusiveSum(d_temp, temp_storage_bytes, ptr_bits, ptr_prefixOnes, N);
        
        // 步骤 3: 计算 1 的总数
        getTotalOnesKernel<<<1, 1>>>(ptr_bits, ptr_prefixOnes, ptr_totalOnes, N);
        
        // 把 totalOnes 拷回 CPU 算出 totalZeros（这一步会引起 CPU/GPU 同步阻塞）
        int totalOnes_cpu;
        cudaMemcpy(&totalOnes_cpu, ptr_totalOnes, sizeof(int), cudaMemcpyDeviceToHost);
        int totalZeros = N - totalOnes_cpu;
        
        // 步骤 4: Naive Scatter
        naiveScatterKernel<<<blocks, threads>>>(ptr_in, ptr_out, ptr_bits, ptr_prefixOnes, totalZeros, N);
        
        std::swap(ptr_in, ptr_out);
    }
    
    return current_in;
}