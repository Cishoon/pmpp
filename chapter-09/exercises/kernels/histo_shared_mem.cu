#include <torch/extension.h>

// 共享内存私有化直方图内核
//
// 与朴素版的区别：
//   每个块在共享内存中维护一份私有直方图，块内原子操作在共享内存上执行（更快），
//   最后将私有直方图累加到全局内存。
//
// 步骤：
//   1. 初始化共享内存直方图为 0（协作初始化）
//   2. __syncthreads()
//   3. 每个线程对共享内存直方图执行 atomicAdd
//   4. __syncthreads()
//   5. 协作将共享内存直方图累加到全局内存（atomicAdd）
//
// 提示：
//   - 初始化：for (bin = threadIdx.x; bin < num_bins; bin += blockDim.x) histo_s[bin] = 0
//   - 累加：同样用循环，每个线程负责若干 bin

#define BLOCK_SIZE 256

__global__ void HistoSharedMemKernel(const int* data, int* histo,
                                      int length, int num_bins) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    extern __shared__ int histo_s[]; // [0, num_bins)
    for (int bin = threadIdx.x; bin < num_bins; bin += BLOCK_SIZE) {
        if (bin < num_bins) histo_s[bin] = 0;
    }
    __syncthreads();
    
    if (i < length) atomicAdd(&histo_s[data[i]], 1);
    
    __syncthreads();
    
    for (int bin = threadIdx.x; bin < num_bins; bin += BLOCK_SIZE) {
        if (bin < num_bins) atomicAdd(&histo[bin], histo_s[bin]); 
    }
    
}

torch::Tensor histoSharedMem(torch::Tensor data, int num_bins) {
    int length = data.size(0);
    auto histo = torch::zeros({num_bins}, data.options().dtype(torch::kInt32));

    dim3 block(BLOCK_SIZE);
    dim3 grid((length + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int shared_mem_size = num_bins * sizeof(int);

    HistoSharedMemKernel<<<grid, block, shared_mem_size>>>(
        data.data_ptr<int>(), histo.data_ptr<int>(), length, num_bins);

    return histo;
}
