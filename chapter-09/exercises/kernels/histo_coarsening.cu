#include <torch/extension.h>

// 线程粗化直方图内核
//
// 与共享内存版的区别：
//   每个线程处理 CFACTOR 个连续元素，而非 1 个。
//   这减少了线程块数量，从而减少了全局内存原子操作次数。
//
// 提示：
//   - tid = blockIdx.x * blockDim.x + threadIdx.x
//   - 每个线程处理 data[tid*CFACTOR] 到 data[min((tid+1)*CFACTOR, length)-1]
//   - 其余逻辑与共享内存版相同

#define BLOCK_SIZE 256
#define CFACTOR 4

__global__ void HistoCoarseningKernel(const int* data, int* histo,
                                       int length, int num_bins) {
    int i = blockIdx.x * BLOCK_SIZE * CFACTOR + threadIdx.x;
    extern __shared__ int histo_s[];
    for (int bin = threadIdx.x; bin < num_bins; bin += BLOCK_SIZE) {
        if (bin < num_bins) histo_s[bin] = 0;
    }
    __syncthreads();
    
    for (int c = 0; c < CFACTOR; c++) {
        if (i + c * BLOCK_SIZE < length) atomicAdd(&histo_s[data[i + c * BLOCK_SIZE]], 1);
    }
    
    __syncthreads();
    
    for (int bin = threadIdx.x; bin < num_bins; bin += BLOCK_SIZE) {
        if (bin < num_bins && histo_s[bin] != 0) atomicAdd(&histo[bin], histo_s[bin]);
    }
    
}

torch::Tensor histoCoarsening(torch::Tensor data, int num_bins) {
    int length = data.size(0);
    auto histo = torch::zeros({num_bins}, data.options().dtype(torch::kInt32));

    dim3 block(BLOCK_SIZE);
    dim3 grid((length + BLOCK_SIZE * CFACTOR - 1) / (BLOCK_SIZE * CFACTOR));
    int shared_mem_size = num_bins * sizeof(int);

    HistoCoarseningKernel<<<grid, block, shared_mem_size>>>(
        data.data_ptr<int>(), histo.data_ptr<int>(), length, num_bins);

    return histo;
}
