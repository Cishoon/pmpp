#include <torch/extension.h>

// 线程粗化 + 合并访存直方图内核
//
// 与线程粗化版的区别：
//   线程粗化版中每个线程处理连续的 CFACTOR 个元素，
//   导致相邻线程访问的内存地址不连续（非合并访存）。
//
//   本版本改用跨步访问模式：
//     for (i = tid; i < length; i += blockDim.x * gridDim.x)
//   这样同一 warp 内的线程访问连续地址，实现合并访存。
//
// 注意：grid 大小不再由数据量决定，而是固定为 SM 数量 × 每 SM 块数，
//       确保每个 SM 有足够的工作量。
//
// 提示：
//   - tid = blockIdx.x * blockDim.x + threadIdx.x
//   - stride = blockDim.x * gridDim.x
//   - for (i = tid; i < length; i += stride) { atomicAdd(&histo_s[data[i]], 1); }

#define BLOCK_SIZE 256

__global__ void HistoCoarsenedCoalescedKernel(const int* data, int* histo,
                                               int length, int num_bins) {
    extern __shared__ int histo_s[];
    for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        histo_s[bin] = 0;
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int j = tid; j < length; j += stride) {
        atomicAdd(&histo_s[data[j]], 1);
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        if (histo_s[bin] > 0) {
            atomicAdd(&histo[bin], histo_s[bin]);
        }
    }
}

torch::Tensor histoCoarsenedCoalesced(torch::Tensor data, int num_bins) {
    int length = data.size(0);
    auto histo = torch::zeros({num_bins}, data.options().dtype(torch::kInt32));

    // 获取 SM 数量来决定 grid 大小
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int num_sms = prop.multiProcessorCount;

    dim3 block(BLOCK_SIZE);
    dim3 grid(num_sms * 32);  // 每个 SM 分配 32 个块
    int shared_mem_size = num_bins * sizeof(int);

    HistoCoarsenedCoalescedKernel<<<grid, block, shared_mem_size>>>(
        data.data_ptr<int>(), histo.data_ptr<int>(), length, num_bins);

    return histo;
}
