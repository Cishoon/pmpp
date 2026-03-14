#include <torch/extension.h>

// 朴素直方图内核
// 输入 data: 1D 整数张量，每个元素值在 [0, num_bins) 范围内
// 输出 histo: 长度为 num_bins 的直方图
//
// 每个线程处理一个输入元素，直接对全局内存执行 atomicAdd
//
// 提示：
//   - i = blockIdx.x * blockDim.x + threadIdx.x
//   - 边界检查：i < length
//   - atomicAdd(&histo[data[i]], 1)

#define BLOCK_SIZE 256

__global__ void HistoBasicKernel(const int* data, int* histo, int length) {
    // TODO: 计算全局索引
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // TODO: 边界检查后对 histo[data[i]] 执行 atomicAdd
    if (i < length)
        atomicAdd(&histo[data[i]], 1);
}

torch::Tensor histoBasic(torch::Tensor data, int num_bins) {
    int length = data.size(0);
    auto histo = torch::zeros({num_bins}, data.options().dtype(torch::kInt32));

    dim3 block(BLOCK_SIZE);
    dim3 grid((length + BLOCK_SIZE - 1) / BLOCK_SIZE);

    HistoBasicKernel<<<grid, block>>>(
        data.data_ptr<int>(), histo.data_ptr<int>(), length);

    return histo;
}
