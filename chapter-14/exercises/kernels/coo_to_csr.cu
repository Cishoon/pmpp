#include <torch/extension.h>

// COO 到 CSR 格式转换（基于 histogram + prefix sum）
//
// 输入:
//   rowIdx: 1D int32 张量，长度 nnz，COO 格式的行索引（已按行排序）
//   colIdx: 1D int32 张量，长度 nnz，COO 格式的列索引
//   values: 1D float 张量，长度 nnz，COO 格式的非零值
//   numRows: 矩阵行数
// 输出: rowPtrs — 1D int32 张量，长度 numRows+1，CSR 格式的行指针
//
// 思路：
//   1. Histogram：统计每行的非零元素个数
//      - 每个线程处理一个非零元素，对 rowPtrs[rowIdx[i] + 1] 做 atomicAdd
//   2. Prefix Sum（exclusive scan）：对 rowPtrs 做前缀和
//      - 使得 rowPtrs[r] 表示第 r 行在 colIdx/values 中的起始位置
//      - rowPtrs[numRows] 等于 nnz
//   - colIdx 和 values 在 COO 已按行排序的情况下可以直接复用，无需重排
//
// host 端需要：
//   - 创建全零张量 rowPtrs（长度 numRows+1）
//   - 启动 histogram kernel
//   - 启动 prefix sum kernel
//   - 返回 rowPtrs

#define BLOCK_SIZE 256
#define cdiv(a, b) (((a)+(b)-1)/(b))

/*
0 0 1 2 2 3 3 // rowIdx， < 总行数

0 2 1 2 2
0 2 3 5 7

0   2 3   5   7
 */ 
 

__global__ void compute_histogram_kernel(int nnz, int* rowIdx, int* rowPtrs) {
    // TODO: 每个线程处理一个非零元素，atomicAdd(&rowPtrs[rowIdx[i] + 1], 1)
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < nnz) {
        atomicAdd(&rowPtrs[rowIdx[i] + 1], 1);
    }
}

__global__ void exclusive_scan_kernel(int* rowPtrs, int numRows) {
    // TODO: 对 rowPtrs[0..numRows] 做 in-place exclusive prefix sum
    // 简单实现：Hillis-Steele 风格扫描（适用于 numRows+1 <= 1024 的情况）
    
    // numRows + 1 <= 1024
    int tid = threadIdx.x;
    
    for (int stride = 1; stride <= numRows; stride <<= 1) {
        int tmp = 0;
        if (tid <= numRows && tid - stride >= 0) tmp = rowPtrs[tid] + rowPtrs[tid - stride];
        __syncthreads();
        if (tid <= numRows && tid - stride >= 0) rowPtrs[tid] = tmp;
        __syncthreads();
    }
}

torch::Tensor coo_to_csr(torch::Tensor rowIdx, torch::Tensor colIdx, torch::Tensor values, int numRows)
{
    int nnz = rowIdx.size(0);
    auto rowPtrs = torch::zeros({numRows + 1}, torch::dtype(torch::kInt32).device(rowIdx.device()));
    
    int block_num = cdiv(nnz, BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    dim3 grid(block_num);
    
    compute_histogram_kernel<<<grid, block>>>(
        nnz, rowIdx.data_ptr<int>(), rowPtrs.data_ptr<int>()
    );
    
    exclusive_scan_kernel<<<1, 1024>>>(
        rowPtrs.data_ptr<int>(), numRows
    );
    
    return rowPtrs;
}
