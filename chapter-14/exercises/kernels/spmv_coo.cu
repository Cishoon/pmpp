#include <torch/extension.h>

// COO 格式稀疏矩阵-向量乘法 (SpMV)
//
// 输入:
//   rowIdx: 1D int32 张量，长度 nnz，每个非零元素的行索引
//   colIdx: 1D int32 张量，长度 nnz，每个非零元素的列索引
//   values: 1D float 张量，长度 nnz，每个非零元素的值
//   x:      1D float 张量，长度 numCols，输入向量
//   numRows: 输出向量的行数
// 输出: 1D float 张量，长度 numRows，y = A * x
//
// 思路：
//   - 每个线程处理一个非零元素
//   - 读取该元素的行索引 row、列索引 col 和值 val
//   - 计算 val * x[col]，用 atomicAdd 累加到 y[row]
//   - 需要原子操作是因为同一行的多个非零元素可能被不同线程处理
//
// host 端需要：
//   - 创建全零输出张量 y（长度 numRows）
//   - 根据 nnz 配置 grid 和 block
//   - 启动内核并返回 y

#define BLOCK_SIZE 256

__global__ void spmv_coo_kernel(int nnz, int* rowIdx, int* colIdx, float* values, float* x, float* y) {
    // TODO
}

// 函数签名: torch::Tensor spmv_coo(torch::Tensor rowIdx, torch::Tensor colIdx, torch::Tensor values, torch::Tensor x, int numRows)
