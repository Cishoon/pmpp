#include <torch/extension.h>

// JDS 格式稀疏矩阵-向量乘法 (SpMV)
//
// 输入:
//   colIdx:  1D int32 张量，长度 nnz，按 JDS 格式排列的列索引
//   values:  1D float 张量，长度 nnz，按 JDS 格式排列的值
//   rowPerm: 1D int32 张量，长度 numRows，行排列映射（排序后行号 → 原始行号）
//   iterPtr: 1D int32 张量，长度 numTiles+1，每个迭代层的起始位置
//   x:       1D float 张量，长度 numCols，输入向量
//   numRows: 矩阵行数
//   numTiles: 迭代层数（等于最长行的非零元素数）
// 输出: 1D float 张量，长度 numRows，y = A * x
//
// 思路：
//   - JDS 格式先按行非零元素数量降序排列行，然后按"迭代层"组织数据
//   - 每个线程负责一个排序后的行（tid 对应排序后的行号）
//   - 遍历所有迭代层 t = 0 到 numTiles-1：
//     - 计算索引 i = iterPtr[t] + tid
//     - 如果 i < iterPtr[t+1]，说明该行在这一层有非零元素
//     - 累加 values[i] * x[colIdx[i]]
//   - 最终结果写入 y[rowPerm[tid]]，通过 rowPerm 恢复原始行顺序
//   - JDS 的优势：排序后相邻线程处理的行长度相近，减少 warp 分化
//
// host 端需要：
//   - 创建输出张量 y（长度 numRows）
//   - 根据 numRows 配置 grid 和 block
//   - 启动内核并返回 y

#define BLOCK_SIZE 256

__global__ void spmv_jds_kernel(int* colIdx, float* values, int* rowPerm, int* iterPtr,
                                 float* x, float* y, int numRows, int numTiles) {
    // TODO
}

// 函数签名: torch::Tensor spmv_jds(torch::Tensor colIdx, torch::Tensor values, torch::Tensor rowPerm, torch::Tensor iterPtr, torch::Tensor x, int numRows, int numTiles)
