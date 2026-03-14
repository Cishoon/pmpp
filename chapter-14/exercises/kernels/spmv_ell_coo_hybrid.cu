#include <torch/extension.h>

// Hybrid ELL-COO 格式稀疏矩阵-向量乘法 (SpMV)
//
// 输入:
//   ell_colIdx:  1D int32 张量，长度 numRows * maxNnzPerRow，ELL 部分列优先存储的列索引（填充位为 -1）
//   ell_values:  1D float 张量，长度 numRows * maxNnzPerRow，ELL 部分列优先存储的值（填充位为 0）
//   coo_rowIdx:  1D int32 张量，长度 coo_nnz，COO 溢出部分的行索引
//   coo_colIdx:  1D int32 张量，长度 coo_nnz，COO 溢出部分的列索引
//   coo_values:  1D float 张量，长度 coo_nnz，COO 溢出部分的值
//   x:           1D float 张量，长度 numCols，输入向量
//   numRows:     矩阵行数
//   maxNnzPerRow: ELL 部分每行最大非零元素数
// 输出: 1D float 张量，长度 numRows，y = A * x
//
// 思路：
//   - Hybrid ELL-COO 将稀疏矩阵分为两部分：
//     1. ELL 部分：每行最多 maxNnzPerRow 个非零元素，列优先存储，适合 GPU 并行
//     2. COO 部分：超出 maxNnzPerRow 的溢出元素，用 COO 格式存储
//   - ELL kernel：每个线程处理一行，遍历 t=0..maxNnzPerRow-1，索引 i = t*numRows+row
//     如果 colIdx[i] >= 0，累加 values[i] * x[colIdx[i]]
//   - COO kernel：每个线程处理一个溢出非零元素，用 atomicAdd 累加到 y[row]
//   - 先启动 ELL kernel 写入 y，再启动 COO kernel 用原子操作累加溢出部分
//
// host 端需要：
//   - 创建全零输出张量 y（长度 numRows）
//   - 分别配置并启动 ELL kernel 和 COO kernel
//   - 返回 y

#define BLOCK_SIZE 256
#define cdiv(a, b) (((a) + (b) - 1) / (b))

__global__ void spmv_ell_part_kernel(int* colIdx, float* values, float* x, float* y,
                                      int numRows, int maxNnzPerRow) {
    // TODO: ELL 部分 — 每个线程处理一行
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < numRows) {
        int end = numRows * maxNnzPerRow;
        float sum = 0.0f;
        for (int i = row; i < end; i += numRows) {
            int col = colIdx[i];
            if (col == -1) break; 
            float val = values[i];
            sum += val * x[col];
        }
        y[row] = sum;
    }
}

__global__ void spmv_coo_part_kernel(int coo_nnz, int* rowIdx, int* colIdx, float* values,
                                      float* x, float* y) {
    // TODO: COO 溢出部分 — 每个线程处理一个非零元素，atomicAdd 累加
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < coo_nnz) {
        int row = rowIdx[idx];
        int col = colIdx[idx];
        float val = values[idx];
        atomicAdd(&y[row], val * x[col]);
    }
}

torch::Tensor spmv_ell_coo_hybrid(torch::Tensor ell_colIdx, torch::Tensor ell_values, torch::Tensor coo_rowIdx, torch::Tensor coo_colIdx, torch::Tensor coo_values, torch::Tensor x, int numRows, int maxNnzPerRow)
{
    auto y = torch::zeros({numRows}, x.options());
    
    dim3 block(BLOCK_SIZE);
    dim3 grid_ell(cdiv(numRows, BLOCK_SIZE));
    
    // 启动 ELL kernel
    spmv_ell_part_kernel<<<grid_ell, block>>>(
        ell_colIdx.data_ptr<int>(),
        ell_values.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        numRows,
        maxNnzPerRow
    );
    
    // 启动 COO kernel
    int coo_nnz = coo_rowIdx.size(0);
    dim3 grid_coo(cdiv(coo_nnz, BLOCK_SIZE));
    spmv_coo_part_kernel<<<grid_coo, block>>>(
        coo_nnz,
        coo_rowIdx.data_ptr<int>(),
        coo_colIdx.data_ptr<int>(),
        coo_values.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>()
    );
    
    return y;
}
