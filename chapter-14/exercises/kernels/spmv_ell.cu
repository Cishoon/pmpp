#include <torch/extension.h>

// ELL 格式稀疏矩阵-向量乘法 (SpMV)
//
// 输入:
//   colIdx:  1D int32 张量，长度 numRows * maxNnzPerRow，列优先存储的列索引（填充位为 -1）
//   values:  1D float 张量，长度 numRows * maxNnzPerRow，列优先存储的值（填充位为 0）
//   x:       1D float 张量，长度 numCols，输入向量
//   numRows: 矩阵行数
//   maxNnzPerRow: 每行最大非零元素数（所有行填充到此长度）
// 输出: 1D float 张量，长度 numRows，y = A * x
//
// 思路：
//   - 每个线程负责一行的计算
//   - ELL 格式使用列优先存储：第 t 个"列"中第 row 行的元素位于索引 t * numRows + row
//   - 遍历 t = 0 到 maxNnzPerRow-1，计算索引 i = t * numRows + row
//   - 如果 colIdx[i] >= 0（非填充），累加 values[i] * x[colIdx[i]]
//   - 列优先存储保证同一 warp 中相邻线程访问连续内存地址（内存合并）
//
// host 端需要：
//   - 创建输出张量 y（长度 numRows）
//   - 根据 numRows 配置 grid 和 block
//   - 启动内核并返回 y

#define BLOCK_SIZE 256
#define cdiv(a, b) (((a) + (b) - 1) / (b))

__global__ void spmv_ell_kernel(int* colIdx, float* values, float* x, float* y,
                                 int numRows, int maxNnzPerRow) {
    // TODO
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < numRows) {
        int end = numRows * maxNnzPerRow;
        float sum = 0.0f;
        for (int i = tid; i < end; i += numRows) {
            int col = colIdx[i];
            if (col == -1) break;
            float val = values[i];
            
            sum += x[col] * val;
        }
        y[tid] = sum;
    }
}

torch::Tensor spmv_ell(torch::Tensor colIdx, torch::Tensor values, torch::Tensor x, int numRows, int maxNnzPerRow) {
    auto y = torch::zeros({numRows}, x.options());
    
    dim3 block(BLOCK_SIZE);
    dim3 grid(cdiv(numRows, BLOCK_SIZE));
    
    spmv_ell_kernel<<<grid, block>>>(
        colIdx.data_ptr<int>(), 
        values.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        numRows, 
        maxNnzPerRow
    );
    
    return y;
}
