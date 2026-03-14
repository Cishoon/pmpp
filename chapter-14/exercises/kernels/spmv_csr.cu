#include <torch/extension.h>

// CSR 格式稀疏矩阵-向量乘法 (SpMV)
//
// 输入:
//   rowPtrs: 1D int32 张量，长度 numRows+1，每行非零元素在 colIdx/values 中的起始位置
//   colIdx:  1D int32 张量，长度 nnz，每个非零元素的列索引
//   values:  1D float 张量，长度 nnz，每个非零元素的值
//   x:       1D float 张量，长度 numCols，输入向量
//   numRows: 矩阵行数
// 输出: 1D float 张量，长度 numRows，y = A * x
//
// 思路：
//   - 每个线程负责一行的计算
//   - 通过 rowPtrs[row] 和 rowPtrs[row+1] 确定该行非零元素的范围
//   - 遍历该范围内的所有非零元素，累加 values[i] * x[colIdx[i]]
//   - 将结果写入 y[row]
//   - 不需要原子操作，因为每行只有一个线程写入
//
// host 端需要：
//   - 创建输出张量 y（长度 numRows）
//   - 根据 numRows 配置 grid 和 block
//   - 启动内核并返回 y

#define BLOCK_SIZE 256
#define cdiv(a, b) (((a) + (b) - 1) / (b))

__global__ void spmv_csr_kernel(int* rowPtrs, int* colIdx, float* values, float* x, float* y, int numRows, int num_row_ptrs) {
    // TODO
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    
    if (tid < num_row_ptrs - 1) {
        float sum = 0.0f;
        for (int i = rowPtrs[tid]; i < rowPtrs[tid + 1]; i++) {
            int col = colIdx[i];
            float val = values[i];
            
            sum += val * x[col];
        }
        y[tid] = sum;
    }
}

torch::Tensor spmv_csr(torch::Tensor rowPtrs, torch::Tensor colIdx, torch::Tensor values, torch::Tensor x, int numRows) {
    int num_row_ptrs = rowPtrs.size(0);
    
    auto y = torch::zeros({numRows}, x.options());
    
    dim3 block(BLOCK_SIZE);
    dim3 grid(cdiv(num_row_ptrs, BLOCK_SIZE));
    
    spmv_csr_kernel<<<grid, block>>>(
        rowPtrs.data_ptr<int>(), 
        colIdx.data_ptr<int>(),
        values.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        numRows, num_row_ptrs
    );
    
    return y;
}
