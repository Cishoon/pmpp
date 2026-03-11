// 习题 1a: 按行的矩阵乘法内核
// 每个线程负责计算输出矩阵的一整行
// M, N, P 都是 size x size 的方阵
//
// 提示：
// - 用 blockIdx.x, blockDim.x, threadIdx.x 算出当前线程负责的行号
// - 对该行的每一列，计算 M 的该行与 N 的该列的点积

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void matrixMulRowKernel(float* M, float* N, float* P, int size) {
    // TODO: 计算当前线程负责的行号 row
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    // TODO: 边界检查，然后对该行的每个列元素计算点积
    if (row < size) {
        for (int col = 0; col < size; col++) {
            float sum = 0;
            for (int j = 0; j < size; j++) {
                sum += M[row * size + j] * N[j * size + col];
            }            
            P[row * size + col] = sum; 
        }
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor matrixRowMul(torch::Tensor M, torch::Tensor N) {
    assert(M.device().type() == torch::kCUDA && N.device().type() == torch::kCUDA);
    assert(M.dtype() == torch::kFloat32 && N.dtype() == torch::kFloat32);
    assert(M.size(0) == M.size(1) && N.size(0) == N.size(1) && M.size(0) == N.size(0));

    const auto size = M.size(0);
    auto P = torch::empty_like(N);

    // TODO: 设置 dimBlock 和 dimGrid
    // 提示：这是一维的，每个线程处理一行
    dim3 dimBlock(16);
    dim3 dimGrid(cdiv(size, dimBlock.x));

    matrixMulRowKernel<<<dimGrid, dimBlock, 0, torch::cuda::getCurrentCUDAStream()>>>(
        M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(), size);

    return P;
}
