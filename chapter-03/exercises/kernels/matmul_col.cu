// 习题 1b: 按列的矩阵乘法内核
// 每个线程负责计算输出矩阵的一整列
// M, N, P 都是 size x size 的方阵
//
// 提示：
// - 用 blockIdx.x, blockDim.x, threadIdx.x 算出当前线程负责的列号
// - 对该列的每一行，计算 M 的该行与 N 的该列的点积

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void matrixMulColKernel(float* M, float* N, float* P, int size) {
    // TODO: 计算当前线程负责的列号 col
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: 边界检查，然后对该列的每个行元素计算点积
    if (col < size) {
        for (int row = 0; row < size; row++) {
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

torch::Tensor matrixColMul(torch::Tensor M, torch::Tensor N) {
    assert(M.device().type() == torch::kCUDA && N.device().type() == torch::kCUDA);
    assert(M.dtype() == torch::kFloat32 && N.dtype() == torch::kFloat32);
    assert(M.size(0) == M.size(1) && N.size(0) == N.size(1) && M.size(0) == N.size(0));

    const auto size = M.size(0);
    auto P = torch::empty_like(N);

    dim3 dimBlock(16);
    dim3 dimGrid(cdiv(size, dimBlock.x));

    matrixMulColKernel<<<dimGrid, dimBlock, 0, torch::cuda::getCurrentCUDAStream()>>>(
        M.data_ptr<float>(), N.data_ptr<float>(), P.data_ptr<float>(), size);

    return P;
}
