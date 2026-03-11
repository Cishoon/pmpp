// 习题 2: 矩阵-向量乘法内核
// B 是 matrix_rows x vector_size 的矩阵，c 是长度为 vector_size 的向量
// result[i] = sum_j( B[i][j] * c[j] )
//
// 提示：
// - 每个线程计算输出向量的一个元素
// - 用 blockIdx.x, blockDim.x, threadIdx.x 算出当前线程负责的行号 i

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void matrixVecMulKernel(float* B, float* c, float* result, int vector_size, int matrix_rows) {
    // TODO: 计算当前线程负责的行号 i
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // TODO: 边界检查，然后计算 B 的第 i 行与向量 c 的点积
    if (i < matrix_rows) {
        float sum = 0;
        for (int j = 0; j < vector_size; j++) {
            sum += B[i * vector_size + j] * c[j];
        }
        result[i] = sum;
    }
}

torch::Tensor matrix_vector_multiplication(torch::Tensor B, torch::Tensor c) {
    assert(B.device().type() == torch::kCUDA && c.device().type() == torch::kCUDA);
    assert(B.dtype() == torch::kFloat32 && c.dtype() == torch::kFloat32);
    assert(B.size(1) == c.size(0));

    int vector_size = c.size(0);
    int matrix_rows = B.size(0);

    auto result = torch::empty({matrix_rows}, torch::TensorOptions().dtype(torch::kFloat32).device(B.device()));

    int threads_per_block = 16;
    int number_of_blocks = (matrix_rows + threads_per_block - 1) / threads_per_block;

    matrixVecMulKernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        B.data_ptr<float>(), c.data_ptr<float>(), result.data_ptr<float>(), vector_size, matrix_rows);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
