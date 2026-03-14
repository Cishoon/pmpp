#include <torch/extension.h>

// 习题 10：分块转置内核
// 对矩阵 A 中的每个 BLOCK_WIDTH x BLOCK_WIDTH 分块进行原地转置
//
// 提示：
//   - 使用 __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH] 作为缓冲区
//   - 先将全局内存数据加载到共享内存
//   - 加入 __syncthreads() 确保所有线程完成加载
//   - 再将转置后的数据写回全局内存

#define BLOCK_WIDTH 16

__global__ void BlockTransposeKernel(float* A, int A_width, int A_height) {
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];
    int baseIdx = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_WIDTH + threadIdx.y) * A_width;
    
    blockA[threadIdx.x][threadIdx.y] = A[baseIdx];
    __syncthreads();
    A[baseIdx] = blockA[threadIdx.y][threadIdx.x];
}

torch::Tensor blockTranspose(torch::Tensor A) {
    TORCH_CHECK(A.dim() == 2, "Input must be 2D");
    int H = A.size(0), W = A.size(1);
    TORCH_CHECK(H % BLOCK_WIDTH == 0 && W % BLOCK_WIDTH == 0,
                "Matrix dims must be multiples of BLOCK_WIDTH");

    auto out = A.clone();

    dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 grid(W / BLOCK_WIDTH, H / BLOCK_WIDTH);

    BlockTransposeKernel<<<grid, block>>>(out.data_ptr<float>(), W, H);

    return out;
}
