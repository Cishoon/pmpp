#include <torch/extension.h>

// 朴素 3D 七点模板计算
// 输入 in: 扁平化的 N×N×N 3D 网格
// 输出 out: 扁平化的 N×N×N 3D 网格（仅内部点被更新）
//
// 七点模板公式：
//   out[i][j][k] = c0*in[i][j][k]
//               + c1*in[i][j][k-1] + c2*in[i][j][k+1]
//               + c3*in[i][j-1][k] + c4*in[i][j+1][k]
//               + c5*in[i-1][j][k] + c6*in[i+1][j][k]
//
// 每个线程负责一个输出元素
// 提示：
//   - i = blockIdx.z * blockDim.z + threadIdx.z
//   - j = blockIdx.y * blockDim.y + threadIdx.y
//   - k = blockIdx.x * blockDim.x + threadIdx.x
//   - 只有 i,j,k 在 [1, N-2] 范围内的线程才计算输出
//   - 3D 索引转 1D: idx = i*N*N + j*N + k

#define BLOCK_DIM 8

__global__ void Stencil3dBasicKernel(const float* in, float* out, int N,
                                      float c0, float c1, float c2,
                                      float c3, float c4, float c5, float c6) {
    // TODO: 计算线程的 3D 坐标 (i, j, k)
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    
    // TODO: 边界检查后计算七点模板
    if (i > 0 && i < N - 1 && j > 0 && j < N - 1 && k > 0 && k < N - 1) {
        out[i * N * N + j * N + k] = c0*in[i * N * N + j * N + k]
            + c1*in[i * N * N + j * N + k - 1] + c2*in[i * N * N + j * N + k + 1]
            + c3*in[i * N * N + (j - 1) * N + k] + c4*in[i * N * N + (j + 1) * N + k]
            + c5*in[(i - 1) * N * N + j * N + k] + c6*in[(i + 1) * N * N + j * N + k];
    }
}

torch::Tensor stencil3dBasic(torch::Tensor in, int N,
                              float c0, float c1, float c2,
                              float c3, float c4, float c5, float c6) {
    auto out = torch::zeros_like(in);

    dim3 block(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N + BLOCK_DIM - 1) / BLOCK_DIM,
              (N + BLOCK_DIM - 1) / BLOCK_DIM,
              (N + BLOCK_DIM - 1) / BLOCK_DIM);

    Stencil3dBasicKernel<<<grid, block>>>(
        in.data_ptr<float>(), out.data_ptr<float>(), N,
        c0, c1, c2, c3, c4, c5, c6);

    return out;
}
