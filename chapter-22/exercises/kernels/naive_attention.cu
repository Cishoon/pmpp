#include <torch/extension.h>
#include <cmath>

// 朴素 Attention 内核（三步法）
//
// 输入:
//   Q: (N, d) 查询矩阵
//   K: (N, d) 键矩阵
//   V: (N, d) 值矩阵
//
// 输出:
//   O: (N, d) = softmax(Q @ K^T / sqrt(d)) @ V
//
// 实现步骤：
//   1. 计算 S = Q @ K^T / sqrt(d)，形状 (N, N)
//      - 每个线程计算 S 的一个元素 S[row][col]
//      - S[row][col] = sum_k(Q[row][k] * K[col][k]) / sqrt(d)
//
//   2. 对 S 的每一行做 softmax：
//      - 先求每行最大值 m[row] = max_j(S[row][j])（数值稳定性）
//      - 再求 P[row][col] = exp(S[row][col] - m[row]) / sum_j(exp(S[row][j] - m[row]))
//
//   3. 计算 O = P @ V
//      - O[row][k] = sum_j(P[row][j] * V[j][k])
//
// 限制：这是最朴素的实现，需要 O(N²) 额外内存存储 S/P 矩阵
//
// host 端需要：
//   - 分配 S 矩阵 (N, N)
//   - 分别启动三个内核（或一个融合内核）
//   - grid/block 配置需覆盖所有 (row, col) 对

// TODO: 实现 ScaledDotProductKernel
// 计算 S[row][col] = sum_k(Q[row][k] * K[col][k]) / sqrt(d)
__global__ void ScaledDotProductKernel(
    const float* Q, const float* K, float* S,
    int N, int d
) {
    // TODO: 你的代码
}

// TODO: 实现 SoftmaxKernel
// 对 S 的每一行做 softmax（带数值稳定性处理）
// 每个 block 处理一行，使用共享内存做归约求 max 和 sum
__global__ void SoftmaxKernel(
    float* S, float* P,
    int N
) {
    // TODO: 你的代码
}

// TODO: 实现 PVMultiplyKernel
// 计算 O[row][k] = sum_j(P[row][j] * V[j][k])
__global__ void PVMultiplyKernel(
    const float* P, const float* V, float* O,
    int N, int d
) {
    // TODO: 你的代码
}

// TODO: 实现 host 函数
// 函数签名: torch::Tensor naiveAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
//
// 步骤：
//   1. 获取 N, d
//   2. 分配 S(N,N), P(N,N), O(N,d)
//   3. 启动 ScaledDotProductKernel: grid(ceil(N/16), ceil(N/16)), block(16, 16)
//   4. 启动 SoftmaxKernel: grid(N), block(min(N, 1024))
//   5. 启动 PVMultiplyKernel: grid(ceil(N/16), ceil(d/16)), block(16, 16)
//   6. 返回 O
torch::Tensor naiveAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // Q [N, d] 
    int N = Q.size(0);
    int d = Q.size(1);
    
    // S = Q @ K^T
    auto S = torch::zeros({N, N}, Q.options());
    
    float* d_Q = Q.data_ptr<float>();
    float* d_K = K.data_ptr<float>();
    float* d_S = S.data_ptr<float>();
    
    dim3 grid1();
    dim3 block1()
    ScaledDotProductKernel<<<grid1, block1>>>(
        d_Q, d_K, d_S, N, d
    );
    
    return torch::zeros_like(Q);
}
