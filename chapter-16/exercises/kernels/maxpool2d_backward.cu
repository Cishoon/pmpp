#include <torch/extension.h>

// MaxPool2D 反向传播内核
//
// 给定输出梯度和前向传播记录的最大值索引，计算输入梯度
//
// 输入:
//   grad_output: 4D float 张量 [batch_size, channels, out_h, out_w]
//   indices:     4D int32 张量 [batch_size, channels, out_h, out_w]，前向传播记录的最大值窗口内索引
//   batch_size, channels, height, width: 原始输入的维度
//   kernel_h, kernel_w, stride_h, stride_w: 池化参数
// 输出: 4D float 张量 [batch_size, channels, height, width]，即 dE/dX
//
// 思路：
//   - 每个线程负责一个输出位置 (b, c, h_out, w_out)
//   - 从 indices 中读取该位置最大值在窗口内的相对索引 max_idx
//   - 将 max_idx 还原为窗口内的 (kh, kw) 坐标
//   - 计算对应的输入位置 h_in = h_out*stride_h + kh, w_in = w_out*stride_w + kw
//   - 用 atomicAdd 将 grad_output 的值累加到 grad_input 的对应位置
//   - 需要原子操作是因为不同输出位置可能映射到同一个输入位置（当 stride < kernel_size 时）
//
// host 端需要：
//   - 计算输出尺寸 out_h, out_w
//   - 创建全零 grad_input 张量
//   - 配置 grid 为 (out_h * out_w, channels, batch_size)
//   - 启动内核并返回 grad_input

__global__ void maxpool2d_backward_kernel(
    const float* grad_output, const int* indices, float* grad_input,
    int batch_size, int channels, int height, int width,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int out_h, int out_w) {
    // TODO
}

// torch::Tensor maxpool2d_backward(torch::Tensor grad_output, torch::Tensor indices,
//                                   int height, int width,
//                                   int kernel_h, int kernel_w,
//                                   int stride_h, int stride_w);
