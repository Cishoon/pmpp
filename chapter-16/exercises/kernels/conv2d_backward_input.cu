#include <torch/extension.h>

// Conv2D 反向传播 - 输入梯度内核
//
// 给定输出梯度 grad_output，计算输入梯度 grad_input
//
// 输入:
//   weights:     4D float 张量 [out_channels, in_channels, kernel_h, kernel_w]
//   grad_output: 4D float 张量 [batch_size, out_channels, out_h, out_w]
//   batch_size, in_channels, height, width: 原始输入的维度
//   pad_h, pad_w, stride_h, stride_w: 卷积参数
// 输出: 4D float 张量 [batch_size, in_channels, height, width]，即 dE/dX
//
// 思路：
//   - 每个线程负责计算一个输入梯度元素 grad_input[b][c_in][h_in][w_in]
//   - 用 blockIdx 的三个维度分别映射 (h_in*width + w_in), c_in, b
//   - 遍历所有输出通道和卷积核位置
//   - 反推对应的输出位置 h_out, w_out，检查是否整除 stride 且在有效范围内
//   - 累加 grad_output[b][c_out][h_out][w_out] * weights[c_out][c_in][kh][kw]
//   - 将结果写入 grad_input
//
// host 端需要：
//   - 计算输出尺寸 out_h, out_w
//   - 创建全零 grad_input 张量
//   - 配置 grid 为 (height * width, in_channels, batch_size)
//   - 启动内核并返回 grad_input

__global__ void conv2d_backward_input_kernel(
    const float* weights, const float* grad_output, float* grad_input,
    int batch_size, int in_channels, int height, int width,
    int out_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int out_h, int out_w) {
    // TODO
}

// torch::Tensor conv2d_backward_input(torch::Tensor weights, torch::Tensor grad_output,
//                                      int height, int width, int pad_h, int pad_w,
//                                      int stride_h, int stride_w);
