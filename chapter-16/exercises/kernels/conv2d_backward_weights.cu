#include <torch/extension.h>

// Conv2D 反向传播 - 权重梯度内核
//
// 给定输入和输出梯度，计算权重梯度 grad_weights 和偏置梯度 grad_bias
//
// 输入:
//   input:       4D float 张量 [batch_size, in_channels, height, width]
//   grad_output: 4D float 张量 [batch_size, out_channels, out_h, out_w]
//   kernel_h, kernel_w: 卷积核大小
//   pad_h, pad_w, stride_h, stride_w: 卷积参数
// 输出:
//   grad_weights: 4D float 张量 [out_channels, in_channels, kernel_h, kernel_w]，即 dE/dW
//   grad_bias:    1D float 张量 [out_channels]，即 dE/db
//
// 思路（权重梯度内核）：
//   - 每个线程负责计算一个权重梯度元素 grad_weights[c_out][c_in][kh][kw]
//   - 用 blockIdx 的三个维度分别映射 (kh*kernel_w + kw), c_in, c_out
//   - 遍历所有 batch 和输出位置，累加 input * grad_output 的对应乘积
//   - 需要根据 stride 和 padding 计算正确的输入位置
//
// 思路（偏置梯度内核）：
//   - 每个线程负责一个输出通道的偏置梯度
//   - 对该通道所有 batch 和空间位置的 grad_output 求和
//
// host 端需要：
//   - 计算输出尺寸 out_h, out_w
//   - 创建全零 grad_weights 和 grad_bias 张量
//   - 分别配置并启动权重梯度内核和偏置梯度内核
//   - 返回 grad_weights 和 grad_bias

__global__ void conv2d_backward_weights_kernel(
    const float* input, const float* grad_output, float* grad_weights,
    int batch_size, int in_channels, int height, int width,
    int out_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int out_h, int out_w) {
    // TODO
}

__global__ void conv2d_backward_bias_kernel(
    const float* grad_output, float* grad_bias,
    int batch_size, int out_channels, int out_h, int out_w) {
    // TODO
}

// std::vector<torch::Tensor> conv2d_backward_weights(torch::Tensor input, torch::Tensor grad_output,
//                                                     int kernel_h, int kernel_w,
//                                                     int pad_h, int pad_w,
//                                                     int stride_h, int stride_w);
