#include <torch/extension.h>

// Conv2D 前向传播内核
//
// 输入:
//   input:   4D float 张量 [batch_size, in_channels, height, width]
//   weights: 4D float 张量 [out_channels, in_channels, kernel_h, kernel_w]
//   bias:    1D float 张量 [out_channels]
//   pad_h, pad_w:     填充大小
//   stride_h, stride_w: 步幅大小
// 输出: 4D float 张量 [batch_size, out_channels, out_h, out_w]
//   其中 out_h = (height + 2*pad_h - kernel_h) / stride_h + 1
//        out_w = (width + 2*pad_w - kernel_w) / stride_w + 1
//
// 思路：
//   - 每个线程负责计算一个输出元素 output[b][c_out][h_out][w_out]
//   - 用 blockIdx 的三个维度分别映射 (h_out*out_w + w_out), c_out, b
//   - 初始化累加值为 bias[c_out]
//   - 遍历所有输入通道和卷积核位置，计算对应输入坐标
//   - 对有效输入位置（边界检查）累加 input * weight
//   - 将结果写入输出张量
//
// host 端需要：
//   - 计算输出尺寸 out_h, out_w
//   - 创建输出张量
//   - 配置 grid 为 (out_h * out_w, out_channels, batch_size)
//   - 启动内核并返回输出

__global__ void conv2d_forward_kernel(
    const float* input, const float* weights, const float* bias, float* output,
    int batch_size, int in_channels, int height, int width,
    int out_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int out_h, int out_w) {
    
    
}

// torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weights, torch::Tensor bias,
//                               int pad_h, int pad_w, int stride_h, int stride_w);
