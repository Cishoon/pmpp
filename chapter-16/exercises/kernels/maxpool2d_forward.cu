#include <torch/extension.h>
#include <float.h>

// MaxPool2D 前向传播内核
//
// 输入:
//   input:    4D float 张量 [batch_size, channels, height, width]
//   kernel_h, kernel_w: 池化窗口大小
//   stride_h, stride_w: 步幅大小
// 输出:
//   output:  4D float 张量 [batch_size, channels, out_h, out_w]，池化结果
//   indices: 4D int32 张量 [batch_size, channels, out_h, out_w]，最大值在窗口内的相对索引
//   其中 out_h = (height - kernel_h) / stride_h + 1
//        out_w = (width - kernel_w) / stride_w + 1
//
// 思路：
//   - 每个线程负责一个输出元素 output[b][c][h_out][w_out]
//   - 用 blockIdx 的三个维度分别映射 (h_out*out_w + w_out), c, b
//   - 初始化最大值为 -FLT_MAX，最大值索引为 -1
//   - 遍历池化窗口内的所有位置 (kh, kw)
//   - 计算输入位置 h_in = h_out*stride_h + kh, w_in = w_out*stride_w + kw
//   - 更新最大值和对应的窗口内相对索引 (kh * kernel_w + kw)
//   - 将最大值写入 output，索引写入 indices（反向传播需要）
//
// host 端需要：
//   - 计算输出尺寸 out_h, out_w
//   - 创建 output 和 indices 张量
//   - 配置 grid 为 (out_h * out_w, channels, batch_size)
//   - 启动内核并返回 output 和 indices

__global__ void maxpool2d_forward_kernel(
    const float* input, float* output, int* indices,
    int batch_size, int channels, int height, int width,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int out_h, int out_w) {
    // TODO
}

// std::vector<torch::Tensor> maxpool2d_forward(torch::Tensor input,
//                                               int kernel_h, int kernel_w,
//                                               int stride_h, int stride_w);
