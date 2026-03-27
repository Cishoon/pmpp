# 第二十二章 练习

## 结构

```
exercises/
├── kernels/
│   ├── naive_attention.cu              # 代码题A: 朴素 Attention（三步法）
│   ├── flash_attention_1pass.cu        # 代码题B: FlashAttention 单 block（1-pass 在线 softmax）
│   ├── flash_attention_tiled.cu        # 代码题C: FlashAttention 分块版（多 block）
│   └── flash_attention_causal.cu       # 代码题D: FlashAttention 因果掩码版
├── reference_kernels.cu                  # GPU 参考实现（生成 ground truth，不要修改）
├── test_harness.cu                      # 测试框架（正确性验证 + 计时）
├── Makefile                             # 编译脚本
├── answers.py                           # 习题1-7的计算/简答题
└── run_tests.py                         # 一键判题
```

## 使用方法

### 一键判题（推荐）

```bash
cd chapter-22/exercises
python run_tests.py
```

自动编译所有内核、运行正确性测试和性能对比、判简答题。

### 逐题调试

```bash
# 只编译和测试某一题
make test_naive   && ./test_naive
make test_1pass   && ./test_1pass
make test_tiled   && ./test_tiled
make test_causal  && ./test_causal

# 编译全部
make all && ./test_all
```

### 清理

```bash
make clean
```

## 内核文件说明

每个 `.cu` 文件包含：
- `__global__` 内核函数（你需要实现）
- `launch_xxx()` host 函数（你需要实现：配置 grid/block、分配临时显存、启动内核）

测试框架 `test_harness.cu` + `reference_kernels.cu` 负责：
- 生成随机输入
- 用 GPU 参考实现（FlashAttention）计算期望输出
- 调用你的 `launch_xxx()` 函数
- 拷回 host 逐元素对比
- 使用 CUDA Event 计时
