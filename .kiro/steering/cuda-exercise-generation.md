---
inclusion: manual
---

# CUDA 章节出题规范（Skills 文档）

本文档描述了为 CUDA 编程教材各章节生成练习题、填空内核和自动判题代码的标准流程与规范。

## 目录结构规范

每章练习统一放在 `chapter-XX/exercises/` 下：

```
chapter-XX/
├── README_zh.md              # 章节中文翻译（含完整解答）
└── exercises/
    ├── README.md             # 练习说明（中文）
    ├── kernels/              # 学生需要填写的 CUDA 内核（.cu 文件）
    │   └── *.cu
    └── run_tests.py          # 自动判题脚本（含正确性验证 + 性能对比）
```

注意：**不创建 answers.py**，不出简答题/计算题，只出代码题。

## README_zh.md 翻译规范

1. 完整翻译原版 README.md，保留所有代码块和图片引用
2. 技术术语保留英文原词并附中文说明，例如：`__syncthreads()`、`shared memory（共享内存）`
3. 数学公式和推导过程完整保留，用中文解释步骤
4. 习题解答全部给出，作为参考答案

## 内核填空规范（kernels/*.cu）

### 文件结构

```cuda
#include <torch/extension.h>

// [功能描述]
// [与其他版本的区别（如有）]
//
// 思路：
//   - 步骤1的算法思路描述
//   - 步骤2的算法思路描述
//   - ...
//
// 注意：思路只描述算法方向，不提供具体代码

#define BLOCK_SIZE 256

__global__ void XxxKernel(...) {
    // TODO
}

// 函数签名（学生需要自行实现 host 端逻辑）
// torch::Tensor xxx(torch::Tensor input, ...);
```

### 关键规则

1. **kernel 函数内部不给任何注释**，只留一行 `// TODO`
2. **思路写在文件顶部注释块**，描述算法方向和关键步骤，但不提供具体代码
3. **不提供 host 函数实现**，只给出函数签名（注释形式），学生需要自行编写 host 端的张量创建、grid/block 配置、内核启动等逻辑
4. 宏定义（如 `BLOCK_SIZE`、`COARSE_FACTOR`）可以保留
5. 必要的 `#include` 保留

## run_tests.py 规范

### 必须包含的部分

1. **正确性验证**：对每个内核用多组测试用例验证结果
2. **性能对比**：所有内核在相同数据规模下的耗时对比，显示加速比和柱状图

### 代码题测试模板

正确性验证通过后，对该用例进行计时并在 PASS 行显示耗时：

```python
def test_correctness(name, fn, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    for params in test_cases:
        result = fn(...)
        expected = reference(...)
        ok = torch.allclose(result, expected, ...)
        if not ok:
            report(f"描述", False, f"详情")
        else:
            t = benchmark_fn(fn, data, warmup=10)
            report(f"描述  {t:.4f} ms", True)
```

### 性能对比模板

```python
def run_benchmark(exts):
    configs = [
        (size1, "描述1"),
        (size2, "描述2"),
    ]
    for size, title in configs:
        print(colored(f"\n性能对比 {title}", "bold"))
        data = ...
        results = {}
        for label, fn in exts:
            t = benchmark_fn(fn, data)
            results[label] = t
        # 显示耗时、加速比、柱状图
```

### 必须包含的基础设施

- `colored()` 函数：终端彩色输出
- `report()` 函数：统一记分和输出
- `compile_kernel()` 函数：使用 `torch.utils.cpp_extension.load_inline` 编译
- `benchmark_fn()` 函数：GPU 计时（cuda events）
- `run_benchmark()` 函数：多内核性能对比
- `main()` 函数：CUDA 可用性检查 + 正确性测试 + 性能对比 + 汇总得分

## 各章节出题要点

### 第三章（并行计算基础）
- 代码题：矩阵乘法（行/列线程映射）、矩阵向量乘法、图像处理内核

### 第五章（共享内存与分块）
- 代码题：分块矩阵乘法（核心算法）、分块转置（同步问题）

### 第六章（线程粗化）
- 代码题：线程粗化矩阵乘法

### 第七章（卷积）
- 代码题：2D 卷积内核（基础版/分块版）

### 第八章（模板计算/Stencil）
- 代码题：3D 模板计算内核

### 第九章（直方图）
- 代码题：原子操作直方图、私有化直方图

### 第十章（归约）
- 代码题：并行归约（求和/最大值）

### 第十一章（前缀扫描）
- 代码题：Kogge-Stone 扫描、分层扫描

## 判题脚本运行方式

```bash
cd chapter-XX/exercises
python run_tests.py
```

输出格式：
```
===================================================
  第XX章 CUDA 编程练习 - 正确性验证 & 性能对比
===================================================

[代码题] 功能名称
  ✓ PASS  测试用例1  0.0123 ms
  ✗ FAIL  测试用例2  (max diff=0.001234)

性能对比 10M 元素
  A 朴素       0.234 ms  ████████████████████████████
  B 优化       0.089 ms  ██████████  (2.63x)

===================================================
  总分: 8/10 (80%)
===================================================
```
