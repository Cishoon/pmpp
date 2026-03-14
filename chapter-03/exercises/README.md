# 第三章 练习

## 结构

```
exercises/
├── kernels/                  # 你需要填写的 CUDA 内核代码
│   ├── matmul_row.cu         # 习题1a: 按行的矩阵乘法内核
│   ├── matmul_col.cu         # 习题1b: 按列的矩阵乘法内核
│   ├── mat_vec_mul.cu        # 习题2: 矩阵-向量乘法内核
│   ├── rgb_to_grayscale.cu   # 附加题: RGB转灰度内核
│   └── gaussian_blur.cu      # 附加题: 高斯模糊内核
├── answers.py                # 习题3-5的选择题/简答题，在这里作答
└── run_tests.py              # 一键运行判题
```

## 使用方法

1. 填写 `kernels/` 下的 `.cu` 文件中标记 `// TODO` 的部分
2. 填写 `answers.py` 中的答案
3. 运行判题：

```bash
cd chapter-03/exercises
python run_tests.py
```

祝你好运！
