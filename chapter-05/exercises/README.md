# 第五章 练习

## 结构

```
exercises/
├── kernels/                        # 你需要填写的 CUDA 内核代码
│   ├── tiled_matmul.cu             # 习题代码题: 分块矩阵乘法内核
│   └── block_transpose.cu         # 习题10: 分块转置内核
├── answers.py                      # 习题1-9、11-12的简答/计算题，在这里作答
└── run_tests.py                    # 一键运行判题
```

## 使用方法

1. 填写 `kernels/` 下的 `.cu` 文件中标记 `// TODO` 的部分
2. 填写 `answers.py` 中的答案
3. 运行判题：

```bash
cd chapter-05/exercises
python run_tests.py
```

祝你好运！
