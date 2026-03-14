# 第六章 练习

## 结构

```
exercises/
├── kernels/
│   ├── col_major_matmul.cu         # 习题1: 列主序 N 的分块矩阵乘法（Corner Turning）
│   └── coarsened_matmul.cu         # 代码题: 线程粗化分块矩阵乘法
├── answers.py                       # 习题2-4的简答/计算题
└── run_tests.py                     # 一键判题
```

## 使用方法

1. 填写 `kernels/` 下 `.cu` 文件中标记 `// TODO` 的部分
2. 填写 `answers.py` 中的答案
3. 运行判题：

```bash
cd chapter-06/exercises
python run_tests.py
```
