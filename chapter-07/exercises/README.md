# 第七章 练习

## 结构

```
exercises/
├── kernels/
│   ├── conv2d_naive.cu          # 代码题A: 朴素 2D 卷积
│   └── conv2d_tiled.cu          # 代码题B: 分块 2D 卷积（常量内存）
├── answers.py                    # 习题1-7的计算/简答题
└── run_tests.py                  # 一键判题
```

## 使用方法

```bash
cd chapter-07/exercises
python run_tests.py
```
