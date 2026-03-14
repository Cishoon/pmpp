# 第九章 练习

## 结构

```
exercises/
├── kernels/
│   ├── histo_basic.cu                # 代码题A: 朴素直方图
│   ├── histo_shared_mem.cu           # 代码题B: 共享内存私有化直方图
│   ├── histo_coarsening.cu           # 代码题C: 线程粗化直方图
│   └── histo_coarsening_coalesced.cu # 代码题D: 线程粗化 + 合并访存直方图
├── answers.py                         # 习题1-6的计算/简答题
└── run_tests.py                       # 一键判题
```

## 使用方法

```bash
cd chapter-09/exercises
python run_tests.py
```
