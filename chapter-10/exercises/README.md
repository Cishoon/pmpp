# 第十章 练习

## 结构

```
exercises/
├── kernels/
│   ├── reduce_simple.cu           # 代码题A: 朴素归约（发散型）
│   ├── reduce_convergent.cu       # 代码题B: 收敛型归约
│   ├── reduce_shared_mem.cu       # 代码题C: 共享内存归约
│   ├── reduce_coarsened_sum.cu    # 代码题D: 线程粗化求和归约（任意长度）
│   └── reduce_coarsened_max.cu    # 代码题E: 线程粗化求最大值归约（任意长度）
├── answers.py                      # 习题1-6的计算/简答题
└── run_tests.py                    # 一键判题
```

## 使用方法

```bash
cd chapter-10/exercises
python run_tests.py
```
