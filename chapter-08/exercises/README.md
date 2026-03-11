# 第八章 练习

## 结构

```
exercises/
├── kernels/
│   ├── stencil_basic.cu             # 代码题A: 朴素 3D 模板计算
│   ├── stencil_shared_mem.cu        # 代码题B: 共享内存 3D 模板计算
│   ├── stencil_thread_coarsening.cu # 代码题C: 线程粗化 3D 模板计算
│   └── stencil_register_tiling.cu   # 代码题D: 寄存器分块 3D 模板计算
├── answers.py                        # 习题1-2的计算/简答题
└── run_tests.py                      # 一键判题
```

## 使用方法

```bash
cd chapter-08/exercises
python run_tests.py
```
