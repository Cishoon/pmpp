# 第十二章 练习

## 结构

```
exercises/
├── kernels/
│   ├── merge_basic.cu       # 代码题A: 基础并行归并（co-rank + 顺序归并）
│   └── merge_tiled.cu       # 代码题B: 分块并行归并（共享内存优化）
└── run_tests.py              # 一键判题
```

## 使用方法

```bash
cd chapter-12/exercises
python run_tests.py
```
