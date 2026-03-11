# 第十三章 练习

## 结构

```
exercises/
├── kernels/
│   ├── radix_sort_naive.cu       # 代码题A: 朴素并行基数排序（三内核：提取bit + 前缀和 + scatter）
│   ├── radix_sort_coalesced.cu   # 代码题B: 内存合并基数排序（共享内存局部扫描 + 合并写入）
│   ├── radix_sort_multibit.cu    # 代码题C: 多位基数排序（多bit + 直方图 + 合并写入）
│   └── merge_sort.cu            # 代码题D: 并行归并排序（co-rank + 自底向上归并）
└── run_tests.py                  # 一键判题
```

## 使用方法

```bash
cd chapter-13/exercises
python run_tests.py
```
