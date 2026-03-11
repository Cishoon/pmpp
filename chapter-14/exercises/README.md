# 第十四章 练习

## 结构

```
exercises/
├── kernels/
│   ├── spmv_coo.cu          # 代码题A: COO 格式 SpMV（原子操作累加）
│   ├── spmv_csr.cu          # 代码题B: CSR 格式 SpMV（每线程一行）
│   ├── spmv_ell.cu          # 代码题C: ELL 格式 SpMV（列优先填充访问）
│   └── spmv_jds.cu          # 代码题D: JDS 格式 SpMV（排序行 + 迭代层）
└── run_tests.py              # 一键判题
```

## 使用方法

```bash
cd chapter-14/exercises
python run_tests.py
```
