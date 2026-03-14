# 第十四章 练习

## 结构

```
exercises/
├── kernels/
│   ├── spmv_coo.cu              # 代码题A: COO 格式 SpMV（原子操作累加）
│   ├── spmv_csr.cu              # 代码题B: CSR 格式 SpMV（每线程一行）
│   ├── spmv_ell.cu              # 代码题C: ELL 格式 SpMV（列优先填充访问）
│   ├── spmv_jds.cu              # 代码题D: JDS 格式 SpMV（排序行 + 迭代层）
│   ├── spmv_ell_coo_hybrid.cu   # 代码题E: Hybrid ELL-COO 格式 SpMV（ELL + COO 溢出）
│   └── coo_to_csr.cu            # 代码题F: COO 到 CSR 格式转换（histogram + prefix sum）
└── run_tests.py                  # 一键判题
```

## 使用方法

```bash
cd chapter-14/exercises
python run_tests.py
```
