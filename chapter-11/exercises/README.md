# 第十一章 练习

## 结构

```
exercises/
├── kernels/
│   ├── kogge_stone_scan.cu              # 代码题A: Kogge-Stone 基础前缀扫描
│   ├── kogge_stone_double_buffer_scan.cu # 代码题B: Kogge-Stone 双缓冲前缀扫描
│   ├── brent_kung_scan.cu               # 代码题C: Brent-Kung 前缀扫描
│   ├── three_phase_scan.cu              # 代码题D: 三阶段前缀扫描（线程粗化）
│   └── hierarchical_scan.cu             # 代码题E: 分层前缀扫描（任意长度）
└── run_tests.py                          # 一键判题
```

## 使用方法

```bash
cd chapter-11/exercises
python run_tests.py
```
