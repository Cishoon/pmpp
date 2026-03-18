# 第二十二章 练习

## 结构

```
exercises/
├── kernels/
│   ├── naive_attention.cu              # 代码题A: 朴素 Attention（三步法）
│   ├── flash_attention_1pass.cu        # 代码题B: FlashAttention 单 block（1-pass 在线 softmax）
│   ├── flash_attention_tiled.cu        # 代码题C: FlashAttention 分块版（多 block）
│   └── flash_attention_causal.cu       # 代码题D: FlashAttention 因果掩码版
├── answers.py                           # 习题1-7的计算/简答题
└── run_tests.py                         # 一键判题
```

## 使用方法

```bash
cd chapter-22/exercises
python run_tests.py
```
