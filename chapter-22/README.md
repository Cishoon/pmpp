# Chapter 22: FlashAttention

## 背景

标准的 Self-Attention 计算公式为：

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

其中 Q, K, V 的形状为 `(N, d)`，N 是序列长度，d 是 head 维度。

朴素实现需要 **O(N²)** 的 HBM 读写量——先把完整的 `N×N` 注意力矩阵 `S = Q @ K^T` 写入 HBM，再读回来做 softmax，再写出 `P`，最后读 `P` 和 `V` 做矩阵乘。当 N 很大时，HBM 带宽成为瓶颈。

FlashAttention 的核心思想是 **分块（tiling）+ 在线 softmax（online softmax）**，将 Q、K、V 按块加载到 SRAM（共享内存），在片上完成所有计算，避免将 `N×N` 矩阵写回 HBM，从而将 HBM 访问量降至 **O(N²d / M)**（M 为 SRAM 大小）。

## 关键概念

### 1. 在线 Softmax（Online Softmax）

标准 softmax 需要两遍扫描：第一遍求 max，第二遍求 exp 和 sum。FlashAttention 使用在线算法，在分块处理时动态维护：

- `m`：当前已见元素的最大值
- `l`：当前已见元素的 exp 之和（以 m 为基准）

当处理新的一块 `S_j` 时，更新规则为：

```
m_new = max(m_old, max(S_j))
l_new = l_old * exp(m_old - m_new) + sum(exp(S_j - m_new))
O_new = O_old * (l_old * exp(m_old - m_new) / l_new) + exp(S_j - m_new) / l_new @ V_j
```

### 2. 分块策略

将 Q 按行分成大小为 `B_r` 的块，K 和 V 按行分成大小为 `B_c` 的块：

- 外层循环：遍历 Q 的每个块 `Q_i`（每个块由一个 thread block 处理）
- 内层循环：遍历 K、V 的每个块 `(K_j, V_j)`
- 每次内层迭代在 SRAM 中计算 `S_ij = Q_i @ K_j^T`，更新在线 softmax 统计量，累加 `O_i`

### 3. 内存层次

```
HBM (全局内存)          SRAM (共享内存)           寄存器
┌──────────────┐       ┌──────────────┐       ┌──────────┐
│ Q  (N × d)   │──────>│ Q_i (B_r × d)│──────>│ 累加器 O │
│ K  (N × d)   │──────>│ K_j (B_c × d)│       │ m, l     │
│ V  (N × d)   │──────>│ V_j (B_c × d)│       └──────────┘
│ O  (N × d)   │<──────│ S_ij(B_r×B_c)│
└──────────────┘       └──────────────┘
```

## 代码

本章提供了从朴素到优化的多个 FlashAttention 实现：

### 朴素 Attention（基线）

```bash
cd code
nvcc -O2 naive_attention.cu -o naive_attention && ./naive_attention
```

标准的三步实现：计算 S、softmax、乘 V。需要 O(N²) 额外内存。

### FlashAttention v1（基础分块版）

```bash
nvcc -O2 flash_attention_v1.cu -o flash_attention_v1 && ./flash_attention_v1
```

实现分块 + 在线 softmax，单 block 处理一行 Q。

### FlashAttention v2（优化版）

```bash
nvcc -O2 flash_attention_v2.cu -o flash_attention_v2 && ./flash_attention_v2
```

交换内外循环顺序（外层遍历 K/V 块），减少对 O 的 HBM 写回次数，并利用 warp 级原语加速归约。

## 练习

练习位于 [exercises/](exercises/) 目录，包含简答题和代码题。

```bash
cd exercises
python run_tests.py
```

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Dao, 2023)](https://arxiv.org/abs/2307.08691)
- [Online normalizer calculation for softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867)
