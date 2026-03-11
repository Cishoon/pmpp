# 第十三章 排序

## 代码

我们实现了第十三章中提到的所有内核，具体包括：

- 朴素并行基数排序（三个独立内核）
- 内存合并写入的基数排序
- 内存合并写入 + 多位基数的基数排序
- 内存合并写入 + 多位基数 + 线程粗化的基数排序
- 并行归并排序

我们将所有这些实现与 CPU 上执行的 `quicksort` 进行比较，并报告加速比。虽然这种比较并不完全公平，但我们希望让读者感受到"GPU 排序相比 CPU 实现有多快"。

运行基准测试：

```bash
make
```

对于 `N = 10,000,000`，你应该看到类似以下的结果：
```
=== Performance Summary ===
+---------------------------------------------------------------------+-------------+----------+
|                            Sort Method                              | Time (ms)   | Speedup  |
+---------------------------------------------------------------------+-------------+----------+
| Naive Parallel Radix Sort                                           |      14.099 |  149.70x |
| Memory-coalesced GPU sort                                           |      18.055 |  116.90x |
| Memory-coalesced GPU sort (multiradix)                              |      20.671 |  102.11x |
| Memory-coalesced GPU sort (multiradix and thread coarsening)        |      39.275 |   53.74x |
| GPU merge sort                                                      |     135.828 |   15.54x |
+---------------------------------------------------------------------+-------------+----------+
```

这里可能还有一些进一步的优化空间来榨取最后一点性能，但我们将此视为一种练习，不再深入探讨。

我们还实现了单内核版本的朴素并行基数排序。这里的主要挑战是整个网格（grid）级别的同步。困难之处在于第一个 block 需要等待最后一个 block，因为我们需要整个输入中零的总数。不幸的是，当元素数量超过大约 100K 时，会发生死锁。因此我们将其排除在基准测试之外。

如果你想简单地实验不同的排序实现，推荐使用 [./code/sort.cu](sort.cu)，在那里你可以轻松尝试新的并行排序实现。它会与标准库的 `quicksort` 进行比较。你也可以在这里尝试上面提到的单内核版本的并行基数排序。

```bash
nvcc sort.cu -o sort
```

## 习题

### 习题 1
**扩展图 13.4 中的内核，使用共享内存（shared memory）来改善内存合并（memory coalescing）。**

该内核相当复杂，建议直接查看实现代码：[gpu_radix_sort.cu](https://github.com/tugot17/pmpp/blob/24a11162953b88652b41011886878c6ebde4c3e2/chapter-13/code/gpu_radix_sort.cu#L287)

核心思路：
- 每个 block 加载自己负责的输入段，在共享内存中计算每个元素的 bit 值
- 使用 Brent-Kung 算法在共享内存中执行 exclusive scan（排他前缀和）
- 将每个 block 的局部扫描结果保存到全局内存 `d_localScan`
- 同时保存每个 block 中 1 的总数到 `d_blockOneCount`
- 在 host 端计算 block 间的全局偏移量（零的偏移和一的偏移）
- scatter 内核根据 bit 值和全局偏移量将元素写入正确位置，实现合并写入

### 习题 2
**扩展图 13.4 中的内核，使其支持多位基数（multibit radix）。**

实现代码：[gpu_radix_sort.cu](https://github.com/tugot17/pmpp/blob/24a11162953b88652b41011886878c6ebde4c3e2/chapter-13/code/gpu_radix_sort.cu#L444)

核心思路：
- 不再每次只看 1 个 bit，而是每次看 r 个 bit，产生 2^r 个桶（bucket）
- 每个 block 在共享内存中构建局部直方图（histogram），统计各桶的元素数量
- 每个线程计算自己在同一桶中的局部偏移（local offset）
- 在 host 端对所有 block 的直方图做前缀和，得到每个 block 每个桶的全局起始位置
- scatter 内核根据全局偏移 + 局部偏移将元素写入目标位置

### 习题 3
**扩展图 13.4 中的内核，应用线程粗化（thread coarsening）来改善内存合并。**

实现代码：[gpu_radix_sort.cu](https://github.com/tugot17/pmpp/blob/24a11162953b88652b41011886878c6ebde4c3e2/chapter-13/code/gpu_radix_sort.cu#L646)

核心思路：
- 在多位基数排序的基础上，每个线程处理 `COARSE_FACTOR` 个元素而不是 1 个
- 每个 block 处理 `BLOCK_SIZE * COARSE_FACTOR` 个元素
- 共享内存需要存储 `numBuckets` 个直方图条目 + `BLOCK_SIZE * COARSE_FACTOR` 个 digit 值
- 每个线程循环处理自己负责的多个元素，计算 digit、更新直方图、计算局部偏移
- scatter 内核同样每个线程处理 `COARSE_FACTOR` 个元素

### 习题 4
**使用第十二章的并行归并实现来实现并行归并排序。**

实现代码：[gpu_merge_sort.cu](https://github.com/tugot17/pmpp/blob/24a11162953b88652b41011886878c6ebde4c3e2/chapter-13/code/gpu_merge_sort.cu#L116)

核心思路：
- 自底向上的归并排序：初始时每个元素是一个长度为 1 的已排序子数组
- 每一轮（pass）将相邻的两个已排序子数组归并为一个更大的已排序子数组
- 每个 block 负责一对子数组的归并
- 使用 co_rank 函数确定每个线程在两个子数组中的分割点
- 每个线程调用 merge_sequential 完成局部归并
- 宽度（width）每轮翻倍，直到整个数组有序
