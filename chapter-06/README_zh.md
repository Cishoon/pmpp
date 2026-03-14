# 第六章：线程粗化（Thread Coarsening）

## 代码

本章在分块矩阵乘法的基础上引入了**线程粗化**优化。运行实验：

```bash
cd code
nvcc -o thread_coarsening thread_coaersing_matmul.cu
./thread_coarsening
```

核心思路：让每个线程负责计算 P 中多个元素（由 `COARSE_FACTOR` 控制），从而减少冗余的全局内存加载，提升计算密度。

## 习题

### 习题 1

**编写对应图 6.4 所示设计的矩阵乘法内核函数。**

这道题要求实现**列主序访问 N（Corner Turning）**的分块矩阵乘法。

问题背景：标准分块乘法中，加载 N 的分块时，同一 warp 内的线程访问的是 N 的同一列的不同行（`N[(ph*TILE+ty)*o + col]`），这是 coalesced 的。但如果 N 以列主序存储，则需要用 corner turning 技巧：先 coalesced 地加载 N 的转置数据到共享内存，再从共享内存中以正确顺序读取。

```cpp
__global__ void TiledMatrixMulKernelColMajorOrder(float* M, float* N, float* P,
                                                   int m, int n, int o) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float PValue = 0;
    for (int ph = 0; ph < (n + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        if (row < m && (ph * TILE_WIDTH + threadIdx.x) < n)
            Mds[threadIdx.y][threadIdx.x] = M[row * n + ph * TILE_WIDTH + threadIdx.x];
        else
            Mds[threadIdx.y][threadIdx.x] = 0.0f;

        // N 以列主序存储（即已转置），coalesced 地按列读取
        if ((ph * TILE_WIDTH + threadIdx.y) < n && col < o)
            Nds[threadIdx.y][threadIdx.x] = N[col * n + (ph * TILE_WIDTH + threadIdx.y)];
        else
            Nds[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++)
            PValue += Mds[threadIdx.y][k] * Nds[k][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < o)
        P[row * o + col] = PValue;
}
```

完整代码见 [excercise1.cu](code/excercise1.cu)。

### 习题 2

**对于分块矩阵乘法，BLOCK_SIZE 取哪些值时内核能完全避免非合并的全局内存访问？（只考虑方形块）**

Coalesced 访问要求同一 warp 内的线程访问连续的内存地址。Warp 由 32 个相邻线程组成。如果 `BLOCK_SIZE < 32`，一个 warp 会跨越多行，访问多行的数据，导致非合并访问。

因此 `BLOCK_SIZE` 必须是 **32 的整数倍**（32、64 等）才能完全避免非合并访问。实际中受共享内存限制，通常不超过 64。

### 习题 3

**对以下 CUDA 内核的每个内存访问，判断是 coalesced、uncoalesced 还是不适用：**

```cpp
__global__ void foo_kernel(float* a, float* b, float* c, float* d, float* e) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float a_s[256];
    __shared__ float bc_s[4*256];
    a_s[threadIdx.x] = a[i];                                    // 05
    for(unsigned int j = 0; j < 4; ++j) {
        bc_s[j*256 + threadIdx.x] = b[j*blockDim.x*gridDim.x + i] + c[i*4 + j];  // 07
    }
    __syncthreads();
    d[i + 8] = a_s[threadIdx.x];                               // 10
    e[i*8] = bc_s[threadIdx.x*4];                              // 11
}
```

- **a. 第 05 行访问数组 a**：`a[blockIdx.x*blockDim.x + threadIdx.x]`，相邻线程访问相邻地址 → **Coalesced**

- **b. 第 05 行访问数组 a_s**：共享内存，不适用 coalescing → **不适用**

- **c. 第 07 行访问数组 b**：`b[j*blockDim.x*gridDim.x + i]`，相邻线程的 `i` 连续 → **Coalesced**

- **d. 第 07 行访问数组 c**：`c[i*4 + j]`，相邻线程的 `i` 差 1，但乘以 4，地址间隔 4 个元素 → **Uncoalesced**

- **e. 第 07 行访问数组 bc_s**：共享内存 → **不适用**

- **f. 第 10 行访问数组 a_s**：共享内存 → **不适用**

- **g. 第 10 行访问数组 d**：`d[i + 8]`，相邻线程访问相邻地址（偏移 8 不影响 coalescing） → **Coalesced**

- **h. 第 11 行访问数组 bc_s**：共享内存 → **不适用**

- **i. 第 11 行访问数组 e**：`e[i*8]`，相邻线程地址间隔 8 个元素 → **Uncoalesced**

### 习题 4

**以下各矩阵乘法内核的浮点运算与全局内存访问比（OP/B）是多少？**

假设 M 为 `(m, n)`，N 为 `(n, o)`，使用 float32（4 字节）。

**a. 第三章的朴素内核（无优化）**

每个输出元素需要：
- 从 M 加载 `n` 个元素，从 N 加载 `n` 个元素 → `2n` 次加载，`2n × 4 = 8n` 字节
- `n` 次乘法 + `n` 次加法 → `2n` 次浮点运算

比值：`2n / (8n) = **0.25 OP/B**`

**b. 第五章的分块内核（32×32 分块）**

每个线程只需从全局内存加载 `n/32` 次（其余由同块线程共享）：
- M 和 N 各 `n/32` 次加载 → `2n/32 × 4 = n/4` 字节
- 浮点运算仍为 `2n` 次

比值：`2n / (n/4) = **8 OP/B**`（比朴素版提升 32 倍）

**c. 本章的线程粗化内核（32×32 分块 + 粗化因子 4）**

线程粗化后，M 的分块被复用 4 次，全局内存加载进一步减少：
- M：`n/32/4 = n/128` 次加载
- N：`n/32` 次加载（不变）
- 总字节：`(n/128 + n/32) × 4 = (5n/128) × 4 = 5n/32` 字节
- 浮点运算：`2n` 次

比值：`2n / (5n/32) = **12.8 OP/B**`
