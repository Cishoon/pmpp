# 第七章：卷积（Convolution）

## 代码

本章实现了四种 2D 卷积内核：

- 朴素卷积
- 使用常量内存的朴素卷积
- 分块卷积（tiled）
- 分块卷积 + L2 缓存利用

运行 Python 版本：

```bash
python code/conv_2d.py
```

运行 C 版本：

```bash
cd code
nvcc run_conv2d.cu conv2d_functions.cu conv2d_kernels.cu -o conv2d_program
./conv2d_program
```

注意：`conv2d_kernels.cuh` 中的 `FILTER_RADIUS` 是硬编码的，C 版本和 Python 版本使用不同的值，运行前请确认。

## 习题

### 习题 1

**计算图 7.3 中 P[0] 的值。**

`x = [0, 0, 8, 2, 5]`，`f = [1, 3, 5, 3, 1]`

逐元素相乘：`[0×1, 0×3, 8×5, 2×3, 5×1] = [0, 0, 40, 6, 5]`

求和：`0 + 0 + 40 + 6 + 5 = **51**`

### 习题 2

**对数组 N = {4,1,3,2,3} 用滤波器 F = {2,1,4} 进行 1D 卷积，结果是什么？**

加入 ghost cell 后：`N' = [0, 4, 1, 3, 2, 3, 0]`

- `P[0]` = `[0,4,1]·[2,1,4]` = `0+4+4` = **8**
- `P[1]` = `[4,1,3]·[2,1,4]` = `8+1+12` = **21**
- `P[2]` = `[1,3,2]·[2,1,4]` = `2+3+8` = **13**
- `P[3]` = `[3,2,3]·[2,1,4]` = `6+2+12` = **20**
- `P[4]` = `[2,3,0]·[2,1,4]` = `4+3+0` = **7**

结果：`P = [8, 21, 13, 20, 7]`

### 习题 3

**以下 1D 卷积滤波器各有什么作用？**

**a. [0 1 0]**：恒等滤波器，输出与输入完全相同。

**b. [0 0 1]**：右移滤波器，将信号整体向右移动一位。

**c. [1 0 0]**：左移滤波器，将信号整体向左移动一位。

**d. [-1/2 0 1/2]**：边缘检测滤波器，相邻值差异大时输出大，相邻值相近时输出接近 0。

**e. [1/3 1/3 1/3]**：平均滤波器，用邻域均值替换当前值，起平滑降噪作用。

### 习题 4

**对大小为 N 的数组用大小为 M 的滤波器进行 1D 卷积：**

设 `r = (M-1)/2`。

**a. ghost cell 总数？**

左右各 `r` 个，共 **`M-1`** 个（即 `2r` 个）。

**b. 将 ghost cell 视为乘以 0 时，乘法次数？**

每个元素做 `M` 次乘法，共 `N` 个元素：**`N×M`** 次。

**c. 不将 ghost cell 视为乘法时，乘法次数？**

从 `N×M` 中减去边界处省略的乘法。左边第 `i` 个元素（`i=0..r-1`）省略 `r-i` 次，右边对称。

省略总数 = `2 × Σ(k=1 to r) k = 2 × r(r+1)/2 = r(r+1)`

结果：**`N×M - r(r+1)`** 次。

### 习题 5

**对 N×N 矩阵用 M×M 滤波器进行 2D 卷积：**

设 `r = (M-1)/2`。

**a. ghost cell 总数？**

四条边各 `N×r` 个，四个角各 `r×r` 个：

**`4r(N + r)`** 个。

**b. 将 ghost cell 视为乘以 0 时，乘法次数？**

**`N² × M²`** 次。

**c. 不将 ghost cell 视为乘法时，乘法次数？**

从 `N²×M²` 中减去边缘和角落省略的乘法：

- 行方向边缘：`2 × N × r(r+1)/2 × 2 = 2N×r(r+1)`（上下各一次）
  实际为 `2N×r(r+1)`
- 角落：`4 × [r(r+1)/2]²`

结果：**`N²×M² - 2N×r(r+1) - 4×[r(r+1)/2]²`** 次。

### 习题 6

**对 N₁×N₂ 矩阵用 M₁×M₂ 滤波器进行 2D 卷积：**

设 `r₁=(M₁-1)/2`，`r₂=(M₂-1)/2`。

**a. ghost cell 总数？**

**`2(N₁×r₂ + N₂×r₁) + 4×r₁×r₂`** 个。

**b. 将 ghost cell 视为乘以 0 时，乘法次数？**

**`N₁×N₂×M₁×M₂`** 次。

**c. 不将 ghost cell 视为乘法时，乘法次数？**

**`N₁×N₂×M₁×M₂ - N₁×r₂(r₂+1) - N₂×r₁(r₁+1) - r₁(r₁+1)/2 × r₂(r₂+1)/2 × 4`** 次。

### 习题 7

**对 N×N 矩阵用 M×M 滤波器进行 2D 分块卷积，输出分块大小为 T×T：**

设 `r = (M-1)/2`，输入分块大小 `IN_TILE = T + 2r`。

**a. 需要多少个线程块？**

**`⌈N/T⌉ × ⌈N/T⌉`** 个。

**b. 每个块需要多少线程？**

**`(T+2r) × (T+2r)`** 个（即 `IN_TILE × IN_TILE`）。

**c. 每个块需要多少共享内存？**

**`(T+2r)² × 4`** 字节。

**d. 使用图 7.15 的内核（L2 缓存版）时：**

- 线程块数量不变：`⌈N/T⌉ × ⌈N/T⌉`
- 每块线程数：**`T × T`**（只处理输出分块）
- 每块共享内存：**`T² × 4`** 字节

### 习题 8

**将图 7.7 的 2D 内核改写为 3D 卷积：**

```cpp
__global__ void convolution_3D_basic_kernel(float *N, float *F, float *P,
    int r, int width, int height, int depth) {
    int outCol   = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow   = blockIdx.y*blockDim.y + threadIdx.y;
    int outDepth = blockIdx.z*blockDim.z + threadIdx.z;

    float Pvalue = 0.0f;
    for (int fDepth = 0; fDepth < 2*r+1; fDepth++) {
        for (int fRow = 0; fRow < 2*r+1; fRow++) {
            for (int fCol = 0; fCol < 2*r+1; fCol++) {
                int inDepth = outDepth - r + fDepth;
                int inRow   = outRow   - r + fRow;
                int inCol   = outCol   - r + fCol;
                if (inRow >= 0 && inRow < height &&
                    inCol >= 0 && inCol < width  &&
                    inDepth >= 0 && inDepth < depth) {
                    Pvalue += F[fDepth*(2*r+1)*(2*r+1) + fRow*(2*r+1) + fCol]
                            * N[inDepth*width*height + inRow*width + inCol];
                }
            }
        }
    }
    P[outDepth*width*height + outRow*width + outCol] = Pvalue;
}
```

### 习题 9

**将图 7.9 的常量内存 2D 内核改写为 3D 卷积：**

与习题 8 相同，但滤波器从常量内存 `F_c` 读取，省略边界检查时的乘法。

### 习题 10

**将图 7.12 的分块 2D 内核改写为 3D 卷积：**

```cpp
#define IN_TILE_DIM 16
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__ void convolution_tiled_3D_const_mem_kernel(float *N, float *P,
                                                      int width, int height, int depth) {
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int dep = blockIdx.z*OUT_TILE_DIM + threadIdx.z - FILTER_RADIUS;

    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    if (dep>=0 && dep<depth && row>=0 && row<height && col>=0 && col<width)
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = N[dep*width*height + row*width + col];
    else
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    int tileDep = threadIdx.z - FILTER_RADIUS;

    if (dep>=0 && dep<depth && row>=0 && row<height && col>=0 && col<width) {
        if (tileCol>=0 && tileCol<OUT_TILE_DIM &&
            tileRow>=0 && tileRow<OUT_TILE_DIM &&
            tileDep>=0 && tileDep<OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fDep = 0; fDep < 2*FILTER_RADIUS+1; fDep++)
                for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++)
                    for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++)
                        Pvalue += F_c[fDep][fRow][fCol]
                                * N_s[tileDep+fDep][tileRow+fRow][tileCol+fCol];
            P[dep*width*height + row*width + col] = Pvalue;
        }
    }
}
```
