# 第三章

在第三章中，我们学习了多维数据网格，并编写了第一批较为复杂的内核。

## 代码

为了简洁起见，我们提供的大部分代码都带有 Python 接口。你只需运行 Python 脚本，底层会自动调用 CUDA 内核。我们重新实现了本章中的内核，并为习题一和习题二实现了对应的内核。

我们实现了：

- 矩阵乘法，内核分别在列级别和行级别上操作。
- 矩阵-向量乘法内核。
- 矩阵乘法内核。
- RGB 转灰度图内核。
- 高斯模糊内核。

对于高斯模糊，我们提供了一个小型 Gradio 应用，方便你可视化内核的效果。运行方式：

```bash
python gaussian_blur/gradio_visualization.py
```

<img src="gradio.png" alt="Gradio 界面" width="1000"/>

## 习题

### 习题 1
本章中我们实现了一个矩阵乘法内核，每个线程生成输出矩阵的一个元素。在本题中，你将实现不同的矩阵乘法内核并进行比较。

完整解答见 [exercise_1](code/exercise_1)

**a.** 编写一个内核，使每个线程生成输出矩阵的一整行。填写该设计的执行配置参数。
```cu
1.  __global__
2.  void matrixMulRowKernel(float* M, float* N, float* P, int size){
3.      int row = blockIdx.x * blockDim.x + threadIdx.x;
4.      if (row < size){
5.          //对该行的每个元素执行：
6.          for (int col=0; col<size; ++col){
7.              float sum = 0;
8.              for (int j=0; j<size; ++j){
9.                  sum += M[row * size + j] * N[j * size + col];
10.             }
11.             P[row * size + col] = sum;
12.         }
13.     }
14. }
```


**b.** 编写一个内核，使每个线程生成输出矩阵的一整列。填写该设计的执行配置参数。
```cu
1. __global__
2. void matrixMulColKernel(float* M, float* N, float* P, int size){
3.     int col = blockIdx.x * blockDim.x + threadIdx.x;
4.     if (col < size){
5.         // 对该列的每个元素执行：
6.         for (int row = 0; row < size; ++row){
7.             float sum = 0;
8.             for (int j = 0; j < size; ++j){
9.                 sum += M[row * size + j] * N[j * size + col];
10.            }
11.            P[row * size + col] = sum;
12.        }
13.    }
14.}
```


**c.** 分析这两种内核设计的优缺点。

两种设计的工作方式类似，都相当低效，对多核的利用率较差。如果我们为非方阵设计内核，当列数远大于行数时，按行处理的方式会更低效（循环次数更多），反之按列处理的方式则更低效。


### 习题 2
矩阵-向量乘法接受一个输入矩阵 B 和一个向量 C，生成一个输出向量 A。输出向量 A 的每个元素是输入矩阵 B 的一行与 C 的点积，即 `A[i] = sum_over_j(B[i][j] * C[j])`。为简单起见，我们只处理元素为单精度浮点数的方阵。编写一个矩阵-向量乘法内核及其主机端桩函数，该函数接受四个参数：输出矩阵指针、输入矩阵指针、输入向量指针和每个维度的元素数量。使用一个线程计算一个输出向量元素。

完整解答见 [exercise_2](code/exercise_2)

```cu
1  __global__
2  void matrixVecMulKernel(float* B, float* c, float* result, int vector_size, int matrix_rows){
3      int i = blockIdx.x * blockDim.x + threadIdx.x;
4      if (i < matrix_rows){
5          float sum = 0;
6          for (int j=0; j < vector_size; ++j){
7              sum += B[i * vector_size + j] * c[j];
8          }
9          result[i] = sum;
10     }
11 }
```



### 习题 3
考虑以下 CUDA 内核及其对应的调用它的主机函数：

```cu
01 __global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N) {
02     unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
03     unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
04     if (row < M && col < N) {
05         b[row*N + col] = a[row*N + col]/2.1f + 4.8f;
06     }
07 }
08 void foo(float* a_d, float* b_d) {
09     unsigned int M = 150;
10     unsigned int N = 300;
11     dim3 bd(16, 32);
12     dim3 gd((N - 1) / 16 + 1, ((M - 1) / 32 + 1));
13     foo_kernel <<< gd, bd >>> (a_d, b_d, M, N);
14 }
```

**a. 每个块有多少个线程？**

线程块中的线程数可以从变量 `bd`（blockDim）推断，为 `16 x 32 = 512` 个线程。


**b. 网格中共有多少个线程？**

网格中的线程总数等于 `网格中的块数 × 每块的线程数`，即 `512 x 95 = 48,640`（参见 **3a** 和 **3c**）。

**c. 网格中共有多少个块？**

网格中的块数可以从变量 `gd`（gridDim）推断。计算公式为 `((N - 1) / 16 + 1, ((M - 1) / 32 + 1))`，其中 `M=150`，`N=300`。因此 `((300-1)/16 + 1, (150-1)/32 + 1)` → `(299/16 + 1, 149/32 + 1)` → `(18 + 1, 4 + 1)` → `(19, 5)` → `19 x 5 = 95` 个块。

**d. 执行第 05 行代码的线程有多少个？**

要回答这个问题，我们需要知道：`M`、最大可能的 `row`、`N` 和最大可能的 `col`。

- `M` 为 150
- 最大 `row` 为 `blockIdx.y * blockDim.y + threadIdx.y`。最大 `blockIdx.y` 为 4（0-4，参见 **3c**），`blockDim.y` 为 32，因此最大 `threadIdx.y` 为 31。所以 `4 x 32 + 31 = 159`
- `N` 为 300
- 最大 `col` 为 `blockIdx.x * blockDim.x + threadIdx.x`。最大 `blockIdx.x` 为 18（0-18，参见 **3c**），`blockDim.x` 为 16，因此最大 `threadIdx.x` 为 15。所以 `18 x 16 + 15 = 303`

因此执行的线程总数为 `min(150, 159) x min(300, 303) = 150 x 300 = 45,000`——略少于线程总数（`48,640`）。

### 习题 4
考虑一个宽度为 400、高度为 500 的二维矩阵。该矩阵以一维数组存储。指定第 20 行第 10 列的矩阵元素的数组索引：
- **a.** 如果矩阵按行优先顺序存储。

在行优先顺序中，数组线性化公式为 `row x width + col`，因此索引为 `20 x 400 + 10 = 8,010`。

- **b.** 如果矩阵按列优先顺序存储。

在列优先顺序中，数组线性化公式为 `col x height + row`，因此索引为 `10 x 500 + 20 = 5,020`。

### 习题 5
考虑一个宽度为 400、高度为 500、深度为 300 的三维张量。该张量以行优先顺序存储为一维数组。指定 x = 10、y = 20、z = 5 处张量元素的数组索引。

三维张量中元素的线性化索引计算公式为 `plane x width x height + row x width + col`，因此索引为 `5 x 400 x 500 + 20 x 400 + 10 = 1,000,000 + 8,000 + 10 = 1,008,010`。
