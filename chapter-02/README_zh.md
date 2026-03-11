# 第二章
《大规模并行处理器编程》习题解答

## 代码

第二章实现了一个基础的向量乘法内核，内核代码位于 [vecMul.cu](code/vecMul.cu)。

### C
直接运行 [Makefile](code/Makefile) 即可执行 C 代码：

```bash
cd code
```

```bash
make
```

### Python

配置 Python 环境并安装所需依赖包（`torch` 和 `Ninja`）：

```bash
pip install -r requirements.txt
```

然后运行 Python 脚本：

```
python vecMul.py
```

## 习题

### 习题 1
**问题：** 如果我们希望网格中的每个线程计算向量加法的一个输出元素，那么将线程/块索引映射到数据索引（i）的表达式应该是什么？

**选项：**
- A. `i = threadIdx.x + threadIdx.y;`
- B. `i = blockIdx.x + threadIdx.x;`
- C. `i = blockIdx.x * blockDim.x + threadIdx.x;`
- D. `i = blockIdx.x * threadIdx.x;`

**解答：**

**C**，即 `i = blockIdx.x * blockDim.x + threadIdx.x`。我们需要这三个量：`blockIdx.x` 用于标识块，`blockDim.x` 用于标识每个块的大小。每个块的长度相同（例如 256）。假设我们要对第一个块中的第 128 个元素运行内核，则 i 为 `1 * 256 + 128 = 384`。


### 习题 2
**问题：** 假设我们希望每个线程计算向量加法中相邻的两个元素，那么将线程/块索引映射到某线程所处理的第一个元素的数据索引（i）的表达式应该是什么？

**选项：**
- A. `i = blockIdx.x * blockDim.x + threadIdx.x * 2;`
- B. `i = blockIdx.x * threadIdx.x * 2;`
- C. `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;`
- D. `i = blockIdx.x * blockDim.x * 2 + threadIdx.x;`

**解答：**

**C**。我们希望每个内核处理两个相邻元素，例如 (0, 1)、(1, 2)、...、(1024, 1025)。为此，需要一个每次跳过一个元素的公式。适用的公式为 `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2`：当 `blockIdx=0`、`threadIdx.x=0` 时，第一个元素为 0；`threadIdx.x=1` 时为 2；`blockIdx=4`、`threadIdx.x=0` 时为 1024（假设块大小为 256）。注意当 `threadIdx.x > 128` 时会自动跳到下一个块。


### 习题 3
**问题：** 我们希望每个线程计算向量加法中的两个元素。每个线程块处理 2 * blockDim.x 个连续元素，分为两个区段。每个块中的所有线程先处理第一个区段（每个线程处理一个元素），然后再处理第二个区段（每个线程处理一个元素）。假设变量 i 为某线程处理的第一个元素的索引，将线程/块索引映射到第一个元素数据索引的表达式应该是什么？

**选项：**
- A. `i=blockIdx.x*blockDim.x + threadIdx.x +2;`
- B. `i=blockIdx.x*threadIdx.x*2;`
- C. `i=(blockIdx.x*blockDim.x + threadIdx.x)*2;`
- D. `i=blockIdx.x*blockDim.x*2 + threadIdx.x;`

**解答：**

**D**。我们希望每个块处理 $2 \times BLOCKSIZE$ 个元素。假设块大小为 256，则块 0 处理元素 0 到 511：第一区段处理 0 到 255，第二区段处理 256 到 511。每个线程处理的示例索引对为 $(0, 256)、(1, 257) \dots (255, 511)$。块 1 需要跳过 256 到 511 的部分，因此需要将块大小对应的部分乘以 2，即 `i=blockIdx.x*blockDim.x*2 + threadIdx.x`。


### 习题 4

**问题：** 对于向量加法，假设向量长度为 8000，每个线程计算一个输出元素，线程块大小为 1024 个线程。程序员配置内核调用时使用覆盖所有输出元素所需的最少线程块数。网格中共有多少个线程？

**选项：**

- A. 8000

- B. 8196

- C. 8192

- D. 8200

**解答：**

**C**。每个块有 1024 个线程，处理 8000 个元素需要 8 个线程块，因此共有 `8 * 1024 = 8192` 个线程。


### 习题 5

**问题：** 如果我们想在 CUDA 设备全局内存中分配一个包含 v 个整数元素的数组，`cudaMalloc` 调用的第二个参数应该是什么？

**选项：**

- A. n

- B. v

- C. n * sizeof(int)

- D. v * sizeof(int)

**解答：**

**D**。`cudaMalloc` 函数接受两个参数：指针的地址和以字节为单位的大小。由于我们要分配 `v × int_size` 个字节，答案 D 是正确的。


### 习题 6

**问题：** 如果我们想分配一个包含 n 个浮点元素的数组，并用浮点指针变量 A_d 指向已分配的内存，`cudaMalloc` 调用的第一个参数应该是什么？

**选项：**

- A. n

- B. (void*) A_d

- C. *A_d

- D. (void**) &A_d

**解答：**

**D**。`cudaMalloc` 函数的第一个参数是指向指针的指针。由于 `A_d` 是一个指针，`&A_d` 就是该指针的地址。`cudaMalloc` 的第一个参数类型为 `void**`，因此需要将其强制转换为正确的类型。


### 习题 7

**问题：** 如果我们想将 3000 字节的数据从主机数组 A_h（A_h 指向源数组第 0 个元素）复制到设备数组 A_d（A_d 指向目标数组第 0 个元素），应该使用哪个 CUDA API 调用？

**选项：**

- A. cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);

- B. cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceToHost);

- C. cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);

- D. cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice);

**解答：**

**C**。`A_d` 是目标，`A_h` 是源，大小为 3000，传输方向是从主机到设备，因此答案为 C。


### 习题 8

**问题：** 如何声明一个变量 err，使其能够正确接收 CUDA API 调用的返回值？

**选项：**

- A. int err;

- B. cudaError err;

- C. cudaError_t err;

- D. cudaSuccess_t err;

**解答：**

**C**。CUDA 错误的正确类型是 `cudaError_t`，因此答案为 C。


### 习题 9

考虑以下 CUDA 内核及其对应的调用它的主机函数：

```c
01 __global__ void foo_kernel(float* a, float* b, unsigned int N) {
02     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
03     
04     if (i < N) {
05         b[i] = 2.7f * a[i] - 4.3f;
06     }
07 }
08 
09 void foo(float* a_d, float* b_d) {
10     unsigned int N = 200000;
11     foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d, N);
12 }
```

**a. 每个块有多少个线程？**
由内核启动参数（`<<< >>>` 中的内容）的第二个参数可知，为 `128`。

**b. 网格中共有多少个线程？**
线程总数等于 `块数 * 块大小`。块数由内核启动参数的第一个参数给出，即 `(N + 128 - 1) / 128` -> `(200000 + 128 - 1) // 128 = 1563`。因此线程总数为 `1563 * 128 = 200064`。

**c. 网格中共有多少个块？**
如上所述，为 `1563`。

**d. 执行第 02 行代码的线程有多少个？**
如上所述，为 `1563 * 128 = 200064`。

**e. 执行第 04 行代码的线程有多少个？**
这里有一个 if 语句，将执行限制在 `cudaMalloc` 分配的内存范围内，即 `N=200000`，因此最后 64 个线程不会被使用，执行第 04 行的线程数为 200000。

### 习题 10：
**问题：** 一位新来的暑期实习生对 CUDA 感到很沮丧，他抱怨 CUDA 非常繁琐——他不得不将许多计划在主机和设备上都执行的函数声明两次，一次作为主机函数，一次作为设备函数。你会如何回应？

**解答：** 可以直接使用同时带有 `__host__` 和 `__device__` 函数类型限定符的 CUDA 函数。这样 CUDA 编译器会同时编译该函数的主机版本和设备版本，无需重复代码。
