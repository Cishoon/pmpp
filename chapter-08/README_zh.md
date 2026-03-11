# 第八章：模板计算（Stencil）

## 代码

本章实现了以下所有模板计算内核：
- 顺序模板计算
- 基本并行模板内核
- 使用共享内存的模板内核
- 使用线程粗化的模板内核
- 使用寄存器分块的模板内核

所有内核及其主机函数代码均在 [stencil.cu](./code/stencil.cu) 中。

此外，我们还提供了所有内核的性能基准测试。运行方式：

```bash
cd code
make
```

然后运行：

```bash
./stencil_benchmark
```

输出示例：

```logs
...
================================================================================
Benchmarking 3D Stencil Operations - Grid Size: 128x128x128
================================================================================
Configuration:
Grid size: 128x128x128
Total elements: 2097152
Memory per array: 8.00 MB
OUT_TILE_DIM: 8, IN_TILE_DIM: 8


Results:
Implementation           | Time (ms) | Speedup vs Sequential | Speedup vs Basic
-------------------------|-----------|----------------------|------------------
Sequential              |    4.634  |                1.00x |            0.76x
Parallel Basic          |    3.535  |                1.31x |            1.00x
Shared Memory           |    3.521  |                1.32x |            1.00x
Thread Coarsening       |    3.512  |                1.32x |            1.01x
Register Tiling         |    3.519  |                1.32x |            1.00x

Correctness Verification:
Parallel Basic vs Sequential: ✓ PASS
Shared Memory vs Sequential: ✓ PASS
Thread Coarsening vs Sequential: ✓ PASS
Register Tiling vs Sequential: ✓ PASS

Overall correctness: ✓ All implementations correct
```

### 热传导模拟

我们还探索了模板计算在实际问题中的应用，实现了一个 CUDA 加速的热扩散模拟。该模拟使用 `stencil_3d_parallel_register_tiling` 内核计算热量随时间的变化。

运行前请先确保 CUDA 代码已编译：

```bash
cd code
make
```

然后运行 [heat_simulation.py](./code/heat_simulation.py)：

```bash
python heat_simulation.py
```

*注意：由于 GIF 生成较慢，可能需要几分钟。*

结果应类似于：

![热方程模拟](./code/heat_equation_3d.gif)


## 习题

### 习题 1

**考虑一个大小为 `120 × 120 × 120`（含边界单元）的 3D 模板计算。**

**a. 每次模板扫描计算的输出网格点数是多少？**

`118 × 118 × 118 = 1,643,032` 个点。

**b. 对于图 8.6 的基本内核，假设块大小为 `8 × 8 × 8`，需要多少线程块？**

```cpp
__global__ void stencil_kernel(float* in, float* out, unsigned int N,
                              int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    unsigned int i = blockIdx.z*blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i*N*N + j*N + k] = c0*in[i*N*N + j*N + k]
                             + c1*in[i*N*N + j*N + (k - 1)]
                             + c2*in[i*N*N + j*N + (k + 1)]
                             + c3*in[i*N*N + (j - 1)*N + k]
                             + c4*in[i*N*N + (j + 1)*N + k]
                             + c5*in[(i - 1)*N*N + j*N + k]
                             + c6*in[(i + 1)*N*N + j*N + k];
    }
}

...
dim3 dimBlock(OUT_TILE_DIM, OUT_TILE_DIM, OUT_TILE_DIM);
dim3 dimGrid(cdiv(N, dimBlock.x), cdiv(N, dimBlock.y), cdiv(N, dimBlock.z));
stencil_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);
```

图 8.6 的内核为输入网格中的每个点启动线程。共有 `120 × 120 × 120` 个点，块大小为 `8 × 8 × 8`，因此需要 `120/8 × 120/8 × 120/8 = 15 × 15 × 15 = 3375` 个块。

**c. 对于图 8.8 的共享内存分块内核，假设块大小为 `8 × 8 × 8`，需要多少线程块？**

```cpp
__global__ void stencil_kernel_shared_memory(float* in, float* out, unsigned int N,
                                           int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    int i = blockIdx.z*OUT_TILE_DIM+ threadIdx.z - 1;
    int j = blockIdx.y*OUT_TILE_DIM+ threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM+ threadIdx.x - 1;
    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if(i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N + j*N + k];
    }
    __syncthreads();
    if(i >= 1 && i < N-1 && j >= 1 && j < N-1 && k >= 1 && k < N-1) {
        if(threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM-1 && threadIdx.y >= 1
           && threadIdx.y<IN_TILE_DIM-1 && threadIdx.x>=1 && threadIdx.x<IN_TILE_DIM-1) {
            out[i*N*N + j*N + k] = c0*in_s[threadIdx.z][threadIdx.y][threadIdx.x]
                                 + c1*in_s[threadIdx.z][threadIdx.y][threadIdx.x-1]
                                 + c2*in_s[threadIdx.z][threadIdx.y][threadIdx.x+1]
                                 + c3*in_s[threadIdx.z][threadIdx.y-1][threadIdx.x]
                                 + c4*in_s[threadIdx.z][threadIdx.y+1][threadIdx.x]
                                 + c5*in_s[threadIdx.z-1][threadIdx.y][threadIdx.x]
                                 + c6*in_s[threadIdx.z+1][threadIdx.y][threadIdx.x];
        }
    }
}
...
dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
dim3 dimGrid(cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM));
stencil_kernel_shared_memory<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);
```

块大小为 `8 × 8 × 8`，但该内核的块大小为 `IN_TILE_DIM`。`OUT_TILE_DIM = IN_TILE_DIM - 2 = 8 - 2 = 6`。

每个轴启动 `cdiv(120, 6) = 20` 个块，共 `20 × 20 × 20 = 8,000` 个块。

**d. 对于图 8.10 的线程粗化内核，假设块大小为 `32 × 32`，需要多少线程块？**

```cpp
__global__ void stencil_kernel_thread_coarsening(float* in, float* out, unsigned int N,
                                                int c0, int c1, int c2, int c3, int c4, int c5, int c6) {
    int iStart = blockIdx.z*OUT_TILE_DIM;
    int j = blockIdx.y*OUT_TILE_DIM+ threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM+ threadIdx.x - 1;
    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];
    ...
}
...
dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, 1);
dim3 dimGrid(cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM));
stencil_kernel_thread_coarsening<<<dimGrid, dimBlock>>>(d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6);
```

块大小为 `32 × 32`，即 `IN_TILE_DIM = 32`，`OUT_TILE_DIM = IN_TILE_DIM - 2 = 30`。

每个方向启动 `cdiv(120, 30) = 4` 个块，共 `4 × 4 × 4 = 64` 个块。

### 习题 2

**考虑一个使用共享内存分块和线程粗化的七点（3D）模板计算实现。该实现类似于图 8.10 和 8.12，但分块不是完美立方体。线程块大小为 `32 × 32`，粗化因子为 16（即每个线程块处理 z 方向上 16 个连续输出平面）。**

作为起点，该内核如下：

```cpp
#define OUT_TILE_DIM 30
#define Z_COARSENING 16
#define IN_TILE_DIM (OUT_TILE_DIM + 2)  // 32 (30 + 1 左 halo + 1 右 halo)

__global__ void stencil_7point_coarsened(
    float* in, float* out, int N,
    float c0, float c1, float c2, float c3, float c4, float c5, float c6
) {
    int z_start = blockIdx.z * Z_COARSENING;
    int global_y = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int global_x = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float s_prev[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float s_curr[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float s_next[IN_TILE_DIM][IN_TILE_DIM];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    s_prev[ty][tx] = 0.0f;
    s_curr[ty][tx] = 0.0f;
    s_next[ty][tx] = 0.0f;

    if (z_start - 1 >= 0 && z_start - 1 < N &&
        global_y >= 0 && global_y < N &&
        global_x >= 0 && global_x < N) {
        s_prev[ty][tx] = in[(z_start - 1) * N * N + global_y * N + global_x];
    }

    if (z_start >= 0 && z_start < N &&
        global_y >= 0 && global_y < N &&
        global_x >= 0 && global_x < N) {
        s_curr[ty][tx] = in[z_start * N * N + global_y * N + global_x];
    }

    for (int z_offset = 0; z_offset < Z_COARSENING; z_offset++) {
        int global_z = z_start + z_offset;

        s_next[ty][tx] = 0.0f;
        if (global_z + 1 >= 0 && global_z + 1 < N &&
            global_y >= 0 && global_y < N &&
            global_x >= 0 && global_x < N) {
            s_next[ty][tx] = in[(global_z + 1) * N * N + global_y * N + global_x];
        }

        __syncthreads();

        if (global_z >= 1 && global_z < N - 1 &&
            global_y >= 1 && global_y < N - 1 &&
            global_x >= 1 && global_x < N - 1 &&
            ty >= 1 && ty < IN_TILE_DIM - 1 &&
            tx >= 1 && tx < IN_TILE_DIM - 1) {

            float result = c0 * s_curr[ty][tx] +
                          c1 * s_curr[ty][tx - 1] +
                          c2 * s_curr[ty][tx + 1] +
                          c3 * s_curr[ty - 1][tx] +
                          c4 * s_curr[ty + 1][tx] +
                          c5 * s_prev[ty][tx] +
                          c6 * s_next[ty][tx];

            out[global_z * N * N + global_y * N + global_x] = result;
        }
        __syncthreads();

        s_prev[ty][tx] = s_curr[ty][tx];
        s_curr[ty][tx] = s_next[ty][tx];
    }
}
...
dim3 block_size(IN_TILE_DIM, IN_TILE_DIM, 1);
dim3 grid_size(cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM), cdiv(N, OUT_TILE_DIM));

stencil_7point_coarsened<<<grid_size, block_size>>>(
    d_in, d_out, N, c0, c1, c2, c3, c4, c5, c6
);
```

**a. 线程块在其生命周期内加载的输入分块大小（元素数）是多少？**

每次加载的分块大小为 `IN_TILE_DIM × IN_TILE_DIM = 32 × 32 = 1024` 个元素。每个线程块处理 16 个连续输出平面，加上前后各一个 halo 平面，共加载 `1024 × 18 = 18432` 个元素。

**b. 线程块在其生命周期内处理的输出分块大小（元素数）是多少？**

每个输出平面大小为 `OUT_TILE_DIM × OUT_TILE_DIM = 30 × 30 = 900` 个元素。处理 16 个平面，共 `900 × 16 = 14400` 个元素。

**c. 内核的浮点运算与全局内存访问比（OP/B）是多少？**

加载 `18432` 个元素，每个 4 字节，共 `18432 × 4 = 73728` 字节。

每个输出元素执行 7 次乘法和 6 次加法，共 13 次浮点运算。`14400` 个输出元素共 `14400 × 13 = 187200` 次浮点运算。

`(14400 × 13) / (18432 × 4) = 187200 / 73728 = 2.54` OP/B。

*注意：这里只计算了读取，未包含写入。*

**d. 不使用寄存器分块（如图 8.10）时，每个线程块需要多少共享内存（字节）？**

需要存储三个连续平面，每个平面 `IN_TILE_DIM × IN_TILE_DIM` 个 4 字节浮点数：`3 × 32 × 32 × 4 = 12288` 字节。

**e. 使用寄存器分块（如图 8.12）时，每个线程块需要多少共享内存（字节）？**

只需存储一个平面：`32 × 32 × 4 = 4096` 字节。代价是寄存器需求增加。
