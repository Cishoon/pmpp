#include <torch/extension.h>

// 分块并行归并内核（共享内存优化）
//
// 与基础版的区别：
//   利用共享内存减少全局内存上的二分搜索次数
//   只有 block 级别的 co-rank 在全局内存上执行，线程级别的 co-rank 在共享内存上执行
//
// 思路：
//   - 每个 block 先用 co_rank 确定自己负责的 C 子数组范围 [C_curr, C_next)
//   - 对应地确定 A 和 B 的子数组范围
//   - 迭代处理：每次将 TILE_SIZE 个 A 和 B 元素加载到共享内存
//   - 每个线程在共享内存上执行 co_rank，确定自己的局部归并范围
//   - 调用 merge_sequential 完成局部归并，结果写回全局内存
//   - 更新已消耗的 A、B 元素计数，进入下一轮迭代
//
// host 端需要：
//   - 创建输出张量（长度 m+n）
//   - 配置 grid 和 block，分配 2*TILE_SIZE*sizeof(float) 的动态共享内存
//   - 启动内核并返回结果

#define TILE_SIZE 2048
#define BLOCK_SIZE 256
#define cdiv(x, y) (((x) + (y) - 1) / (y))

__device__ void merge_sequential(float* A, int m, float* B, int n, float* C) {
    int i = 0, j = 0, k = 0;
    
    while(i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    while(i < m) C[k++] = A[i++];
    while(j < n) C[k++] = B[j++];
}

__device__ int co_rank(int k, float* A, int m, float* B, int n) {
    int i_low = max(0, k - n);
    int i_high = min(m, k);
    
    while(true) {
        int i = i_low + (i_high - i_low) / 2;
        int j = k - i;
        
        
        if (i > 0 && i - 1 < m && j >= 0 && j < n && A[i - 1] > B[j]) {
            i_high = i - 1;
        } else if (i >= 0 && i < m && j > 0 && j - 1 < n && A[i] < B[j - 1]) {
            i_low = i + 1;
        } else {
            return i;
        }
    }
}

__global__ void MergeTiledKernel(float* A, int m, float* B, int n, float* C) {
    extern __shared__ float shared_memory[];
    float* s_A = shared_memory;
    float* s_B = &shared_memory[TILE_SIZE];
    // tile版本，每个block先把数据读到共享内存里，每个block完成TILE_SIZE个数据的排序
    // 每个block线程数是 blockDim.x，所以每个线程要完成 2 * TILE_SIZE / blockDim.x 个数据的排序
    
    /// 1. 确定block的数据范围
    int C_curr = blockIdx.x * TILE_SIZE;
    int C_next = min((blockIdx.x + 1) * TILE_SIZE, m + n);
    if (C_curr >= m + n) return;
    
    __shared__ int A_curr, A_next, B_curr, B_next;
    if (threadIdx.x == 0) {
        A_curr = co_rank(C_curr, A, m, B, n);
        A_next = co_rank(C_next, A, m, B, n);
        B_curr = C_curr - A_curr;
        B_next = C_next - A_next;
    }
    __syncthreads();
    
    /// 2. 加载数据到共享内存
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int tid = threadIdx.x;
    int num_per_thread = TILE_SIZE / blockDim.x;
    for (int i = tid; i < A_length; i += blockDim.x) {
        s_A[i] = A[A_curr + i];
    }
    for (int i = tid; i < B_length; i += blockDim.x) {
        s_B[i] = B[B_curr + i];
    }
    __syncthreads();
    
    /// 3. 共享内存上局部归并
    // int k_curr = tid * num_per_thread;
    // int k_next = min((tid + 1) * num_per_thread, C_next - C_curr);
    // int i_curr = co_rank(k_curr, s_A, A_next - A_curr, s_B, B_next - B_curr);
    // int i_next = co_rank(k_next, s_A, A_next - A_curr, s_B, B_next - B_curr);
    // int j_curr = k_curr - i_curr;
    // int j_next = k_next - i_next;
    
    // merge_sequential(&s_A[i_curr], i_next - i_curr, 
    //                  &s_B[j_curr], j_next - j_curr, 
    //                  &C[C_curr + k_curr]);
    
    int elements_in_tile = C_next - C_curr;
    int elements_per_thread = cdiv(elements_in_tile, blockDim.x);
    
    // 计算当前线程在这个 Tile 内部的相对偏移量 k
    int k_curr_local = min(tid * elements_per_thread, elements_in_tile);
    int k_next_local = min((tid + 1) * elements_per_thread, elements_in_tile);
    
    // 如果当前线程分不到任务，直接结束
    if (k_curr_local >= k_next_local) return;
    
    // 在共享内存中做极速小二分
    int i_curr_local = co_rank(k_curr_local, s_A, A_length, s_B, B_length);
    int i_next_local = co_rank(k_next_local, s_A, A_length, s_B, B_length);
    int j_curr_local = k_curr_local - i_curr_local;
    int j_next_local = k_next_local - i_next_local;
    
    // 4. 【局部归并】每个线程只合并自己负责的那一小块，并直接写回 Global Memory
    // 💡 注意指针的写法：传入的是局部的起始地址！
    merge_sequential(&s_A[i_curr_local], i_next_local - i_curr_local,
                     &s_B[j_curr_local], j_next_local - j_curr_local,
                     &C[C_curr + k_curr_local]); // 输出坐标 = Block起点 + Thread相对起点
}

// 函数签名: torch::Tensor merge_tiled(torch::Tensor A, torch::Tensor B)
torch::Tensor merge_tiled(torch::Tensor A, torch::Tensor B) {
    int m = A.size(0);
    int n = B.size(0);
    int total = m + n;
    auto C = torch::empty({total}, A.options());

    int threadsPerBlock = BLOCK_SIZE;
    int numBlocks = cdiv(total, threadsPerBlock);
    numBlocks = min(numBlocks, 65535);

    dim3 block(threadsPerBlock);
    dim3 grid(numBlocks);
    int sharedMemBytes = 2 * TILE_SIZE * sizeof(float);

    MergeTiledKernel<<<grid, block, sharedMemBytes>>>(
        A.data_ptr<float>(), m,
        B.data_ptr<float>(), n,
        C.data_ptr<float>());

    return C;
}
