#include <torch/extension.h>

// 基础并行归并内核
//
// 输入 A: 已排序的 1D 浮点张量（长度 m）
// 输入 B: 已排序的 1D 浮点张量（长度 n）
// 输出: 长度为 m+n 的已排序张量
//
// 思路：
//   - 实现 co_rank 函数：给定输出位置 k，通过二分搜索确定从 A 取 i 个元素、从 B 取 j=k-i 个元素
//   - 每个线程计算自己负责的输出范围 [k_curr, k_next)
//   - 用 co_rank 分别求出 k_curr 和 k_next 对应的 A、B 子数组边界
//   - 调用 merge_sequential 完成局部归并
//
// host 端需要：
//   - 创建输出张量（长度 m+n）
//   - 配置 grid 和 block，每个线程处理 ceil((m+n) / total_threads) 个元素
//   - 启动内核并返回结果

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
    // 算出 C 的前 k 个位置里，有几个(i)来自 A
    int i_low = max(0, k - n); 
    int i_high = min(m, k);
    
    while(true) {
        int i = i_low + (i_high - i_low) / 2;
        int j = k - i;
        
        // 1：判断是不是A选的太多了
        if (j < n && i > 0 && A[i - 1] > B[j]) {
            i_high = i - 1;
        } 
        // 2：判断是不是B选的太多了
        else if (j > 0 && i < m && A[i] < B[j - 1]) {
            i_low = i + 1;
        } else {
            return i;   
        }
    }
}

__global__ void MergeBasicKernel(float* A, int m, float* B, int n, float* C) {
    // 总共有totalThreads个线程，每个块里256个。一个块处理2048个元素，1个线程要排序8个元素存到C里
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    int elements_per_thread = cdiv(m + n, total_threads);
    
    // 计算当前线程的输出范围
    int k_curr = tid * elements_per_thread;
    int k_next = min((tid + 1) * elements_per_thread, m + n);
    if (k_curr >= m + n) return;
    
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    
    merge_sequential(&A[i_curr], i_next - i_curr, 
                     &B[j_curr], j_next - j_curr,
                     &C[k_curr]);
}

// 函数签名: torch::Tensor merge_basic(torch::Tensor A, torch::Tensor B)
torch::Tensor merge_basic(torch::Tensor A, torch::Tensor B) {
    int m = A.size(0);
    int n = B.size(0);
    auto C = torch::empty({m + n}, A.options());

    int totalThreads = BLOCK_SIZE * cdiv(m + n, BLOCK_SIZE * 8);
    dim3 block(BLOCK_SIZE);
    dim3 grid(cdiv(totalThreads, BLOCK_SIZE));

    MergeBasicKernel<<<grid, block>>>(
        A.data_ptr<float>(), m,
        B.data_ptr<float>(), n,
        C.data_ptr<float>());

    return C;
}
