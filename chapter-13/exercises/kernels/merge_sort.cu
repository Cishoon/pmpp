#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

// ==========================================
// 辅助与数学函数 (Device & Host)
// ==========================================
__device__ int cdiv_d(int a, int b) { return (a + b - 1) / b; }
__device__ int min_d(int a, int b) { return (a < b) ? a : b; }
int cdiv(int a, int b) { return (a + b - 1) / b; }

// Co-Rank 函数：二分查找分割点
__device__ int co_rank(int k, int* A, int m, int* B, int n) {
    int i_low = (0 > k - n) ? 0 : (k - n); 
    int i_high = (k < m) ? k : m;          

    while (i_low <= i_high) {
        int i = (i_low + i_high) / 2;
        int j = k - i;

        if (i > 0 && j < n && A[i - 1] > B[j]) {
            i_high = i - 1;
        } else if (j > 0 && i < m && B[j - 1] > A[i]) {
            i_low = i + 1;
        } else {
            return i;
        }
    }
    return 0;
}

// 顺序归并
__device__ void merge_sequential(int* A, int m, int* B, int n, int* C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] <= B[j]) C[k++] = A[i++];
        else C[k++] = B[j++];
    }
    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}

// ==========================================
// Kernel 1: 基础归并内核 (适用于早期 width 较小时)
// 逻辑：1个 Block 负责 1对子数组
// ==========================================
__global__ void merge_pass_kernel_basic(int* d_in, int* d_out, int N, int width) {
    int pair_idx = blockIdx.x;
    int start = pair_idx * 2 * width;
    if (start >= N) return;

    int mid = min_d(start + width, N);
    int end = min_d(start + 2 * width, N);

    int m = mid - start;
    int n = end - mid;
    int total = m + n;

    int chunk = cdiv_d(total, blockDim.x);
    int k_start_local = threadIdx.x * chunk;
    int k_end_local = min_d(k_start_local + chunk, total);

    if (k_start_local < total) {
        int i_start = co_rank(k_start_local, d_in + start, m, d_in + mid, n);
        int i_end = co_rank(k_end_local, d_in + start, m, d_in + mid, n);

        int j_start = k_start_local - i_start;
        int j_end = k_end_local - i_end;

        merge_sequential(
            d_in + start + i_start, i_end - i_start, 
            d_in + mid + j_start, j_end - j_start, 
            d_out + start + k_start_local
        );
    }
}

// ==========================================
// Kernel 2: 双重 Co-Rank 内核 (适用于中后期 width 巨大时)
// 逻辑：1个 Block 负责 C_BLOCK 长度的全局输出窗口
// ==========================================
__global__ void merge_pass_kernel_two_level(int* d_in, int* d_out, int N, int width, int C_BLOCK) {
    int global_k_start = blockIdx.x * C_BLOCK;
    if (global_k_start >= N) return;

    // 因为调用此内核的前提是 2*width >= C_BLOCK 且完美对齐
    // 所以这一个 C_BLOCK 必然属于且仅属于下面这一个 pair
    int pair_idx = global_k_start / (2 * width);
    int start = pair_idx * 2 * width;
    int mid = min_d(start + width, N);
    int end = min_d(start + 2 * width, N);

    int m = mid - start;
    int n = end - mid;
    int total = m + n;

    int k_start_block = global_k_start - start;
    int k_end_block = min_d(k_start_block + C_BLOCK, total);
    int block_total = k_end_block - k_start_block;

    if (block_total <= 0) return;

    int* A = d_in + start;
    int* B = d_in + mid;

    // Grid-Level Co-Rank
    int i_start_block = co_rank(k_start_block, A, m, B, n);
    int j_start_block = k_start_block - i_start_block;

    int i_end_block = co_rank(k_end_block, A, m, B, n);
    int j_end_block = k_end_block - i_end_block;

    int* A_block = A + i_start_block;
    int m_block = i_end_block - i_start_block;
    int* B_block = B + j_start_block;
    int n_block = j_end_block - j_start_block;

    // Block-Level Co-Rank
    int chunk = cdiv_d(block_total, blockDim.x);
    int k_start_thread = threadIdx.x * chunk;
    int k_end_thread = min_d(k_start_thread + chunk, block_total);

    if (k_start_thread < block_total) {
        int i_start_thread = co_rank(k_start_thread, A_block, m_block, B_block, n_block);
        int i_end_thread = co_rank(k_end_thread, A_block, m_block, B_block, n_block);

        int j_start_thread = k_start_thread - i_start_thread;
        int j_end_thread = k_end_thread - i_end_thread;

        merge_sequential(
            A_block + i_start_thread, i_end_thread - i_start_thread,
            B_block + j_start_thread, j_end_thread - j_start_thread,
            d_out + global_k_start + k_start_thread
        );
    }
}

// ==========================================
// 主机端调度函数 (PyTorch C++ 接口)
// ==========================================
torch::Tensor merge_sort(torch::Tensor input) {
    int N = input.size(0);
    auto current_in = input.clone();
    auto current_out = torch::empty_like(input);

    int* ptr_in = current_in.data_ptr<int>();
    int* ptr_out = current_out.data_ptr<int>();

    // 1. 动态获取当前显卡的最佳 Block Size
    int minGridSize, bestBlockSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &bestBlockSize, 
        merge_pass_kernel_two_level, 0, 0
    );

    // 2. 将 bestBlockSize 向下对齐到最近的 2的幂次方 (例如 768 -> 512)
    // 这是保证 C_BLOCK 能够完美整除 2*width 的数学基石
    int power2_block = 1;
    while (power2_block * 2 <= bestBlockSize) {
        power2_block *= 2;
    }
    bestBlockSize = power2_block;

    // 3. 设定每个 Block 的工作负载。让每个线程处理多个元素 (如 4 个)，提升计算访存比
    int elements_per_thread = 4;
    int C_BLOCK = bestBlockSize * elements_per_thread;

    int width = 1;
    int passes = 0;

    // 4. 自底向上的归并循环
    while (width < N) {
        if (2 * width < C_BLOCK) {
            // --- Phase 1: 早期阶段，基础内核 ---
            int numPairs = cdiv(N, 2 * width);
            merge_pass_kernel_basic<<<numPairs, bestBlockSize>>>(ptr_in, ptr_out, N, width);
        } else {
            // --- Phase 2: 中后期阶段，双重 Co-Rank 内核 ---
            // 彻底打破 "1对数组=1个Block" 的限制，恒定满载！
            int numBlocks = cdiv(N, C_BLOCK);
            merge_pass_kernel_two_level<<<numBlocks, bestBlockSize>>>(ptr_in, ptr_out, N, width, C_BLOCK);
        }

        std::swap(ptr_in, ptr_out);
        width *= 2;
        passes++;
    }

    if (passes % 2 != 0) {
        current_in.copy_(current_out);
    }

    return current_in;
}