#include <torch/extension.h>

// Frontier-based BFS with Shared Memory Privatization（私有化优化）
//
// 输入: 与 bfs_frontier 相同
//
// 思路：
//   - 在共享内存中维护一个局部前沿数组 currFrontier_s[LOCAL_FRONTIER_CAPACITY]
//   - 使用共享内存计数器 numCurrFrontier_s 跟踪局部前沿大小
//   - 每个线程处理前沿中的一个顶点，遍历其邻居
//   - 使用 atomicCAS 检查邻居是否未访问，成功则用 atomicAdd 加入局部前沿
//   - 如果局部前沿溢出（超过 LOCAL_FRONTIER_CAPACITY），直接用 atomicAdd 写入全局前沿
//   - __syncthreads() 后，线程 0 用 atomicAdd 在全局前沿中预留空间
//   - 所有线程协作将局部前沿批量拷贝到全局前沿
//
// 关键点：
//   - 两阶段提交减少全局原子操作竞争
//   - 溢出处理保证正确性
//   - 注意 levels 中未访问标记使用 UINT_MAX（而非 -1）

#define BLOCK_SIZE 256
#define LOCAL_FRONTIER_CAPACITY 256
#define cdiv(a, b) (((a) + (b) - 1) / (b))

__global__ void bfs_frontier_private_kernel(
    int* srcPtrs, int* dst, int* levels,
    int* prevFrontier, int* currFrontier,
    int numPrevFrontier, int* numCurrFrontier,
    int currLevel) {
    
    __shared__ int currFrontier_s[LOCAL_FRONTIER_CAPACITY];
    __shared__ int numCurrFrontier_s;
    __shared__ int global_offset_s;
    
    if (threadIdx.x == 0) {
        numCurrFrontier_s = 0;
    }
    __syncthreads();
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < numPrevFrontier) {
        int u = prevFrontier[tid];
        int start = srcPtrs[u];
        int end = srcPtrs[u + 1];
        
        for (int i = start; i < end; i++) {
            int v = dst[i];
            
            if (atomicCAS(&levels[v], -1, currLevel) == -1) {
                int local_pos = atomicAdd(&numCurrFrontier_s, 1);
                
                if (local_pos < LOCAL_FRONTIER_CAPACITY) {
                    currFrontier_s[local_pos] = v;
                } else {
                    int global_pos = atomicAdd(numCurrFrontier, 1);
                    currFrontier[global_pos] = v;
                }
            }
        }
    }
    __syncthreads();
    
    int local_count = min(numCurrFrontier_s, LOCAL_FRONTIER_CAPACITY);
    
    if (threadIdx.x == 0 && local_count > 0) {
        global_offset_s = atomicAdd(numCurrFrontier, local_count);
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < local_count; i += blockDim.x) {
        currFrontier[global_offset_s + i] = currFrontier_s[i];
    }
}

torch::Tensor bfs_frontier_private(torch::Tensor srcPtrs, torch::Tensor dst, torch::Tensor levels, int startVertex)
{
    int numVertices = levels.size(0);
    
    auto prevFrontier = torch::empty({numVertices}, torch::dtype(torch::kInt32).device(levels.device()));
    auto currFrontier = torch::empty({numVertices}, torch::dtype(torch::kInt32).device(levels.device()));
    auto numCurrFrontier = torch::empty({1}, torch::dtype(torch::kInt32).device(levels.device()));
    
    int* d_prevFrontier = prevFrontier.data_ptr<int>();
    int* d_currFrontier = currFrontier.data_ptr<int>();
    int* d_numCurrFrontier = numCurrFrontier.data_ptr<int>();
    
    int numPrevFrontier = 1;
    cudaMemcpy(d_prevFrontier, &startVertex, sizeof(int), cudaMemcpyHostToDevice);
    
    int zero = 0;
    cudaMemcpy(levels.data_ptr<int>() + startVertex, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    int currLevel = 1;
    while(numPrevFrontier > 0) {
        dim3 block(BLOCK_SIZE);
        dim3 grid(cdiv(numPrevFrontier, BLOCK_SIZE));
        
        cudaMemset(d_numCurrFrontier, 0, sizeof(int));
        
        bfs_frontier_private_kernel<<<grid, block>>>(
            srcPtrs.data_ptr<int>(), dst.data_ptr<int>(), levels.data_ptr<int>(),
            d_prevFrontier, d_currFrontier,
            numPrevFrontier, d_numCurrFrontier,
            currLevel
        );
        
        cudaMemcpy(&numPrevFrontier, d_numCurrFrontier, sizeof(int), cudaMemcpyDeviceToHost);
        
        int *tmp = d_prevFrontier;
        d_prevFrontier = d_currFrontier;
        d_currFrontier = d_prevFrontier;
        
        currLevel++;
    }
    return levels;
}
