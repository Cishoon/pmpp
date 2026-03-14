#include <torch/extension.h>

// Vertex-Centric Push BFS
//
// 输入:
//   srcPtrs:  1D int32 张量，长度 numVertices+1，CSR 行指针
//   dst:      1D int32 张量，CSR 目标顶点数组
//   levels:   1D int32 张量，长度 numVertices，初始全 -1，起始顶点为 0
//   newVisited: 1D int32 张量，长度 1，标志位
//   currLevel: 当前 BFS 层级
//   numVertices: 顶点数
//
// 思路：
//   - 每个线程对应一个顶点
//   - 如果该顶点在上一层级（levels[vertex] == currLevel - 1），遍历其所有出边邻居
//   - 对于未访问的邻居（levels[neighbor] == -1），将其标记为 currLevel
//   - 设置 newVisited 标志为 1，表示还需要继续迭代
//
// host 端需要：
//   - 循环调用内核直到 newVisited 为 0
//   - 每次迭代前重置 newVisited 为 0
//   - 根据 numVertices 配置 grid 和 block

#define BLOCK_SIZE 256
#define cdiv(a, b) (((a) + (b) - 1) / (b))

__global__ void bfs_push_kernel(
    int* srcPtrs, int* dst, int* levels,
    int* newVisited, int currLevel, int numVertices) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < numVertices) {
        if (levels[tid] == currLevel - 1) {
            int start = srcPtrs[tid];
            int end = srcPtrs[tid + 1];
            
            for (int i = start; i < end; i++) {
                int v = dst[i];
                
                if (atomicCAS(&levels[v], -1, currLevel) == -1) {
                    *newVisited = 1;
                }
            }
        }
    }
}


torch::Tensor bfs_push(torch::Tensor srcPtrs, torch::Tensor dst, torch::Tensor levels, int startVertex)
{
    int numVertices = levels.size(0);
    
    int zero = 0;
    cudaMemcpy(levels.data_ptr<int>() + startVertex, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    auto newVisited = torch::zeros({1}, torch::dtype(torch::kInt32).device(srcPtrs.device()));
    int* d_newVisited = newVisited.data_ptr<int>();
    int h_newVisited = 1;
    
    int currLevel = 1;
    
    dim3 grid(cdiv(numVertices, BLOCK_SIZE));
    dim3 block(BLOCK_SIZE);
    
    while(h_newVisited != 0) {
        newVisited.fill_(0);
        
        bfs_push_kernel<<<grid, block>>>(
            srcPtrs.data_ptr<int>(), dst.data_ptr<int>(), levels.data_ptr<int>(), 
            d_newVisited, currLevel, numVertices
        );
        
        cudaMemcpy(&h_newVisited, d_newVisited, sizeof(int), cudaMemcpyDeviceToHost);
        currLevel++;
    }
    
    
    
    return levels;
}