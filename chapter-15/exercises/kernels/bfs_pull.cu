#include <torch/extension.h>

// Vertex-Centric Pull BFS（使用 CSC 格式）
//
// 输入:
//   dstPtrs:  1D int32 张量，长度 numVertices+1，CSC 列指针（入边指针）
//   src:      1D int32 张量，CSC 源顶点数组
//   levels:   1D int32 张量，长度 numVertices，初始全 -1，起始顶点为 0
//   newVisited: 1D int32 张量，长度 1，标志位
//   currLevel: 当前 BFS 层级
//   numVertices: 顶点数
//
// 思路：
//   - 每个线程对应一个顶点
//   - 如果该顶点尚未被访问（levels[vertex] == -1），遍历其所有入边邻居
//   - 如果任一入边邻居在上一层级（levels[neighbor] == currLevel - 1），标记自己为 currLevel
//   - 找到一个即可 break，设置 newVisited 标志为 1
//
// host 端需要：
//   - 循环调用内核直到 newVisited 为 0
//   - 每次迭代前重置 newVisited 为 0
//   - 根据 numVertices 配置 grid 和 block

#define BLOCK_SIZE 256
#define cdiv(a, b) (((a) + (b) - 1) / (b))

__global__ void bfs_pull_kernel(
    int* dstPtrs, int* src, int* levels,
    int* newVisited, int currLevel, int numVertices) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numVertices) {
        if (levels[tid] == -1) {
            int start = dstPtrs[tid];    
            int end = dstPtrs[tid + 1];    
            
            for (int i = start; i < end; i++) {
                int u = src[i];
                if (levels[u] == currLevel - 1) {
                    *newVisited = 1;
                    levels[tid] = currLevel;
                    break;
                }
            }
        }
    }
}

torch::Tensor bfs_pull(torch::Tensor dstPtrs, torch::Tensor src, torch::Tensor levels, int startVertex)
{
    int numVertices = levels.size(0);
    
    int zero = 0;
    cudaMemcpy(levels.data_ptr<int>() + startVertex, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 grid(cdiv(numVertices, BLOCK_SIZE));
    dim3 block(BLOCK_SIZE);
    
    int currLevel = 1;
    auto newVisited = torch::zeros({1}, torch::dtype(torch::kInt32).device(levels.device()));
    int h_newVisited = 1;
    int* d_newVisited = newVisited.data_ptr<int>();
    
    while(h_newVisited != 0) {
        newVisited.fill_(0);
        
        bfs_pull_kernel<<<grid, block>>>(
            dstPtrs.data_ptr<int>(), src.data_ptr<int>(), levels.data_ptr<int>(),
            newVisited.data_ptr<int>(), currLevel, numVertices
        );
        
        cudaMemcpy(&h_newVisited, d_newVisited, sizeof(int), cudaMemcpyDeviceToHost);
        currLevel++;
    }
    
    return levels;
}
