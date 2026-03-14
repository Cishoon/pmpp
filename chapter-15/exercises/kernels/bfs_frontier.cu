#include <torch/extension.h>

// Frontier-based Vertex-Centric BFS
//
// 输入:
//   srcPtrs:       1D int32 张量，CSR 行指针
//   dst:           1D int32 张量，CSR 目标顶点数组
//   levels:        1D int32 张量，BFS 层级数组
//   prevFrontier:  1D int32 张量，上一层前沿顶点列表
//   currFrontier:  1D int32 张量，当前层前沿（输出）
//   numPrevFrontier: 上一层前沿大小
//   numCurrFrontier: 1D int32 张量，长度 1，当前前沿计数器
//   currLevel: 当前 BFS 层级
//
// 思路：
//   - 只对前沿中的顶点启动线程（线程数 = numPrevFrontier）
//   - 每个线程取出前沿中的一个顶点，遍历其出边邻居
//   - 使用 atomicCAS 尝试将未访问邻居的 levels 从 -1 改为 currLevel
//   - 如果 atomicCAS 成功（返回 -1），用 atomicAdd 将邻居加入 currFrontier
//   - atomicCAS 保证每个顶点只被一个线程成功标记
//
// host 端需要：
//   - 初始前沿包含起始顶点
//   - 循环：拷贝前沿到 device，重置计数器，启动内核，读回新前沿大小
//   - 前沿为空时停止

#define BLOCK_SIZE 256
#define cdiv(a, b) (((a) + (b) - 1) / (b))

__global__ void bfs_frontier_kernel(
    int* srcPtrs, int* dst, int* levels,
    int* prevFrontier, int* currFrontier,
    int numPrevFrontier, int* numCurrFrontier,
    int currLevel) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < numPrevFrontier) {
        int u = prevFrontier[tid];
        int start = srcPtrs[u];
        int end = srcPtrs[u + 1];
        for (int i = start; i < end; i++) {
            int v = dst[i];
            if (atomicCAS(&levels[v], -1, currLevel) == -1) {
                int pos = atomicAdd(numCurrFrontier, 1);
                currFrontier[pos] = v;
            }
        }
    }
}

torch::Tensor bfs_frontier(torch::Tensor srcPtrs, torch::Tensor dst, torch::Tensor levels, int startVertex) {
    int numVertices = levels.size(0);
    
    auto prevFrontier = torch::zeros({numVertices}, torch::dtype(torch::kInt32).device(levels.device()));
    int* d_prevFrontier = prevFrontier.data_ptr<int>();
    int numPrevFrontier = 1;
    cudaMemcpy(prevFrontier.data_ptr<int>(), &startVertex, sizeof(int), cudaMemcpyHostToDevice);
    
    int zero = 0;
    cudaMemcpy(levels.data_ptr<int>() + startVertex, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    auto currFrontier = torch::zeros({numVertices}, torch::dtype(torch::kInt32).device(levels.device()));
    int* d_currFrontier = currFrontier.data_ptr<int>();
    auto numCurrFrontier = torch::zeros({1}, torch::dtype(torch::kInt32).device(levels.device()));
    
    int currLevel = 1;
    
    while(numPrevFrontier) {
        dim3 block(BLOCK_SIZE);
        dim3 grid(cdiv(numPrevFrontier, BLOCK_SIZE));
        
        cudaMemset(numCurrFrontier.data_ptr<int>(), 0, sizeof(int));
        
        bfs_frontier_kernel<<<grid, block>>>(
            srcPtrs.data_ptr<int>(), dst.data_ptr<int>(), levels.data_ptr<int>(),
            // prevFrontier.data_ptr<int>(), currFrontier.data_ptr<int>(), 
            d_prevFrontier, d_currFrontier,
            numPrevFrontier, numCurrFrontier.data_ptr<int>(),
            currLevel
        );
        
        currLevel++;
        cudaMemcpy(&numPrevFrontier, numCurrFrontier.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
        
        int* tmp = d_prevFrontier;
        d_prevFrontier = d_currFrontier;
        d_currFrontier = tmp;
    }
    
    return levels;
}
