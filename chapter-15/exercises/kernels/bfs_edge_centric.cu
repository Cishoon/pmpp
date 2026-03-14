#include <torch/extension.h>

// Edge-Centric BFS（使用 COO 格式）
//
// 输入:
//   edgeSrc:  1D int32 张量，长度 numEdges，每条边的源顶点
//   edgeDst:  1D int32 张量，长度 numEdges，每条边的目标顶点
//   levels:   1D int32 张量，长度 numVertices，初始全 -1，起始顶点为 0
//   newVisited: 1D int32 张量，长度 1，标志位
//   currLevel: 当前 BFS 层级
//   numEdges: 边数
//
// 思路：
//   - 每个线程对应一条边
//   - 检查该边的源顶点是否在上一层级（levels[src] == currLevel - 1）
//   - 如果是，且目标顶点未被访问（levels[dst] == -1），标记目标顶点为 currLevel
//   - 设置 newVisited 标志为 1
//
// host 端需要：
//   - 循环调用内核直到 newVisited 为 0
//   - 每次迭代前重置 newVisited 为 0
//   - 根据 numEdges 配置 grid 和 block

#define BLOCK_SIZE 256
#define cdiv(a, b) (((a) + (b) - 1) / (b))

__global__ void bfs_edge_centric_kernel(
    const int* __restrict__ edgeSrc, 
    const int* __restrict__ edgeDst, 
    int* __restrict__ levels,
    int* __restrict__ newVisited, 
    int currLevel, 
    int numEdges) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numEdges) {
        int u = edgeSrc[tid];
        
        // 检查源顶点是否属于上一层
        if (levels[u] == currLevel - 1) {
            int v = edgeDst[tid];
            
            // 检查目标顶点是否未被访问
            // 这里使用 atomicCAS 来严格保证竞态安全。
            // 即使多个线程(多条边)同时指向同一个未访问的 v，
            // 也只有一个线程会返回 -1 并将 newVisited 设为 1。
            if (atomicCAS(&levels[v], -1, currLevel) == -1) {
                // 标记本轮循环有新的顶点被访问到，告知 Host 端继续下一层
                // 注意：多个线程同时写入 1 是“良性数据竞争(Benign Data Race)”，
                // 在 CUDA 中写入相同的值是安全且高效的，因此不需要 atomic 操作。
                *newVisited = 1;
            }
        }
    }
}

torch::Tensor bfs_edge_centric(torch::Tensor edgeSrc, torch::Tensor edgeDst, torch::Tensor levels, int startVertex, int numVertices) {
    int numEdges = edgeSrc.size(0);
    
    // 2. 状态初始化
    int zero = 0;
    cudaMemcpy(levels.data_ptr<int>() + startVertex, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    auto newVisited = torch::zeros({1}, torch::dtype(torch::kInt32).device(levels.device()));
    int* d_newVisited = newVisited.data_ptr<int>();
    
    // Host 端的控制标志
    int h_newVisited = 1; 
    int currLevel = 1;
    
    // 3. 循环启动 Kernel
    while (h_newVisited) {
        dim3 block(BLOCK_SIZE);
        dim3 grid(cdiv(numEdges, BLOCK_SIZE));
        
        // 每一层开始前，重置标志位
        cudaMemset(d_newVisited, 0, sizeof(int));
        
        bfs_edge_centric_kernel<<<grid, block>>>(
            edgeSrc.data_ptr<int>(), 
            edgeDst.data_ptr<int>(), 
            levels.data_ptr<int>(),
            d_newVisited, 
            currLevel, 
            numEdges
        );
        
        // 将标志位拷贝回 Host 端，决定是否继续
        cudaMemcpy(&h_newVisited, d_newVisited, sizeof(int), cudaMemcpyDeviceToHost);
        
        currLevel++;
    }
    
    return levels;
}