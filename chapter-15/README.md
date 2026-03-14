# Chapter 15

## Code

We implement (nearly) all methods mentioned in this chapter in particular:

- Vertex Centric Push 
- Vertex Centric Pull
- Vertex Centric direction optimized, first push then pull
- Edge Centric
- Frontier Vertex Centric (with and without privatization)
- Frontier Vertex Centric with the single-block BFS optimization from section `15.7` 

You can find all of them in [bfs_parallel.cu](./code/src/bfs_parallel.cu). 

We also benchmark the performance on synthetically generated graphs using [scale-free and small-world](./code/src/graph_generators.cu) methods. 

Compile the benchmark using

```bash
make
```

and run it via

```bash
./bfs
```

You should see something resembling:

```
BFS Benchmark Results
=====================

Verifying BFS implementations correctness...
All BFS implementations passed correctness verification!

Graph Size: 1000 vertices
--------------------
Generating scale-free graph...
Allocating graphs on device...
Verifying BFS correctness for 1000 vertices graph... Passed!
Sequential BFS: 0.37 ms
Push Vertex-Centric BFS: 0.33 ms (1.12x speedup)
Pull Vertex-Centric BFS: 0.14 ms (2.57x speedup)
Edge-Centric BFS: 0.08 ms (4.89x speedup)
Frontier-based BFS: 0.59 ms (0.63x speedup)
Optimized Frontier-based BFS: 0.60 ms (0.62x speedup)
Direction-Optimized BFS: 0.13 ms (2.82x speedup)

Graph Size: 5000 vertices
--------------------
Generating scale-free graph...
Allocating graphs on device...
Verifying BFS correctness for 5000 vertices graph... Passed!
Sequential BFS: 2.11 ms
Push Vertex-Centric BFS: 0.71 ms (2.97x speedup)
Pull Vertex-Centric BFS: 0.24 ms (8.82x speedup)
Edge-Centric BFS: 0.10 ms (20.50x speedup)
Frontier-based BFS: 1.24 ms (1.70x speedup)
Optimized Frontier-based BFS: 1.25 ms (1.69x speedup)
Direction-Optimized BFS: 0.24 ms (8.71x speedup)

Graph Size: 10000 vertices
--------------------
Generating scale-free graph...
Allocating graphs on device...
Verifying BFS correctness for 10000 vertices graph... Passed!
Sequential BFS: 4.71 ms
Push Vertex-Centric BFS: 1.00 ms (4.69x speedup)
Pull Vertex-Centric BFS: 0.32 ms (14.57x speedup)
Edge-Centric BFS: 0.13 ms (35.46x speedup)
Frontier-based BFS: 1.77 ms (2.66x speedup)
Optimized Frontier-based BFS: 1.83 ms (2.58x speedup)
Direction-Optimized BFS: 0.35 ms (13.57x speedup)

Graph Size: 20000 vertices
--------------------
Generating scale-free graph...
Allocating graphs on device...
Verifying BFS correctness for 20000 vertices graph... Passed!
Sequential BFS: 10.26 ms
Push Vertex-Centric BFS: 1.44 ms (7.12x speedup)
Pull Vertex-Centric BFS: 0.44 ms (23.53x speedup)
Edge-Centric BFS: 0.19 ms (54.81x speedup)
Frontier-based BFS: 2.49 ms (4.11x speedup)
Optimized Frontier-based BFS: 2.60 ms (3.95x speedup)
Direction-Optimized BFS: 0.48 ms (21.54x speedup)

Small-World Graph (10000 vertices)
--------------------
Generating small-world graph...
Verifying BFS correctness for small-world graph... Passed!
Sequential BFS: 1.14 ms
Push Vertex-Centric BFS: 0.20 ms (5.80x speedup)
Pull Vertex-Centric BFS: 0.16 ms (7.23x speedup)
Edge-Centric BFS: 0.13 ms (9.13x speedup)
Frontier-based BFS: 0.32 ms (3.54x speedup)
Optimized Frontier-based BFS: 0.31 ms (3.71x speedup)
Direction-Optimized BFS: 0.23 ms (4.91x speedup)
````


## Exercises

### Exercise 1

**Consider the following directed unweighted graph:**

![Source graph](exercise_1_question.png)

**a. Represent the graph using an adjacency matrix.**

![Adjacency matrix](exercise_1a.png)

**b. Represent the graph in the CSR format. The neighbor list of each vertex must be sorted.**

![Adjacency matrix in CSR format](exercise_1b.png)

**Parallel BFS is executed on this graph starting from vertex 0 (i.e., vertex 0 is in level 0). For each iteration of the BFS traversal:**

**i. If a vertex-centric push implementation is used:**

### Iteration 1

**How many threads are launched?**

We have 8 vertices - hence 8 threads and launched. 

**How many threads iterate over their vertex’s neighbors?**

Only vertex 0 will iterate over its neighbours, so only 1 thread. 

### Iteration 2

**How many threads are launched?**

We have 8 vertices - hence 8 threads and launched. 

**How many threads iterate over their vertex’s neighbors?**

Two threads, for vectex `5` and for vertex `2` will iterate over their neighbours. 

### Iteration 3

**How many threads are launched?**

Same as above, 8 vertices - 8 threads. 

**How many threads iterate over their vertex’s neighbors?**

Now we have three vertices at level 2: `1`, `7` and `3`, so three vertices will iterate over their neighbours. 

### Iteration 4

**How many threads are launched?**

Again, 8 vertices - 8 threads. 

**How many threads iterate over their vertex’s neighbors?**

We have two vertices at level 3: `4` and `6` - so two threads will iterate. 

This is also the last iteration, all of the vertices are already visited so, no new vertices will be added and none of the therads will set flag `newVertexVisited` to 1. 

**ii. If a vertex-centric pull implementation is used:**

### Iteration 1
**How many threads are launched?**

We have 8 verices so we launch 8 threads. 

**How many threads iterate over their vertex’s neighbors?**

7 threads - all but thread for vertex `0` will iterate over its neighbours. 

**How many threads label their vertex?**

Two threads, for vertex `5` and for vertex `2` will label their vertex.

### Iteration 2
**How many threads are launched?**

We have 8 verices so we launch 8 threads. 

**How many threads iterate over their vertex’s neighbors?**

5 threads, for vertices `1`, `7`, `6`, `3`, `4` will iterate over their neighbours. 

**How many threads label their vertex?**

3 threads, for vertex `1`, `7`, and `3` will label their vertex. 

### Iteration 3
**How many threads are launched?**

Yet again - 8 vertices 8 threads launched. 

**How many threads iterate over their vertex’s neighbors?**

2 threads, for vertex `6`, and for vertex `4` will iterate over their neighbours. 

**How many threads label their vertex?**

Two threads will label their vertex. 

### Iteration 4

**How many threads are launched?**

Yet again - 8 vertices 8 threads launched. 

**How many threads iterate over their vertex’s neighbors?**

None, at this point we don't have any more unvisited vertices. 

**How many threads label their vertex?**

As above - none. This is also our last iteration as none of the threads will set flag `newVertexVisited` to 1. 

**iii If an edge-centric implementation is used:**

### Iteration 1

**How many threads are launched?**

We have 15 edges between the vertices, so 15 threads.

**How many threads may label a vertex?**

Two threads, for edge `0 -> 5` and for for edge `0 -> 2`, label their vertex. 

### Iteration 2

**How many threads are launched?**

Again, we have 15 edges so we launch 15 threads. 

**How many threads may label a vertex?**

Three threads, for edge `5 -> 1`, `5 -> 7` and `2 -> 3`

### Iteration 3

**How many threads are launched?**

Again, we have 15 edges so we launch 15 threads. 

**How many threads may label a vertex?**

Four threads, for edges `3 -> 6`, `7 -> 6`, `1 -> 4`, `7 -> 4`, note that two of these operations are ***impotent*** as they don't really change the label assigned to the vertex. 

### Iteration 4

**How many threads are launched?**

Again, we have 15 edges so we launch 15 threads. 

**How many threads may label a vertex?**

0, we have 2 vertices at level 3, but all its neighbours were already visited, hence none thread will label a vertex. This is also our last iteration, as none of the threads will modify the flag `newVertexVisited`


**iv. If a vertex-centric push frontier-based implementation is used:**

We assume that iteration 1 will start with `[0]` in `prevFrontier`.

### Iteration 1
**How many threads are launched?**

We launch only 1 thread for vertex 0.

**How many threads iterate over their vertex’s neighbors?**

Only one thread iterate over their vertex neighbours. We add `[5, 2]` to `currFrontier`. 

### Iteration 2
**How many threads are launched?**

We have two elements in `prevFrontier`, so two threads are launched. 

**How many threads iterate over their vertex’s neighbors?**

Two threads, for vertices `5` and `2` are iterating over their vertex neighbours. We add `[1, 7, 3]` to `currFrontier`. 

### Iteration 3
**How many threads are launched?**

We have three elements in `prevFrontier`, so three threads are launched. 

**How many threads iterate over their vertex’s neighbors?**

Three threads, for vertices `1`, `7` and `3`, are iterting over their vertex neighbours. We add `[4, 6]` to the `currFrontier`, not that depending on which thread will access it first `6` will be added either by thread for vertex `3` or a thread for vertex `7`.

### Iteration 4
**How many threads are launched?**

Two elements in `prevFrontier` - `[4, 6]`, so two threads are launched. 

**How many threads iterate over their vertex’s neighbors?**

Two of them, but there is no vertices that are still not visited, hence no will be added to the `currFrontier`, which will signal to the function that we don't need to launch the grid anymore. 

### Exercise 2

**Implement the host code for the direction-optimized BFS implementation described in Section 15.3**

Full impmenetnation to be found in [bfs_parallel.cu](./code/src/bfs_parallel.cu):

```cpp
int* bfsDirectionOptimizedDevice(const CSRGraph& deviceCSRGraph, const CSCGraph& deviceCSCGraph, int startingNode, float alpha) {
    // Initialize host levels array
    int* hostLevels = (int*)malloc(sizeof(int) * deviceCSRGraph.numVertices);
    for (int i = 0; i < deviceCSRGraph.numVertices; i++) {
        hostLevels[i] = -1;
    }
    hostLevels[startingNode] = 0;

    // Calculate size needed for vertex array
    size_t vertexSize = sizeof(int) * deviceCSRGraph.numVertices;

    // Allocate device memory for levels and flag
    int *d_levels, *d_newVertexVisited;
    cudaMalloc(&d_levels, vertexSize);
    cudaMalloc(&d_newVertexVisited, sizeof(int));

    // Copy initial levels to device
    cudaMemcpy(d_levels, hostLevels, vertexSize, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (deviceCSRGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;

    int currLevel = 1;
    int hostNewVertexVisited = 1;
    
    // Track visited vertices for determining when to switch
    int totalVertices = deviceCSRGraph.numVertices;
    int visitedVertices = 1; // Starting with just the root
    bool usingPush = true; // Start with push strategy
    
    // we iterate over the levels as long as any vertex reports finding a new unvisited neighbour
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice);

        // Decide whether to use push or pull based on the fraction of vertices visited
        float visitedFraction = (float)visitedVertices / totalVertices;
        if (usingPush && visitedFraction > alpha) {
            // Switch to pull strategy if we've visited more than alpha fraction of vertices
            usingPush = false;
        }
        
        if (usingPush) {
            // Use push strategy
            bsf_push_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                deviceCSRGraph, d_levels, d_newVertexVisited, currLevel);
        } else {
            // Use pull strategy
            bsf_pull_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                deviceCSCGraph, d_levels, d_newVertexVisited, currLevel);
        }

        cudaDeviceSynchronize();
        cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost);
        
        // If new vertices were visited, update our count
        if (hostNewVertexVisited) {
            // For simplicity, we'll count the visited vertices by copying back and checking
            // In a more optimized implementation, we'd track this on the device
            cudaMemcpy(hostLevels, d_levels, vertexSize, cudaMemcpyDeviceToHost);
            int newVisitedCount = 0;
            for (int i = 0; i < totalVertices; i++) {
                if (hostLevels[i] != -1) {
                    newVisitedCount++;
                }
            }
            visitedVertices = newVisitedCount;
        }
        
        currLevel++;
    }

    // Copy final results back to host
    cudaMemcpy(hostLevels, d_levels, vertexSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_levels);
    cudaFree(d_newVertexVisited);

    return hostLevels;
}
```

### Exercise 3

**Implement the single-block BFS kernel described in Section 15.7.**

Full impmenetnation to be found in [bfs_parallel.cu](./code/src/bfs_parallel.cu):

```cpp
__global__ void bfs_multi_level_frontier_kernel(CSRGraph graph, int* levels, 
                                         int* frontier, int* frontierSize,
                                         int* nextFrontier, int* nextFrontierSize,
                                         int currLevel) {
    // Shared memory for local frontier management
    __shared__ int localFrontier[LOCAL_FRONTIER_CAPACITY];
    __shared__ int localFrontierSize;
    __shared__ int nextLocalFrontierSize;
    __shared__ bool overflowed;
    
    // Initialize shared variables
    if (threadIdx.x == 0) {
        localFrontierSize = *frontierSize;
        nextLocalFrontierSize = 0;
        overflowed = false;
        
        // Copy initial frontier to shared memory
        for (int i = 0; i < localFrontierSize && i < LOCAL_FRONTIER_CAPACITY; i++) {
            localFrontier[i] = frontier[i];
        }
    }
    
    __syncthreads();
    
    // Process current frontier vertices
    for (int i = threadIdx.x; i < localFrontierSize; i += blockDim.x) {
        int vertex = localFrontier[i];
        
        // Explore neighbors
        for (unsigned int edge = graph.srcPtrs[vertex]; 
             edge < graph.srcPtrs[vertex + 1]; edge++) {
            unsigned int neighbor = graph.dst[edge];
            
            // If neighbor not visited, add to next frontier
            if (atomicCAS(&levels[neighbor], -1, currLevel) == -1) {
                int idx = atomicAdd(&nextLocalFrontierSize, 1);
                
                // Check if the next frontier exceeds capacity
                if (idx < LOCAL_FRONTIER_CAPACITY) {
                    localFrontier[idx] = neighbor;
                } else {
                    // Mark as overflowed - need to switch to multi-block kernel
                    overflowed = true;
                    
                    // Add to global frontier directly
                    int globalIdx = atomicAdd(nextFrontierSize, 1);
                    nextFrontier[globalIdx] = neighbor;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Update for next iteration or return results
    if (threadIdx.x == 0) {
        if (overflowed) {
            // Copy remaining valid local frontier entries to global
            for (int i = 0; i < nextLocalFrontierSize && i < LOCAL_FRONTIER_CAPACITY; i++) {
                int globalIdx = atomicAdd(nextFrontierSize, 1);
                nextFrontier[globalIdx] = localFrontier[i];
            }
            // Signal that we need to switch to multi-block
            *frontierSize = 0;
        } else {
            // We can continue with single-block
            *frontierSize = nextLocalFrontierSize;
            
            // Copy the next frontier for the host
            for (int i = 0; i < nextLocalFrontierSize; i++) {
                frontier[i] = localFrontier[i];
            }
        }
    }
}
```