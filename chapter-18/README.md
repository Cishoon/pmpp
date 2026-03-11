# Chapter 18

## Code

We implement all of the kernels discussed in Chapter 18. 

In particular, we implement:
- Scatter kernel and its corresponding host code from `Fig. 18.5`
- Gather kernel and its corresponding host code from `Fig. 18.6`
- Thread coarsening kernel and its corresponding host code from `Fig. 18.8`
- Thread coarsening and memory coalescing kernel and its corresponding host code from `Fig. 18.10`

For each kernel, we implement the corresponding host codebase. All implementations can be found in [cenergy.cu](./code/cenergy.cu). We implement a simple benchmark, comparing the performance between different approaches. The benchmark implementation can be found in [benchmark.cu](./code/benchmark.cu). Note that our benchmark includes copying the data from host memory to device constant memory via `cudaMemcpyToSymbol`. 

## Exercises

### Exercise 1

**Complete the host code for configuring the grid and calling the kernel in Fig. 18.6 with all the execution configuration parameters.**

Full implementation to be found in [cenergy.cu](./code/cenergy.cu). 

```cpp
void cenergyParallelGather(float* host_energygrid, dim3 grid_dim, float gridspacing, float z, const float* host_atoms,
                           int numatoms) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate device memory for the energy grid
    float* d_energygrid = NULL;
    size_t grid_size = grid_dim.x * grid_dim.y * grid_dim.z * sizeof(float);
    err = cudaMalloc((void**)&d_energygrid, grid_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for energy grid (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // Initialize energy grid to zero
    err = cudaMemset(d_energygrid, 0, grid_size);
    if (err != cudaSuccess) {
        printf("Failed to initialize device memory (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_energygrid);
        return;
    }

    // Process atoms in chunks
    int num_chunks = (numatoms + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Define 2D thread block and grid
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    dim3 blocksPerGrid((grid_dim.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (grid_dim.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // For each chunk of atoms
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        // Calculate the number of atoms in this chunk
        int start_atom = chunk * CHUNK_SIZE;
        int atoms_in_chunk = (start_atom + CHUNK_SIZE <= numatoms) ? CHUNK_SIZE : (numatoms - start_atom);

        // Copy this chunk to constant memory
        size_t chunk_size = atoms_in_chunk * 4 * sizeof(float);
        err = cudaMemcpyToSymbol(atoms, &host_atoms[start_atom * 4], chunk_size);
        if (err != cudaSuccess) {
            printf("Failed to copy atoms chunk %d to constant memory (error code %s)!\n", chunk,
                   cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        cenergyGatherKernel<<<blocksPerGrid, threadsPerBlock>>>(d_energygrid, grid_dim, gridspacing, z, atoms_in_chunk,
                                                                start_atom);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to launch kernel for chunk %d (error code %s)!\n", chunk, cudaGetErrorString(err));
            cudaFree(d_energygrid);
            return;
        }

        cudaDeviceSynchronize();
    }
```

### Exercise 2

**Compare the number of operations (memory loads, floating-point arithmetic, branches) executed in each iteration of the kernel in Fig. 18.8 with that in Fig. 18.6 for a coarsening factor of 8. Keep in mind that each iteration of the former corresponds to eight iterations of the latter.**

#### Fig. 18.6

```cpp
01 __constant__ float atoms[CHUNK_SIZE*4];
02 void __global__ cenergy(float *energygrid, dim3 grid, float gridspacing,
03                     float z, int numatoms) {
04     int i = blockIdx.x * blockDim.x + threadIdx.x;
05     int j = blockIdx.y * blockDim.y + threadIdx.y;
06     int atomarrdim = numatoms * 4;
07     int k = z / gridspacing;
08     float y = gridspacing * (float) j;
09     float x = gridspacing * (float) i;
10     float energy = 0.0f;
      // calculate potential contribution from all atoms
11     for (int n=0; n<atomarrdim; n+=4) {
12         float dx = x - atoms[n  ];
13         float dy = y - atoms[n+1];
14         float dz = z - atoms[n+2];
15         energy += atoms[n+3] / sqrtf(dx*dx + dy*dy + dz*dz);
16     }
17     energygrid[grid.x*grid.y*k + grid.x*j + i] += energy;
18 }
```

##### Memory load
We load from global memory (`atoms`) 4 times, in lines 12, 13, 14, and 15, for every point in the grid. Due to the `COARSEN_FACTOR` factor impact, we multiply it by `8` (we launch `8x` as many kernels in the grid as in the other kernel), so we end up with 4 x 8 = 32 loads from the global memory for every atom in the grid. 


##### Arithmetic operations
For every kernel, for all the atoms, we do:
- 3 substractions (lines 12, 13, and 14)
- 3 multiplications (line 15)
- 2 additions (line 15)
- 1 square root (line 15)
- 1 division (line 15)
- 1 more addition (line 15)

**11 operations** in total per thread. 

Due to the `COARSEN_FACTOR` factor impact we multiply it by `8` (we launch `8x` as many threads in the grid as in the other kernel), so we end up with *11 x 8 = 88* floating-point operations for every atom in the grid. 

##### Branches
1 branch per kernel per atom (line 11, the condition on our loop). 
Due to the `COARSEN_FACTOR` factor impact we multiply it by `8` (we launch `8x` as many threads in the grid as in the other kernel), so we end up with 1 x 8 = 8 branches.


#### Fig. 18.8
We had to slightly modify the kernel in `Fig. 18.8`. The original one hardcoded the COARSEN_FACTOR (the `energy0`, `energy1` ... thing in lines 25-28). This one works conceptually in the same way, but it allows for a more flexible thread coarsening scheme. 

```cpp
1  // Thread coarsening kernel - each thread processes COARSEN_FACTOR grid points
2  **global** void cenergyCoarsenKernel(float* *energygrid*, dim3 *grid*, float *gridspacing*, float *z*, int *atoms_in_chunk*) {
3  int base_i = blockIdx.x * blockDim.x * COARSEN_FACTOR + threadIdx.x * COARSEN_FACTOR;
4  int j = blockIdx.y * blockDim.y + threadIdx.y;
5  if (j >= *grid*.y) {
6    return;
7   }
8  int k = *z* / *gridspacing*;
9  float y = *gridspacing* * (float)j;
10  // Array to store energy values for COARSEN_FACTOR points
11  float energies[COARSEN_FACTOR];
12
13  // Initialize all energies to 0
14  for (int c = 0; c < COARSEN_FACTOR; c++) {
15    energies[c] = 0.0f;
16   }
17
18  // Calculate for all atoms
19  for (int n = 0; n < *atoms_in_chunk* * 4; n += 4) {
20    float dy = y - atoms[n + 1];
21    float dz = *z* - atoms[n + 2];
22    float dysqdzq = dy * dy + dz * dz; // Precalculate dy² + dz²
23    float charge = atoms[n + 3];
24
25    // Calculate energy for each of the COARSEN_FACTOR points
26    for (int c = 0; c < COARSEN_FACTOR; c++) {
27      int i = base_i + c;
28      if (i < grid.x) { // Check bounds
29        float x = gridspacing * (float)i;
30        float dx = x - atoms[n];
31        energies[c] += charge / sqrtf(dx * dx + dysqdzq);
32      }
33    }
34  }
35
36  // Write results
37  for (int c = 0; c < COARSEN_FACTOR; c++) {
38    int i = base_i + c;
39    if (i < *grid*.x) { // Check bounds again
40      *energygrid*[*grid*.x * grid.y * k + *grid*.x * j + i] += energies[c];
41    }
42  }
43 }
```

##### Memory load

In the outer loop we have three loads from the global memory (lines 20, 21, and 23). In the inner loop, we load from the global memory `COARSEN_FACTOR` times (line 30), in this case eight. So we have a total of 3 + 8 = 11 loads from the global memory. 

##### Arithmetic operations

In the outer loop we have:
- 2 subtractions (lines 20 & 21)
- 2 multiplications (line 22)
- 1 addition (line 22)

5 operations

In the inner loop we have:
- 1 multiplication (line 29)
- 1 subtraction (line 30)
- 1 multiplication (line 31)
- 1 addition (line 31)
- 1 square root (line 31)
- 1 division (line 31)
- 1 more addition (line 31)

Total of `7` times `COARSEN_FACTOR` or `7 x 8 = 56` operations. 

Bringing us to the total of *5 + 56 = 61* FLOPs. 

##### Branches

We have
Outer loop:
- 1 branch for the loop condition (line 19).

Inner loop:
- 1 branch for the loop condition (line 26).
- 1 branch for the boundary check `if (i < grid.x)`.

And the inner one is multiplied by the `COARSEN_FACTOR`.
So `1 + 2 x 8 = 17` branches. 



| Operation Type | Fig. 18.6 (Original) | Fig. 18.8 (Thread Coarsening) | Difference |
|----------------|----------------------|-------------------------------|------------|
| Memory Loads   | 32                   | 11                            | -21 (-65.6%) |
| Arithmetic Operations | 88            | 61                            | -27 (-30.7%) |
| Branches       | 8                    | 17                            | +9 (+112.5%) |


So, in the Thread Coarsening kernel, we have substantially fewer loads from the global memory and much improved arithmetic intensity. 

### Exercise 3

**Give two potential disadvantages associated with increasing the amount of work done in each CUDA thread, as shown in Section 18.3.**

1. We use substantially more registers. Depending on the hardware, if we use too many registers, it might reduce how many kernels we can launch at the same time, reducing the occupancy.
2. We are at risk of underutilizing the compute resources available. If we choose too high `COARSEN_FACTOR` our code will become de facto sequential, not taking enough advantage of the parallelism that GPUs enable. 

### Exercise 4

**Use Fig. 18.13 to explain how control divergence can arise when threads in a block process a bin in the neighborhood list.**

Depending on the number of atoms in the neighboring bins, different threads will need to iterate over different numbers of atoms. This can cause control divergence. There are some ways to mitigate this (like introducing dummy, 0-energy atoms), but they come at a cost (e.g., wasted space, increased memory bandwidth).


