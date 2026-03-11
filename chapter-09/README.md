# Chapter 9

## Code

We implement various methods of calculating the values of the histogram, benchmark, and compare them in [histogram.cu](code/histogram.cu). To execute the benchmark run:

```bash
cd code

nvcc histogram.cu -o histogram

./histogram
```

## Exercises

### Exercise 1
**Assume that each atomic operation in a DRAM system has a total latency of 100 ns. What is the maximum throughput that we can get for atomic operations on the same global memory variable?**

Each operation is 1 ns, or in other words, it is 1 ns per operation `10^-9 s/op`. In one second we can do `1s / (100 * 10^-9 s/op) = 10^7 op/s`, so our throughput is 10M operations per second.

### Exercise 2
**For a processor that supports atomic operations in L2 cache, assume that each atomic operation takes 4 ns to complete in L2 cache and 100 ns to complete in DRAM. Assume that 90% of the atomic operations hit in L2 cache. What is the approximate throughput for atomic operations on the same global memory variable?**

In the standard workload, 90% of the atomic operations hit the L2 cache memory, and the remaining 10% hit the DRAM. So on average an operation takes `0.9 x 4 ns + 0.1 x 100 ns = 13.6 ns`. Given that the throughput will be `1s / (13.6 op/s * 10^-9) = 0.073 * 10^9 = 73 * 10^6` or 73M operations per second. 

*since we are operating on the same global variable these atomic operations need to be sequential. 

### Exercise 3
**In Exercise 1, assume that a kernel performs five floating-point operations per atomic operation. What is the maximum floating-point throughput of the kernel execution as limited by the throughput of the atomic operations?**

As we calculated in **Exercise 1**, the throughput bound by the atomic operations is 10M per second. To find the floating point throughput, we need to multiply the number of floating point operations per atomic operation with the number of atomic operations per second. 

This brings us to (5 FP ops/Atomic op) x (10M Atomic ops/s) = 50 MFLOPS, which is considerably below the flops achieved by the modern GPUs (by multiple orders of magnitude).

### Exercise 4
**In Exercise 1, assume that we privatize the global memory variable into shared memory variables in the kernel and that the shared memory access latency is 1 ns. All original global memory atomic operations are converted into shared memory atomic operation. For simplicity, assume that the additional global memory atomic operations for accumulating privatized variable into the global variable adds 10% to the total execution time. Assume that a kernel performs five floating-point operations per atomic operation. What is the maximum floating-point throughput of the kernel execution as limited by the throughput of the atomic operations?**

To calculate the floating point throughput first we need to calculate the atomic operations throughput. This is quite straightforward. 

time per atomic operation times, plus 10% for accumulating privitized variable into the global memory: `1s / (1ns/op x 10^9 x 1.1) = 0.91 x 10^9 op/s = 9.1 x 10^8 op/s`. 

Now we multiply it with the `5 FP ops/Atomic op` and we get `45.5 x 10^8 = 4.55 GFLOPS` - still like 5 orders of magnitude below the theoretical peak performance of a modern GPU like H100. 


### Exercise 5
**To perform an atomic add operation to add the value of an integer variable Partial to a global memory integer variable Total, which one of the following statements should be used?**

**a. atomicAdd(Total, 1);**

**b. atomicAdd(&Total, &Partial);**

**c. atomicAdd(Total, &Partial);**

**d. atomicAdd(&Total, Partial);**

The `atomicAdd` function has the following interface:

```cpp
int atomicAdd(int* address, int val);
```

address of the value to be updated followed by the integer value. So the answer is `D`—the address of the variable `Total` and the value of the variable `Partial`.

### Exercise 6
**Consider a histogram kernel that processes an input with 524,288 elements to produce a histogram with 128 bins. The kernel is configured with 1024 threads per block.**

**a. What is the total number of atomic operations that are performed on global memory by the kernel in Fig. 9.6 where no privatization, shared memory, and thread coarsening are used?**

In the `Fig. 9.6` kernel we perform the `atomicAdd` on the global memory for every single thread that is being executed (line 06). Since we launched 524,288 kernels, there will be 524,288 atomic operations.

**b. What is the maximum number of atomic operations that may be performed on global memory by the kernel in Fig. 9.10 where privatization and shared memory are used but not thread coarsening?**

For `Fig. 9.10` we perform an atomic operation on the global memory only after all the values were already calculated in the private copy of the histogram stored in the shared memory. This makes it significantly cheaper than the previous example. We perform the `atomicAdd` once per thread in a block if the `threadId` is `< NUM_BINS`—in` our case, `128` times per each thread block. 

We have 524,288 elements in the list, and each block we run has 1024 threads; therefore, we need to run `(524288 + 1024 - 1)/1024 = 512` blocks. 

So in total we will perform `number of blocks x NUM_BINS = 512 x 128 = 65536` `atomicAdd` operations. 


**c. What is the maximum number of atomic operations that may be performed on global memory by the kernel in Fig. 9.14 where privatization, shared memory, and thread coarsening are used with a coarsening factor of 4?**

The coarsening affects how many blocks will be run, since here we are running with a factor of 4; the number of blocks in the grid will be reduced to `512 / 4 = 128` blocks. While commit itself doesn't change the commit mechanism (copying the private copies to the global memory via an atomic operation), since the number of blocks changed, the number of `atomicAdd` operations will be affected as well. Similarly to **6b**, we will have `blocks x NUM_BINS = 128 x 128 = 16384` atomic operations.
