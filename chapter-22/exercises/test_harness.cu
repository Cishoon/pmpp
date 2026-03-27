#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cuda_runtime.h>

// ============================================================
// 测试框架：GPU 参考实现 + 正确性验证 + 计时
// 由 Makefile 与各内核文件 + reference_kernels.cu 链接编译
// ============================================================

// ---- 参考实现（reference_kernels.cu 提供）----
void launch_reference_attention(
    const float* d_Q, const float* d_K, const float* d_V, float* d_O,
    int N, int d, bool causal);

// ---- 外部声明：由各内核文件提供 ----
#ifdef TEST_NAIVE
void launch_naive_attention(const float* d_Q, const float* d_K, const float* d_V, float* d_O, int N, int d);
#endif
#ifdef TEST_1PASS
void launch_flash_attention_1pass(const float* d_Q, const float* d_K, const float* d_V, float* d_O, int N, int d);
#endif
#ifdef TEST_TILED
void launch_flash_attention_tiled(const float* d_Q, const float* d_K, const float* d_V, float* d_O, int N, int d);
#endif
#ifdef TEST_CAUSAL
void launch_flash_attention_causal(const float* d_Q, const float* d_K, const float* d_V, float* d_O, int N, int d);
#endif

// ---- 颜色输出 ----
#define C_GREEN  "\033[92m"
#define C_RED    "\033[91m"
#define C_YELLOW "\033[93m"
#define C_BOLD   "\033[1m"
#define C_END    "\033[0m"

// ---- 随机初始化（host）----
void rand_init(float* data, int size, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
}

// ---- 正确性检查（host 端逐元素对比）----
bool check_correctness(const float* result, const float* expected, int count,
                       float rtol, float atol, float* max_diff_out) {
    float max_diff = 0.0f;
    bool ok = true;
    for (int i = 0; i < count; i++) {
        float diff = fabsf(result[i] - expected[i]);
        float tol = atol + rtol * fabsf(expected[i]);
        if (diff > tol) ok = false;
        if (diff > max_diff) max_diff = diff;
    }
    *max_diff_out = max_diff;
    return ok;
}

// ---- GPU 计时 ----
float benchmark_kernel(
    void (*launch_fn)(const float*, const float*, const float*, float*, int, int),
    const float* d_Q, const float* d_K, const float* d_V, float* d_O,
    int N, int d, int warmup, int reps
) {
    for (int i = 0; i < warmup; i++) {
        launch_fn(d_Q, d_K, d_V, d_O, N, d);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < reps; i++) {
        launch_fn(d_Q, d_K, d_V, d_O, N, d);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / reps;
}

// ---- 单次正确性测试 ----
struct TestConfig { int N; int d; };

bool run_single_test(
    void (*launch_fn)(const float*, const float*, const float*, float*, int, int),
    const char* name, int N, int d, bool causal
) {
    size_t qkv_bytes = N * d * sizeof(float);

    // host 分配
    float* h_Q       = (float*)malloc(qkv_bytes);
    float* h_K       = (float*)malloc(qkv_bytes);
    float* h_V       = (float*)malloc(qkv_bytes);
    float* h_O_test  = (float*)malloc(qkv_bytes);
    float* h_O_ref   = (float*)malloc(qkv_bytes);

    rand_init(h_Q, N * d, 42 + N);
    rand_init(h_K, N * d, 137 + N);
    rand_init(h_V, N * d, 256 + N);

    // device 分配
    float *d_Q, *d_K, *d_V, *d_O_test, *d_O_ref;
    cudaMalloc(&d_Q, qkv_bytes);
    cudaMalloc(&d_K, qkv_bytes);
    cudaMalloc(&d_V, qkv_bytes);
    cudaMalloc(&d_O_test, qkv_bytes);
    cudaMalloc(&d_O_ref, qkv_bytes);

    cudaMemcpy(d_Q, h_Q, qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_O_test, 0, qkv_bytes);
    cudaMemset(d_O_ref, 0, qkv_bytes);

    // GPU 参考实现
    launch_reference_attention(d_Q, d_K, d_V, d_O_ref, N, d, causal);

    // 学生实现
    launch_fn(d_Q, d_K, d_V, d_O_test, N, d);
    cudaDeviceSynchronize();

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  " C_RED "✗ FAIL" C_END "  %s N=%d d=%d  (CUDA error: %s)\n",
               name, N, d, cudaGetErrorString(err));
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
        cudaFree(d_O_test); cudaFree(d_O_ref);
        free(h_Q); free(h_K); free(h_V); free(h_O_test); free(h_O_ref);
        return false;
    }

    // 拷回 host 对比
    cudaMemcpy(h_O_test, d_O_test, qkv_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_O_ref, d_O_ref, qkv_bytes, cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    bool ok = check_correctness(h_O_test, h_O_ref, N * d, 1e-2f, 1e-2f, &max_diff);

    if (ok) {
        printf("  " C_GREEN "✓ PASS" C_END "  %s N=%d d=%d\n", name, N, d);
    } else {
        printf("  " C_RED "✗ FAIL" C_END "  %s N=%d d=%d  (max_diff=%.6f)\n",
               name, N, d, max_diff);
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O_test); cudaFree(d_O_ref);
    free(h_Q); free(h_K); free(h_V); free(h_O_test); free(h_O_ref);
    return ok;
}

// ---- 性能测试 ----
void run_benchmark(
    void (*launch_fn)(const float*, const float*, const float*, float*, int, int),
    const char* name, int N, int d
) {
    size_t qkv_bytes = N * d * sizeof(float);
    float* h_Q = (float*)malloc(qkv_bytes);
    float* h_K = (float*)malloc(qkv_bytes);
    float* h_V = (float*)malloc(qkv_bytes);
    rand_init(h_Q, N * d, 42);
    rand_init(h_K, N * d, 137);
    rand_init(h_V, N * d, 256);

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, qkv_bytes);
    cudaMalloc(&d_K, qkv_bytes);
    cudaMalloc(&d_V, qkv_bytes);
    cudaMalloc(&d_O, qkv_bytes);
    cudaMemcpy(d_Q, h_Q, qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_bytes, cudaMemcpyHostToDevice);

    float ms = benchmark_kernel(launch_fn, d_Q, d_K, d_V, d_O, N, d, 10, 50);
    printf("  %-28s  N=%-4d d=%-3d  %.3f ms\n", name, N, d, ms);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    free(h_Q); free(h_K); free(h_V);
}

// ============================================================
int main() {
    printf(C_BOLD "============================================================\n" C_END);
    printf(C_BOLD "  第二十二章 FlashAttention - 正确性验证 & 性能对比\n" C_END);
    printf(C_BOLD "============================================================\n" C_END);

    int total = 0, passed = 0;

    TestConfig cases[] = { {64, 32}, {128, 64}, {256, 64}, {512, 64} };
    int num_cases = sizeof(cases) / sizeof(cases[0]);

#ifdef TEST_NAIVE
    {
        printf(C_BOLD "\n[代码题 A] 朴素 Attention (naive_attention.cu)\n" C_END);
        for (int i = 0; i < num_cases; i++) {
            total++;
            if (run_single_test(launch_naive_attention, "朴素", cases[i].N, cases[i].d, false))
                passed++;
        }
    }
#endif

#ifdef TEST_1PASS
    {
        printf(C_BOLD "\n[代码题 B] FlashAttention 1-pass (flash_attention_1pass.cu)\n" C_END);
        for (int i = 0; i < num_cases; i++) {
            total++;
            if (run_single_test(launch_flash_attention_1pass, "1-pass", cases[i].N, cases[i].d, false))
                passed++;
        }
    }
#endif

#ifdef TEST_TILED
    {
        printf(C_BOLD "\n[代码题 C] FlashAttention 分块版 (flash_attention_tiled.cu)\n" C_END);
        for (int i = 0; i < num_cases; i++) {
            total++;
            if (run_single_test(launch_flash_attention_tiled, "分块", cases[i].N, cases[i].d, false))
                passed++;
        }
    }
#endif

#ifdef TEST_CAUSAL
    {
        printf(C_BOLD "\n[代码题 D] FlashAttention 因果掩码版 (flash_attention_causal.cu)\n" C_END);
        for (int i = 0; i < num_cases; i++) {
            total++;
            if (run_single_test(launch_flash_attention_causal, "因果", cases[i].N, cases[i].d, true))
                passed++;
        }
    }
#endif

    // 性能对比
    printf(C_BOLD "\n性能对比\n" C_END);
    int bench_configs[][2] = { {256, 64}, {512, 64}, {1024, 64}, {16384, 64} };
    int num_bench = 4;
    for (int b = 0; b < num_bench; b++) {
        int N = bench_configs[b][0], d = bench_configs[b][1];
        printf(C_BOLD "\n  N=%d, d=%d:\n" C_END, N, d);
#ifdef TEST_NAIVE
        run_benchmark(launch_naive_attention, "A 朴素", N, d);
#endif
#ifdef TEST_1PASS
        run_benchmark(launch_flash_attention_1pass, "B 1-pass", N, d);
#endif
#ifdef TEST_TILED
        run_benchmark(launch_flash_attention_tiled, "C 分块", N, d);
#endif
#ifdef TEST_CAUSAL
        run_benchmark(launch_flash_attention_causal, "D 因果", N, d);
#endif
    }

    // 总分
    printf(C_BOLD "\n============================================================\n" C_END);
    float pct = total > 0 ? (100.0f * passed / total) : 0.0f;
    const char* color = pct >= 80 ? C_GREEN : (pct >= 50 ? C_YELLOW : C_RED);
    printf("%s  代码题总分: %d/%d (%.0f%%)\n" C_END, color, passed, total, pct);
    printf(C_BOLD "============================================================\n" C_END);

    return (passed == total) ? 0 : 1;
}
