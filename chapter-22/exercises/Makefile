NVCC      = /usr/local/cuda-12.4/bin/nvcc
NVCCFLAGS = -O2 -std=c++17
KERNELS   = kernels
REF       = reference_kernels.cu

# V=0 使用 naive_attention.cu（默认）
# V=1 使用 naive_attention_1.cu（tiled matmul）
# V=2, V=3 ... 以此类推
V ?= 0
ifeq ($(V),0)
  NAIVE_SRC = $(KERNELS)/naive_attention.cu
else
  NAIVE_SRC = $(KERNELS)/naive_attention_$(V).cu
endif

# 默认目标：编译所有
all: test_all test_naive test_1pass test_tiled test_causal

# 全部内核一起编译
test_all: test_harness.cu $(REF) $(NAIVE_SRC) $(KERNELS)/flash_attention_1pass.cu $(KERNELS)/flash_attention_tiled.cu $(KERNELS)/flash_attention_causal.cu
	$(NVCC) $(NVCCFLAGS) -DTEST_NAIVE -DTEST_1PASS -DTEST_TILED -DTEST_CAUSAL \
		$^ -o $@

# 单独编译各题
test_naive: test_harness.cu $(REF) $(NAIVE_SRC)
	$(NVCC) $(NVCCFLAGS) -DTEST_NAIVE $^ -o $@

test_1pass: test_harness.cu $(REF) $(KERNELS)/flash_attention_1pass.cu
	$(NVCC) $(NVCCFLAGS) -DTEST_1PASS $^ -o $@

test_tiled: test_harness.cu $(REF) $(KERNELS)/flash_attention_tiled.cu
	$(NVCC) $(NVCCFLAGS) -DTEST_TILED $^ -o $@

test_causal: test_harness.cu $(REF) $(KERNELS)/flash_attention_causal.cu
	$(NVCC) $(NVCCFLAGS) -DTEST_CAUSAL $^ -o $@

clean:
	rm -f test_all test_naive test_1pass test_tiled test_causal

.PHONY: all clean
