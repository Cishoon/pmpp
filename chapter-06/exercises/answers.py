# ============================================================
# 第六章 简答题 & 计算题
# 在每个变量中填入你的答案，然后运行 run_tests.py 判题
# ============================================================

# --------------------------------------------------
# 习题 2: 分块矩阵乘法中，BLOCK_SIZE 取哪些值能完全避免非合并的全局内存访问？
#
# 选项:
#   "A" - BLOCK_SIZE 必须等于 32
#   "B" - BLOCK_SIZE 必须是 32 的整数倍
#   "C" - BLOCK_SIZE 必须是 2 的幂次
#   "D" - 任意 BLOCK_SIZE 都可以
# --------------------------------------------------
ex2 = "B"  # TODO

# --------------------------------------------------
# 习题 3: 判断以下访问是 coalesced、uncoalesced 还是 not_applicable（共享内存）
#
# 选项: "coalesced", "uncoalesced", "not_applicable"
# --------------------------------------------------
ex3a = "coalesced"  # TODO: 第05行 a[i]
ex3b = "not_applicable"  # TODO: 第05行 a_s[threadIdx.x]（共享内存）
ex3c = "coalesced"  # TODO: 第07行 b[j*blockDim.x*gridDim.x + i]
ex3d = "uncoalesced"  # TODO: 第07行 c[i*4 + j]
ex3e = "not_applicable"  # TODO: 第07行 bc_s[...]（共享内存）
ex3f = "not_applicable"  # TODO: 第10行 a_s[threadIdx.x]（共享内存）
ex3g = "coalesced"  # TODO: 第10行 d[i + 8]
ex3h = "not_applicable"  # TODO: 第11行 bc_s[threadIdx.x*4]（共享内存）
ex3i = "uncoalesced"  # TODO: 第11行 e[i*8]

# --------------------------------------------------
# 习题 4: 各内核的浮点运算与全局内存访问比（OP/B），保留两位小数
#
# 4a. 第三章朴素内核（无优化）
# --------------------------------------------------
ex4a = None  # TODO: 填浮点数，例如 0.25

# 4b. 第五章分块内核（32×32 分块）
ex4b = None  # TODO

# 4c. 本章线程粗化内核（32×32 分块 + 粗化因子 4）
ex4c = None  # TODO
