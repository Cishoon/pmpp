#!/usr/bin/env python3
"""
第十四章 一键判题脚本
用法: python run_tests.py
"""
import os
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

import torch
import torch.sparse
from torch.utils.cpp_extension import load_inline

torch.cuda.set_device(3)  
KERNELS_DIR = Path(__file__).parent / "kernels"
SCORE = {"total": 0, "passed": 0}
REPS = 200


def colored(text, color):
    colors = {
        "green": "\033[92m", "red": "\033[91m",
        "yellow": "\033[93m", "bold": "\033[1m", "end": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['end']}"


def report(name, passed, detail=""):
    SCORE["total"] += 1
    if passed:
        SCORE["passed"] += 1
    status = colored("✓ PASS", "green") if passed else colored("✗ FAIL", "red")
    msg = f"  {status}  {name}"
    if detail and not passed:
        msg += f"  ({detail})"
    print(msg)


def compile_kernel(cu_file, cpp_sources, functions, ext_name):
    cuda_source = (KERNELS_DIR / cu_file).read_text()
    return load_inline(
        name=ext_name,
        cpp_sources=cpp_sources,
        cuda_sources=cuda_source,
        functions=functions,
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
    )


def benchmark_fn(fn, *args, warmup=20):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPS):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / REPS


# ============================================================
# 稀疏矩阵生成工具
# ============================================================

def make_sparse_matrix(numRows, numCols, density=0.1, device="cuda"):
    """生成随机稀疏矩阵，返回 COO 格式的各数组和稠密矩阵（向量化实现）"""
    nnz_per_row = max(1, int(numCols * density))
    # 每行随机选 nnz_per_row 个列，批量生成
    col_indices = torch.stack([torch.randperm(numCols)[:nnz_per_row].sort()[0]
                               for _ in range(numRows)])  # (numRows, nnz_per_row)
    row_indices = torch.arange(numRows).unsqueeze(1).expand_as(col_indices)  # (numRows, nnz_per_row)

    rowIdx = row_indices.reshape(-1).to(dtype=torch.int32, device=device)
    colIdx = col_indices.reshape(-1).to(dtype=torch.int32, device=device)
    values = torch.randn(numRows * nnz_per_row, dtype=torch.float32, device=device)

    dense = torch.zeros(numRows, numCols, dtype=torch.float32, device=device)
    dense[rowIdx.long(), colIdx.long()] = values

    return rowIdx, colIdx, values, dense


def coo_to_csr_ref(rowIdx, numRows):
    """将 COO 的 rowIdx 转换为 CSR 的 rowPtrs（向量化实现）"""
    counts = torch.zeros(numRows + 1, dtype=torch.int32, device=rowIdx.device)
    counts.scatter_add_(0, (rowIdx.long() + 1), torch.ones_like(rowIdx, dtype=torch.int32))
    rowPtrs = counts.cumsum(0).to(torch.int32)
    return rowPtrs


def coo_to_ell(rowIdx, colIdx, values, numRows):
    """将 COO 转换为 ELL 格式（列优先存储，向量化实现）"""
    device = rowIdx.device
    counts = torch.zeros(numRows, dtype=torch.int64, device=device)
    counts.scatter_add_(0, rowIdx.long(), torch.ones(rowIdx.size(0), dtype=torch.int64, device=device))
    maxNnzPerRow = counts.max().item()

    ell_colIdx = torch.full((numRows * maxNnzPerRow,), -1, dtype=torch.int32, device=device)
    ell_values = torch.zeros(numRows * maxNnzPerRow, dtype=torch.float32, device=device)

    # 计算每个非零元素在其行内的偏移
    # 先按行排序（COO 通常已排序，但保险起见）
    row_long = rowIdx.long()
    # 用 cumcount 技巧：对每行内的元素编号
    ones = torch.ones(rowIdx.size(0), dtype=torch.int64, device=device)
    cumcounts = torch.zeros(rowIdx.size(0), dtype=torch.int64, device=device)
    # 对排序后的 rowIdx，相同行内的元素连续，用 cumsum 减去起始位置
    row_starts = torch.zeros(numRows, dtype=torch.int64, device=device)
    row_starts.scatter_add_(0, row_long, ones)
    row_starts = row_starts.cumsum(0) - row_starts  # exclusive prefix sum = start index of each row
    # 全局 cumsum 按行分组
    global_idx = torch.arange(rowIdx.size(0), dtype=torch.int64, device=device)
    in_row_offset = global_idx - row_starts[row_long]

    # 列优先索引: t * numRows + r
    ell_idx = in_row_offset * numRows + row_long
    ell_colIdx[ell_idx] = colIdx
    ell_values[ell_idx] = values

    return ell_colIdx, ell_values, maxNnzPerRow


def coo_to_jds(rowIdx, colIdx, values, numRows):
    """将 COO 转换为 JDS 格式（向量化实现）"""
    device = rowIdx.device
    row_long = rowIdx.long()

    # 统计每行非零元素数
    counts = torch.zeros(numRows, dtype=torch.int64, device=device)
    counts.scatter_add_(0, row_long, torch.ones(rowIdx.size(0), dtype=torch.int64, device=device))

    # 按非零元素数降序排列行
    sorted_counts, rowPerm = counts.sort(descending=True)
    maxLen = sorted_counts[0].item()

    # 建立逆映射: inv_perm[原始行号] = 排序后位置
    inv_perm = torch.empty(numRows, dtype=torch.int64, device=device)
    inv_perm[rowPerm] = torch.arange(numRows, dtype=torch.int64, device=device)

    # 计算每个非零元素在其行内的偏移
    ones = torch.ones(rowIdx.size(0), dtype=torch.int64, device=device)
    row_counts_cumsum = torch.zeros(numRows, dtype=torch.int64, device=device)
    row_counts_cumsum.scatter_add_(0, row_long, ones)
    row_starts = row_counts_cumsum.cumsum(0) - row_counts_cumsum
    global_idx = torch.arange(rowIdx.size(0), dtype=torch.int64, device=device)
    in_row_offset = global_idx - row_starts[row_long]  # 该元素是其行内第几个

    # 排序后的行位置
    sorted_pos = inv_perm[row_long]

    # 计算 iterPtr: 每个 tile t 包含 sorted_counts > t 的行数
    # iterPtr[0] = 0, iterPtr[t+1] = iterPtr[t] + (sorted_counts > t 的行数)
    # 即 iterPtr[t+1] - iterPtr[t] = 在 tile t 中有元素的行数
    iter_counts = torch.zeros(maxLen, dtype=torch.int64, device=device)
    for t in range(maxLen):
        iter_counts[t] = (sorted_counts > t).sum()
    iterPtr = torch.zeros(maxLen + 1, dtype=torch.int32, device=device)
    iterPtr[1:] = iter_counts.cumsum(0).to(torch.int32)

    # 计算每个非零元素在 JDS 数组中的位置
    # tile t 的元素从 iterPtr[t] 开始，按排序后行号顺序排列
    # 位置 = iterPtr[in_row_offset] + sorted_pos
    iterPtr_long = iterPtr.long()
    jds_pos = iterPtr_long[in_row_offset] + sorted_pos

    jds_colIdx = torch.empty(rowIdx.size(0), dtype=torch.int32, device=device)
    jds_values = torch.empty(rowIdx.size(0), dtype=torch.float32, device=device)
    jds_colIdx[jds_pos] = colIdx
    jds_values[jds_pos] = values

    return (
        jds_colIdx,
        jds_values,
        rowPerm.to(torch.int32),
        iterPtr,
        maxLen,
    )


def coo_to_ell_coo_hybrid(rowIdx, colIdx, values, numRows, maxNnzPerRow):
    """将 COO 转换为 Hybrid ELL-COO 格式（向量化实现）"""
    device = rowIdx.device
    row_long = rowIdx.long()

    # 计算每个元素在其行内的偏移
    ones = torch.ones(rowIdx.size(0), dtype=torch.int64, device=device)
    row_counts = torch.zeros(numRows, dtype=torch.int64, device=device)
    row_counts.scatter_add_(0, row_long, ones)
    row_starts = row_counts.cumsum(0) - row_counts
    global_idx = torch.arange(rowIdx.size(0), dtype=torch.int64, device=device)
    in_row_offset = global_idx - row_starts[row_long]

    # ELL 部分: in_row_offset < maxNnzPerRow
    ell_mask = in_row_offset < maxNnzPerRow
    ell_colIdx_out = torch.full((numRows * maxNnzPerRow,), -1, dtype=torch.int32, device=device)
    ell_values_out = torch.zeros(numRows * maxNnzPerRow, dtype=torch.float32, device=device)

    if ell_mask.any():
        ell_idx = in_row_offset[ell_mask] * numRows + row_long[ell_mask]
        ell_colIdx_out[ell_idx] = colIdx[ell_mask]
        ell_values_out[ell_idx] = values[ell_mask]

    # COO 溢出部分: in_row_offset >= maxNnzPerRow
    coo_mask = ~ell_mask
    if coo_mask.any():
        coo_rowIdx = rowIdx[coo_mask]
        coo_colIdx = colIdx[coo_mask]
        coo_values = values[coo_mask]
    else:
        coo_rowIdx = torch.zeros(0, dtype=torch.int32, device=device)
        coo_colIdx = torch.zeros(0, dtype=torch.int32, device=device)
        coo_values = torch.zeros(0, dtype=torch.float32, device=device)

    return ell_colIdx_out, ell_values_out, coo_rowIdx, coo_colIdx, coo_values


def reference_spmv(dense, x):
    """参考 SpMV: y = A @ x"""
    return dense @ x


# ============================================================
# 正确性测试
# ============================================================

def test_coo_correctness(name, fn, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    for numRows, numCols, density in test_cases:
        rowIdx, colIdx, values, dense = make_sparse_matrix(numRows, numCols, density)
        x = torch.randn(numCols, dtype=torch.float32, device="cuda")
        expected = reference_spmv(dense, x)
        result = fn(rowIdx, colIdx, values, x, numRows)
        ok = torch.allclose(result, expected, rtol=1e-4, atol=1e-5)
        if not ok:
            max_diff = (result - expected).abs().max().item()
            report(f"{numRows}x{numCols} density={density}", False, f"max_diff={max_diff:.6f}")
        else:
            t = benchmark_fn(fn, rowIdx, colIdx, values, x, numRows, warmup=10)
            report(f"{numRows}x{numCols} density={density:<4}  {t:.4f} ms", True)


def test_csr_correctness(name, fn, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    for numRows, numCols, density in test_cases:
        rowIdx, colIdx, values, dense = make_sparse_matrix(numRows, numCols, density)
        rowPtrs = coo_to_csr_ref(rowIdx, numRows)
        x = torch.randn(numCols, dtype=torch.float32, device="cuda")
        expected = reference_spmv(dense, x)
        result = fn(rowPtrs, colIdx, values, x, numRows)
        ok = torch.allclose(result, expected, rtol=1e-4, atol=1e-5)
        if not ok:
            max_diff = (result - expected).abs().max().item()
            report(f"{numRows}x{numCols} density={density}", False, f"max_diff={max_diff:.6f}")
        else:
            t = benchmark_fn(fn, rowPtrs, colIdx, values, x, numRows, warmup=10)
            report(f"{numRows}x{numCols} density={density:<4}  {t:.4f} ms", True)


def test_ell_correctness(name, fn, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    for numRows, numCols, density in test_cases:
        rowIdx, colIdx, values, dense = make_sparse_matrix(numRows, numCols, density)
        ell_colIdx, ell_values, maxNnzPerRow = coo_to_ell(rowIdx, colIdx, values, numRows)
        x = torch.randn(numCols, dtype=torch.float32, device="cuda")
        expected = reference_spmv(dense, x)
        result = fn(ell_colIdx, ell_values, x, numRows, maxNnzPerRow)
        ok = torch.allclose(result, expected, rtol=1e-4, atol=1e-5)
        if not ok:
            max_diff = (result - expected).abs().max().item()
            report(f"{numRows}x{numCols} density={density}", False, f"max_diff={max_diff:.6f}")
        else:
            t = benchmark_fn(fn, ell_colIdx, ell_values, x, numRows, maxNnzPerRow, warmup=10)
            report(f"{numRows}x{numCols} density={density:<4}  {t:.4f} ms", True)


def test_jds_correctness(name, fn, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    for numRows, numCols, density in test_cases:
        rowIdx, colIdx, values, dense = make_sparse_matrix(numRows, numCols, density)
        jds_col, jds_val, rowPerm, iterPtr, numTiles = coo_to_jds(rowIdx, colIdx, values, numRows)
        x = torch.randn(numCols, dtype=torch.float32, device="cuda")
        expected = reference_spmv(dense, x)
        result = fn(jds_col, jds_val, rowPerm, iterPtr, x, numRows, numTiles)
        ok = torch.allclose(result, expected, rtol=1e-4, atol=1e-5)
        if not ok:
            max_diff = (result - expected).abs().max().item()
            report(f"{numRows}x{numCols} density={density}", False, f"max_diff={max_diff:.6f}")
        else:
            t = benchmark_fn(fn, jds_col, jds_val, rowPerm, iterPtr, x, numRows, numTiles, warmup=10)
            report(f"{numRows}x{numCols} density={density:<4}  {t:.4f} ms", True)


def test_ell_coo_hybrid_correctness(name, fn, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    for numRows, numCols, density in test_cases:
        rowIdx, colIdx, values, dense = make_sparse_matrix(numRows, numCols, density)
        # 用每行非零元素数中位数作为 ELL 截断阈值
        counts = torch.zeros(numRows, dtype=torch.int64, device="cuda")
        counts.scatter_add_(0, rowIdx.long(), torch.ones(rowIdx.size(0), dtype=torch.int64, device="cuda"))
        maxNnzPerRow = max(1, int(counts.float().median().item()))
        ell_col, ell_val, coo_row, coo_col, coo_val = coo_to_ell_coo_hybrid(
            rowIdx, colIdx, values, numRows, maxNnzPerRow)
        x = torch.randn(numCols, dtype=torch.float32, device="cuda")
        expected = reference_spmv(dense, x)
        result = fn(ell_col, ell_val, coo_row, coo_col, coo_val, x, numRows, maxNnzPerRow)
        ok = torch.allclose(result, expected, rtol=1e-4, atol=1e-5)
        if not ok:
            max_diff = (result - expected).abs().max().item()
            report(f"{numRows}x{numCols} density={density}", False, f"max_diff={max_diff:.6f}")
        else:
            t = benchmark_fn(fn, ell_col, ell_val, coo_row, coo_col, coo_val, x, numRows, maxNnzPerRow, warmup=10)
            report(f"{numRows}x{numCols} density={density:<4}  {t:.4f} ms", True)


def test_coo_to_csr_correctness(name, fn, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    for numRows, numCols, density in test_cases:
        rowIdx, colIdx, values, dense = make_sparse_matrix(numRows, numCols, density)
        expected = coo_to_csr_ref(rowIdx, numRows)
        result = fn(rowIdx, colIdx, values, numRows)
        ok = torch.equal(result, expected)
        if not ok:
            max_diff = (result.int() - expected.int()).abs().max().item()
            report(f"{numRows}x{numCols} density={density}", False, f"max_diff={max_diff}")
        else:
            t = benchmark_fn(fn, rowIdx, colIdx, values, numRows, warmup=10)
            report(f"{numRows}x{numCols} density={density:<4}  {t:.4f} ms", True)


# ============================================================
# 性能对比
# ============================================================

def run_benchmark(exts):
    configs = [
        ((4096, 4096, 0.01), "4Kx4K density=0.01"),
        ((10000, 10000, 0.05), "10Kx10K density=0.05"),
        ((50000, 50000, 0.001), "50Kx50K density=0.001"),
    ]

    for (nr, nc, d), title in configs:
        print(colored(f"\n性能对比 {title}", "bold"))
        rowIdx, colIdx, values, dense = make_sparse_matrix(nr, nc, d)
        x = torch.randn(nc, dtype=torch.float32, device="cuda")

        rowPtrs = coo_to_csr_ref(rowIdx, nr)
        ell_col, ell_val, maxNnz = coo_to_ell(rowIdx, colIdx, values, nr)
        jds_col, jds_val, rowPerm, iterPtr, numTiles = coo_to_jds(rowIdx, colIdx, values, nr)

        format_args = {
            "COO": (rowIdx, colIdx, values, x, nr),
            "CSR": (rowPtrs, colIdx, values, x, nr),
            "ELL": (ell_col, ell_val, x, nr, maxNnz),
            "JDS": (jds_col, jds_val, rowPerm, iterPtr, x, nr, numTiles),
        }

        results = {}
        for label, fn, fmt_key in exts:
            if fmt_key not in format_args:
                continue
            try:
                args = format_args[fmt_key]
                t = benchmark_fn(fn, *args)
                results[label] = t
            except Exception:
                results[label] = None

        valid = {k: v for k, v in results.items() if v is not None}
        if not valid:
            continue
        baseline_key = next(iter(valid))
        baseline = valid[baseline_key]
        max_t = max(valid.values())
        max_label_len = max(len(k) for k in valid)

        for label, t in results.items():
            if t is None:
                print(f"  {label}: 跳过")
                continue
            speedup = f"  ({baseline/t:.2f}x)" if label != baseline_key else ""
            bar = "█" * int(t / max_t * 28)
            color = "green" if (t < baseline * 0.98) else (
                "red" if (t > baseline * 1.02) else "yellow")
            if label == baseline_key:
                color = "yellow"
            print(f"  {label:{max_label_len}}  {colored(f'{t:6.3f} ms', color)}  {bar}{speedup}")


# ============================================================
# 主函数
# ============================================================

CPP_COO = "torch::Tensor spmv_coo(torch::Tensor rowIdx, torch::Tensor colIdx, torch::Tensor values, torch::Tensor x, int numRows);"
CPP_CSR = "torch::Tensor spmv_csr(torch::Tensor rowPtrs, torch::Tensor colIdx, torch::Tensor values, torch::Tensor x, int numRows);"
CPP_ELL = "torch::Tensor spmv_ell(torch::Tensor colIdx, torch::Tensor values, torch::Tensor x, int numRows, int maxNnzPerRow);"
CPP_JDS = "torch::Tensor spmv_jds(torch::Tensor colIdx, torch::Tensor values, torch::Tensor rowPerm, torch::Tensor iterPtr, torch::Tensor x, int numRows, int numTiles);"
CPP_HYBRID = "torch::Tensor spmv_ell_coo_hybrid(torch::Tensor ell_colIdx, torch::Tensor ell_values, torch::Tensor coo_rowIdx, torch::Tensor coo_colIdx, torch::Tensor coo_values, torch::Tensor x, int numRows, int maxNnzPerRow);"
CPP_COO_TO_CSR = "torch::Tensor coo_to_csr(torch::Tensor rowIdx, torch::Tensor colIdx, torch::Tensor values, int numRows);"


def main():
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print(colored("警告: 未检测到 CUDA 设备，代码题将跳过", "yellow"))

    print(colored("=" * 55, "bold"))
    print(colored("  第十四章 CUDA 稀疏矩阵 SpMV - 正确性验证 & 性能对比", "bold"))
    print(colored("=" * 55, "bold"))

    test_cases = [
        (64, 64, 0.2),
        (512, 512, 0.1),
        (4096, 4096, 0.001),
        (10000, 10000, 0.0005),
    ]

    # COO to CSR 转换只测较小规模（prefix sum kernel 限制）
    coo_to_csr_test_cases = [
        (64, 64, 0.2),
        (512, 512, 0.1),
    ]

    bench_fns = []

    if cuda_available:
        # A: COO SpMV
        try:
            ext_coo = compile_kernel(
                "spmv_coo.cu", [CPP_COO], ["spmv_coo"], "ch14_coo")
            test_coo_correctness(
                "COO 格式 SpMV (spmv_coo.cu)",
                ext_coo.spmv_coo, test_cases)
            bench_fns.append(("A COO SpMV", ext_coo.spmv_coo, "COO"))
        except Exception as e:
            print(colored(f"\n[A] 编译失败: {e}", "red"))

        # B: CSR SpMV
        try:
            ext_csr = compile_kernel(
                "spmv_csr.cu", [CPP_CSR], ["spmv_csr"], "ch14_csr")
            test_csr_correctness(
                "CSR 格式 SpMV (spmv_csr.cu)",
                ext_csr.spmv_csr, test_cases)
            bench_fns.append(("B CSR SpMV", ext_csr.spmv_csr, "CSR"))
        except Exception as e:
            print(colored(f"\n[B] 编译失败: {e}", "red"))

        # C: ELL SpMV
        try:
            ext_ell = compile_kernel(
                "spmv_ell.cu", [CPP_ELL], ["spmv_ell"], "ch14_ell")
            test_ell_correctness(
                "ELL 格式 SpMV (spmv_ell.cu)",
                ext_ell.spmv_ell, test_cases)
            bench_fns.append(("C ELL SpMV", ext_ell.spmv_ell, "ELL"))
        except Exception as e:
            print(colored(f"\n[C] 编译失败: {e}", "red"))

        # D: JDS SpMV
        try:
            ext_jds = compile_kernel(
                "spmv_jds.cu", [CPP_JDS], ["spmv_jds"], "ch14_jds")
            test_jds_correctness(
                "JDS 格式 SpMV (spmv_jds.cu)",
                ext_jds.spmv_jds, test_cases)
            bench_fns.append(("D JDS SpMV", ext_jds.spmv_jds, "JDS"))
        except Exception as e:
            print(colored(f"\n[D] 编译失败: {e}", "red"))

        # E: Hybrid ELL-COO SpMV
        try:
            ext_hybrid = compile_kernel(
                "spmv_ell_coo_hybrid.cu", [CPP_HYBRID], ["spmv_ell_coo_hybrid"], "ch14_hybrid")
            test_ell_coo_hybrid_correctness(
                "Hybrid ELL-COO 格式 SpMV (spmv_ell_coo_hybrid.cu)",
                ext_hybrid.spmv_ell_coo_hybrid, test_cases)
        except Exception as e:
            print(colored(f"\n[E] 编译失败: {e}", "red"))

        # F: COO to CSR 转换
        try:
            ext_coo_to_csr = compile_kernel(
                "coo_to_csr.cu", [CPP_COO_TO_CSR], ["coo_to_csr"], "ch14_coo_to_csr")
            test_coo_to_csr_correctness(
                "COO → CSR 格式转换 (coo_to_csr.cu)",
                ext_coo_to_csr.coo_to_csr, coo_to_csr_test_cases)
        except Exception as e:
            print(colored(f"\n[F] 编译失败: {e}", "red"))

        if bench_fns:
            run_benchmark(bench_fns)

    print(colored("\n" + "=" * 55, "bold"))
    total = SCORE["total"]
    passed = SCORE["passed"]
    pct = (passed / total * 100) if total > 0 else 0
    color = "green" if pct >= 80 else ("yellow" if pct >= 50 else "red")
    print(colored(f"  总分: {passed}/{total} ({pct:.0f}%)", color))
    print(colored("=" * 55, "bold"))


if __name__ == "__main__":
    main()
