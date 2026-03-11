#!/usr/bin/env python3
"""
第十四章 一键判题脚本
用法: python run_tests.py
"""
import os
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

import torch
import torch.sparse
from torch.utils.cpp_extension import load_inline

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
    """生成随机稀疏矩阵，返回 COO 格式的各数组和稠密矩阵"""
    nnz_per_row = max(1, int(numCols * density))
    rows = []
    cols = []
    vals = []
    for r in range(numRows):
        # 每行随机选 nnz_per_row 个列
        c = torch.randperm(numCols)[:nnz_per_row].sort()[0]
        for ci in c:
            rows.append(r)
            cols.append(ci.item())
            vals.append(torch.randn(1).item())

    rowIdx = torch.tensor(rows, dtype=torch.int32, device=device)
    colIdx = torch.tensor(cols, dtype=torch.int32, device=device)
    values = torch.tensor(vals, dtype=torch.float32, device=device)

    # 构建稠密矩阵用于参考计算
    dense = torch.zeros(numRows, numCols, dtype=torch.float32, device=device)
    for i in range(len(rows)):
        dense[rows[i], cols[i]] = vals[i]

    return rowIdx, colIdx, values, dense


def coo_to_csr(rowIdx, numRows):
    """将 COO 的 rowIdx 转换为 CSR 的 rowPtrs"""
    rowPtrs = torch.zeros(numRows + 1, dtype=torch.int32, device=rowIdx.device)
    for r in rowIdx:
        rowPtrs[r + 1] += 1
    for i in range(1, numRows + 1):
        rowPtrs[i] += rowPtrs[i - 1]
    return rowPtrs


def coo_to_ell(rowIdx, colIdx, values, numRows):
    """将 COO 转换为 ELL 格式（列优先存储）"""
    device = rowIdx.device
    # 统计每行非零元素数
    counts = torch.zeros(numRows, dtype=torch.int32)
    for r in rowIdx.cpu():
        counts[r.item()] += 1
    maxNnzPerRow = counts.max().item()

    ell_colIdx = torch.full((numRows * maxNnzPerRow,), -1, dtype=torch.int32, device=device)
    ell_values = torch.zeros(numRows * maxNnzPerRow, dtype=torch.float32, device=device)

    # 记录每行已填入的元素数
    filled = [0] * numRows
    for i in range(rowIdx.size(0)):
        r = rowIdx[i].item()
        t = filled[r]
        idx = t * numRows + r  # 列优先
        ell_colIdx[idx] = colIdx[i]
        ell_values[idx] = values[i]
        filled[r] = t + 1

    return ell_colIdx, ell_values, maxNnzPerRow


def coo_to_jds(rowIdx, colIdx, values, numRows):
    """将 COO 转换为 JDS 格式"""
    device = rowIdx.device
    # 按行分组
    row_data = [[] for _ in range(numRows)]
    for i in range(rowIdx.size(0)):
        r = rowIdx[i].item()
        row_data[r].append((colIdx[i].item(), values[i].item()))

    # 按非零元素数量降序排列
    row_lengths = [(len(row_data[r]), r) for r in range(numRows)]
    row_lengths.sort(key=lambda x: -x[0])

    rowPerm = [r for _, r in row_lengths]
    sorted_rows = [row_data[r] for _, r in row_lengths]

    maxLen = max(len(r) for r in sorted_rows) if sorted_rows else 0

    jds_colIdx = []
    jds_values = []
    iterPtr = [0]

    for t in range(maxLen):
        count = 0
        for r in range(numRows):
            if t < len(sorted_rows[r]):
                jds_colIdx.append(sorted_rows[r][t][0])
                jds_values.append(sorted_rows[r][t][1])
                count += 1
        iterPtr.append(iterPtr[-1] + count)

    return (
        torch.tensor(jds_colIdx, dtype=torch.int32, device=device),
        torch.tensor(jds_values, dtype=torch.float32, device=device),
        torch.tensor(rowPerm, dtype=torch.int32, device=device),
        torch.tensor(iterPtr, dtype=torch.int32, device=device),
        maxLen,
    )


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
        rowPtrs = coo_to_csr(rowIdx, numRows)
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

        # 预计算各格式
        rowPtrs = coo_to_csr(rowIdx, nr)
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
        (4096, 4096, 0.01),
        (10000, 10000, 0.05),
        (50000, 50000, 0.001),
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
