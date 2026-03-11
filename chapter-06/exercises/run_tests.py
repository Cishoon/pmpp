#!/usr/bin/env python3
"""
第六章 一键判题脚本
用法: python run_tests.py
"""
import os
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

import torch
from torch.utils.cpp_extension import load_inline

torch.cuda.set_device(4)  

KERNELS_DIR = Path(__file__).parent / "kernels"
SCORE = {"total": 0, "passed": 0}


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


# ============================================================
# 代码题测试
# ============================================================

def test_col_major_matmul():
    print(colored("\n[习题 1] 列主序 N 的分块矩阵乘法 (col_major_matmul.cu)", "bold"))
    try:
        ext = compile_kernel(
            "col_major_matmul.cu",
            ["torch::Tensor colMajorMatMul(torch::Tensor M, torch::Tensor N_T);"],
            ["colMajorMatMul"],
            "ch06_col_major",
        )
        test_cases = [(16, 16, 16), (32, 64, 32), (100, 200, 150), (1, 16, 1)]
        for m, n, o in test_cases:
            M = torch.randn(m, n, device="cuda", dtype=torch.float32)
            N = torch.randn(n, o, device="cuda", dtype=torch.float32)
            N_T = N.T.contiguous()  # 转置后传入，形状 (o, n)
            result = ext.colMajorMatMul(M, N_T)
            expected = torch.matmul(M, N)
            ok = torch.allclose(result, expected, rtol=1e-3, atol=1e-3)
            report(f"M={m}x{n}, N={n}x{o}", ok,
                   f"max diff={torch.max(torch.abs(result-expected)).item():.6f}" if not ok else "")
    except Exception as e:
        report("编译/运行失败", False, str(e))


def test_coarsened_matmul():
    print(colored("\n[代码题] 线程粗化分块矩阵乘法 (coarsened_matmul.cu)", "bold"))
    try:
        ext = compile_kernel(
            "coarsened_matmul.cu",
            ["torch::Tensor coarsenedMatMul(torch::Tensor M, torch::Tensor N);"],
            ["coarsenedMatMul"],
            "ch06_coarsened",
        )
        test_cases = [
            (16, 16, 16),
            (64, 64, 64),
            (128, 256, 128),
            (100, 200, 150),   # 非整数倍
        ]
        for m, n, o in test_cases:
            M = torch.randn(m, n, device="cuda", dtype=torch.float32)
            N = torch.randn(n, o, device="cuda", dtype=torch.float32)
            result = ext.coarsenedMatMul(M, N)
            expected = torch.matmul(M, N)
            ok = torch.allclose(result, expected, rtol=1e-3, atol=1e-3)
            report(f"M={m}x{n}, N={n}x{o}", ok,
                   f"max diff={torch.max(torch.abs(result-expected)).item():.6f}" if not ok else "")
    except Exception as e:
        report("编译/运行失败", False, str(e))


# ============================================================
# 简答题测试
# ============================================================

def test_written_answers():
    print(colored("\n[简答题] 习题 2-4", "bold"))
    try:
        from answers import (
            ex2,
            ex3a, ex3b, ex3c, ex3d, ex3e, ex3f, ex3g, ex3h, ex3i,
            ex4a, ex4b, ex4c,
        )
    except ImportError as e:
        report(f"无法导入 answers.py: {e}", False)
        return

    def check(name, answer, expected, transform=None):
        if answer is None:
            report(name, False, "未作答")
            return
        a = transform(answer) if transform else answer
        e = transform(expected) if transform else expected
        report(name, a == e,
               f"你的答案={answer!r}, 正确答案={expected!r}" if a != e else "")

    def norm(x):
        return str(x).upper().strip()

    check("2: BLOCK_SIZE 条件", ex2, "B", norm)

    check("3a: a[i]",                    ex3a, "coalesced",     lambda x: x.lower().strip())
    check("3b: a_s (shared)",            ex3b, "not_applicable", lambda x: x.lower().strip())
    check("3c: b[j*N+i]",               ex3c, "coalesced",     lambda x: x.lower().strip())
    check("3d: c[i*4+j]",               ex3d, "uncoalesced",   lambda x: x.lower().strip())
    check("3e: bc_s (shared)",           ex3e, "not_applicable", lambda x: x.lower().strip())
    check("3f: a_s line10 (shared)",     ex3f, "not_applicable", lambda x: x.lower().strip())
    check("3g: d[i+8]",                  ex3g, "coalesced",     lambda x: x.lower().strip())
    check("3h: bc_s line11 (shared)",    ex3h, "not_applicable", lambda x: x.lower().strip())
    check("3i: e[i*8]",                  ex3i, "uncoalesced",   lambda x: x.lower().strip())

    def check_float(name, answer, expected, tol=0.02):
        if answer is None:
            report(name, False, "未作答")
            return
        ok = abs(float(answer) - expected) < tol
        report(name, ok, f"你的答案={answer}, 正确答案≈{expected}" if not ok else "")

    check_float("4a: 朴素内核 OP/B",          ex4a, 0.25)
    check_float("4b: 分块内核 OP/B",          ex4b, 8.0,  tol=0.1)
    check_float("4c: 线程粗化内核 OP/B",      ex4c, 12.8, tol=0.2)


# ============================================================
# 主函数
# ============================================================

def main():
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print(colored("警告: 未检测到 CUDA 设备，代码题将跳过", "yellow"))

    print(colored("=" * 55, "bold"))
    print(colored("  第六章 CUDA 编程练习 - 自动判题", "bold"))
    print(colored("=" * 55, "bold"))

    if cuda_available:
        test_col_major_matmul()
        test_coarsened_matmul()

    test_written_answers()

    print(colored("\n" + "=" * 55, "bold"))
    total = SCORE["total"]
    passed = SCORE["passed"]
    pct = (passed / total * 100) if total > 0 else 0
    color = "green" if pct >= 80 else ("yellow" if pct >= 50 else "red")
    print(colored(f"  总分: {passed}/{total} ({pct:.0f}%)", color))
    print(colored("=" * 55, "bold"))


if __name__ == "__main__":
    main()
