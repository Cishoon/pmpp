#!/usr/bin/env python3
"""
第五章 一键判题脚本
用法: python run_tests.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

import torch
from torch.utils.cpp_extension import load_inline

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
    status = colored("✓ PASS", "green") if passed else colored("✗ FAIL", "red")
    if passed:
        SCORE["passed"] += 1
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

def test_tiled_matmul():
    print(colored("\n[代码题] 分块矩阵乘法内核 (tiled_matmul.cu)", "bold"))
    try:
        ext = compile_kernel(
            "tiled_matmul.cu",
            ["torch::Tensor tiledMatMul(torch::Tensor M, torch::Tensor N);"],
            ["tiledMatMul"],
            "ch05_tiled_matmul",
        )
        test_cases = [
            (16, 16, 16),
            (32, 64, 32),
            (128, 256, 128),
            (100, 200, 150),   # 非 TILE_WIDTH 整数倍
            (1, 64, 1),
        ]
        for m, n, o in test_cases:
            M = torch.randn(m, n, device="cuda", dtype=torch.float32)
            N = torch.randn(n, o, device="cuda", dtype=torch.float32)
            result = ext.tiledMatMul(M, N)
            expected = torch.matmul(M, N)
            ok = torch.allclose(result, expected, rtol=1e-3, atol=1e-3)
            max_diff = torch.max(torch.abs(result - expected)).item()
            report(f"M={m}x{n}, N={n}x{o}", ok,
                   f"max diff={max_diff:.6f}" if not ok else "")
    except Exception as e:
        report("编译/运行失败", False, str(e)[:200])


def test_transpose_benchmark():
    print(colored("\n[代码题] 朴素 vs 分块转置速度对比 (tiled_transpose_benchmark.cu)", "bold"))
    try:
        ext = compile_kernel(
            "tiled_transpose_benchmark.cu",
            [
                "torch::Tensor naiveTranspose(torch::Tensor A);",
                "torch::Tensor tiledTransposeBench(torch::Tensor A);",
            ],
            ["naiveTranspose", "tiledTransposeBench"],
            "ch05_transpose_bench",
        )

        # 正确性验证
        print(colored("  正确性验证:", "yellow"))
        for m, n in [(64, 64), (512, 1024), (100, 200)]:
            A = torch.randn(m, n, device="cuda", dtype=torch.float32)
            expected = A.T.contiguous()
            ok_naive = torch.allclose(ext.naiveTranspose(A), expected, atol=1e-5)
            ok_tiled = torch.allclose(ext.tiledTransposeBench(A), expected, atol=1e-5)
            report(f"朴素  {m}x{n}", ok_naive)
            report(f"分块  {m}x{n}", ok_tiled)

        # 速度对比（用大矩阵）
        print(colored("  速度对比 (4096x4096):", "yellow"))
        A = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
        REPS = 200

        # 预热
        for _ in range(20):
            ext.naiveTranspose(A)
            ext.tiledTransposeBench(A)
        torch.cuda.synchronize()

        # 计时朴素
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(REPS):
            ext.naiveTranspose(A)
        end.record()
        torch.cuda.synchronize()
        t_naive = start.elapsed_time(end) / REPS

        # 计时分块
        start.record()
        for _ in range(REPS):
            ext.tiledTransposeBench(A)
        end.record()
        torch.cuda.synchronize()
        t_tiled = start.elapsed_time(end) / REPS

        speedup = t_naive / t_tiled if t_tiled > 0 else float("inf")
        print(f"    朴素转置:  {t_naive:.3f} ms")
        print(f"    分块转置:  {t_tiled:.3f} ms")
        color = "green" if speedup >= 1.0 else "red"
        print(colored(f"    加速比:    {speedup:.2f}x", color))

        # 分块比朴素快才算通过
        report("分块比朴素更快", speedup > 1.0,
               f"加速比={speedup:.2f}x，分块应更快" if speedup <= 1.0 else "")

    except Exception as e:
        report("编译/运行失败", False, str(e))


def test_tiled_transpose():
    print(colored("\n[代码题] 分块矩阵转置 (tiled_transpose.cu)", "bold"))
    try:
        ext = compile_kernel(
            "tiled_transpose.cu",
            ["torch::Tensor tiledTranspose(torch::Tensor A);"],
            ["tiledTranspose"],
            "ch05_tiled_transpose",
        )
        test_cases = [
            (16, 16),
            (32, 64),
            (100, 200),  # 非整数倍
            (1, 16),
            (64, 1),
        ]
        for m, n in test_cases:
            A = torch.randn(m, n, device="cuda", dtype=torch.float32)
            result = ext.tiledTranspose(A)
            expected = A.T.contiguous()
            ok = torch.allclose(result, expected, atol=1e-5)
            report(f"A={m}x{n} -> B={n}x{m}", ok)
    except Exception as e:
        report("编译/运行失败", False, str(e))


def test_block_transpose():
    print(colored("\n[习题 10] 分块转置内核 (block_transpose.cu)", "bold"))
    try:
        ext = compile_kernel(
            "block_transpose.cu",
            ["torch::Tensor blockTranspose(torch::Tensor A);"],
            ["blockTranspose"],
            "ch05_block_transpose",
        )
        # 对每个 BLOCK_WIDTH x BLOCK_WIDTH 分块，转置后应等于原分块的转置
        import torch.nn.functional as F

        def reference_block_transpose(A, bw=16):
            """CPU 参考：对每个 bw×bw 分块做转置"""
            H, W = A.shape
            out = A.clone()
            for bi in range(H // bw):
                for bj in range(W // bw):
                    block = A[bi*bw:(bi+1)*bw, bj*bw:(bj+1)*bw].clone()
                    out[bi*bw:(bi+1)*bw, bj*bw:(bj+1)*bw] = block.T
            return out

        for H, W in [(16, 16), (32, 32), (64, 64), (32, 64)]:
            A = torch.randn(H, W, device="cuda", dtype=torch.float32)
            result = ext.blockTranspose(A)
            expected = reference_block_transpose(A.cpu()).cuda()
            ok = torch.allclose(result, expected, atol=1e-5)
            report(f"{H}x{W}", ok)
    except Exception as e:
        report("编译/运行失败", False, str(e)[:200])


# ============================================================
# 简答题 / 计算题测试
# ============================================================

def test_written_answers():
    print(colored("\n[简答题] 习题 1、3、5-12", "bold"))
    try:
        from answers import (
            ex1,
            ex3a, ex3b,
            ex5, ex6, ex7,
            ex8a, ex8b,
            ex9a, ex9b,
            ex10b,
            ex11a, ex11b, ex11c, ex11d, ex11e, ex11f,
            ex12a_full_occupancy, ex12a_bottleneck, ex12a_occupancy_pct,
            ex12b_full_occupancy,
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
        ok = a == e
        report(name, ok, f"你的答案={answer!r}, 正确答案={expected!r}" if not ok else "")

    check("1: 矩阵加法能否用共享内存", ex1, "no",
          transform=lambda x: str(x).lower().strip())

    check("3a: 忘记第一个 __syncthreads()", ex3a, "A",
          transform=lambda x: str(x).upper().strip())
    check("3b: 忘记第二个 __syncthreads()", ex3b, "A",
          transform=lambda x: str(x).upper().strip())

    check("5: 32×32分块带宽降低倍数", ex5, 32)
    check("6: 局部变量副本数", ex6, 512000)
    check("7: 共享内存变量副本数", ex7, 1000)

    check("8a: 无分块时每元素加载次数", ex8a, "N",
          transform=lambda x: str(x).strip().upper())
    check("8b: T×T分块时每元素加载次数", ex8b, "N/T",
          transform=lambda x: str(x).strip().upper().replace(" ", ""))

    check("9a: 200GFLOPS/100GB/s 类型", ex9a, "memory",
          transform=lambda x: str(x).lower().strip())
    check("9b: 300GFLOPS/250GB/s 类型", ex9b, "compute",
          transform=lambda x: str(x).lower().strip())

    check("10b: BlockTranspose 缺少什么", ex10b, "A",
          transform=lambda x: str(x).upper().strip())

    check("11a: 变量 i 的版本数", ex11a, 1024)
    check("11b: 数组 x[] 的版本数", ex11b, 1024)
    check("11c: 变量 y_s 的版本数", ex11c, 8)
    check("11d: 数组 b_s[] 的版本数", ex11d, 8)
    check("11e: 每块共享内存字节数", ex11e, 516)

    # 11f: 允许小误差
    if ex11f is None:
        report("11f: 浮点/内存访问比 (OP/B)", False, "未作答")
    else:
        ok = abs(float(ex11f) - 0.5) < 0.02
        report("11f: 浮点/内存访问比 (OP/B)", ok,
               f"你的答案={ex11f}, 正确答案≈0.5" if not ok else "")

    check("12a: 能否完全占用", ex12a_full_occupancy, "no",
          transform=lambda x: str(x).lower().strip())
    check("12a: 瓶颈因素", ex12a_bottleneck, "shared_memory",
          transform=lambda x: str(x).lower().strip() if x else "none")
    check("12a: 实际占用率(%)", ex12a_occupancy_pct, 75)
    check("12b: 能否完全占用", ex12b_full_occupancy, "yes",
          transform=lambda x: str(x).lower().strip())


# ============================================================
# 主函数
# ============================================================

def main():
    if not torch.cuda.is_available():
        print(colored("警告: 未检测到 CUDA 设备，代码题将跳过", "yellow"))
        cuda_available = False
    else:
        cuda_available = True

    print(colored("=" * 55, "bold"))
    print(colored("  第五章 CUDA 编程练习 - 自动判题", "bold"))
    print(colored("=" * 55, "bold"))

    if cuda_available:
        test_tiled_matmul()
        test_tiled_transpose()
        test_transpose_benchmark()
        test_block_transpose()

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
