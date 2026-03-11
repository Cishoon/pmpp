#!/usr/bin/env python3
"""
第八章 一键判题脚本
用法: python run_tests.py
"""
import os
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

import torch
from torch.utils.cpp_extension import load_inline


torch.cuda.set_device(5)  

KERNELS_DIR = Path(__file__).parent / "kernels"
SCORE = {"total": 0, "passed": 0}
REPS = 200
BENCH_N = 128


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


def stencil_3d_ref(in_tensor, N, c0, c1, c2, c3, c4, c5, c6):
    """CPU 参考实现：七点 3D 模板"""
    inp = in_tensor.cpu().clone()
    out = torch.zeros_like(inp)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            for k in range(1, N - 1):
                idx = i * N * N + j * N + k
                out[idx] = (c0 * inp[idx]
                            + c1 * inp[idx - 1]
                            + c2 * inp[idx + 1]
                            + c3 * inp[idx - N]
                            + c4 * inp[idx + N]
                            + c5 * inp[idx - N * N]
                            + c6 * inp[idx + N * N])
    return out


def stencil_3d_ref_fast(in_tensor, N, c0, c1, c2, c3, c4, c5, c6):
    """向量化参考实现（用于大尺寸验证）"""
    inp = in_tensor.reshape(N, N, N)
    out = torch.zeros_like(inp)
    out[1:-1, 1:-1, 1:-1] = (
        c0 * inp[1:-1, 1:-1, 1:-1]
        + c1 * inp[1:-1, 1:-1, :-2]
        + c2 * inp[1:-1, 1:-1, 2:]
        + c3 * inp[1:-1, :-2, 1:-1]
        + c4 * inp[1:-1, 2:, 1:-1]
        + c5 * inp[:-2, 1:-1, 1:-1]
        + c6 * inp[2:, 1:-1, 1:-1]
    )
    return out.reshape(-1)


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
# 正确性测试
# ============================================================

COEFFS = (0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def test_correctness(name, fn, test_sizes):
    print(colored(f"\n[代码题] {name}", "bold"))
    c0, c1, c2, c3, c4, c5, c6 = COEFFS
    ok_all = True
    for N in test_sizes:
        in_tensor = torch.randn(N * N * N, device="cuda", dtype=torch.float32)
        result = fn(in_tensor, N, c0, c1, c2, c3, c4, c5, c6)
        expected = stencil_3d_ref_fast(in_tensor.cpu(), N, c0, c1, c2, c3, c4, c5, c6).cuda()
        ok = torch.allclose(result, expected, rtol=1e-3, atol=1e-3)
        if not ok:
            ok_all = False
        max_diff = torch.max(torch.abs(result - expected)).item() if not ok else 0
        report(f"N={N}", ok, f"max diff={max_diff:.6f}" if not ok else "")
    return ok_all


# ============================================================
# 性能对比
# ============================================================

def run_benchmark(exts):
    configs = [
        (64, "64³ (小网格)"),
        (128, "128³ (基准)"),
        (256, "256³ (大网格)"),
        (512, "512³ (大网格)"),
        (1024, "1024³ (大网格)"),
        (2048, "2048³ (大网格)"),
    ]
    c0, c1, c2, c3, c4, c5, c6 = COEFFS

    for N, title in configs:
        print(colored(f"\n性能对比 {title}", "bold"))
        in_tensor = torch.randn(N * N * N, device="cuda", dtype=torch.float32)

        results = {}
        for label, fn in exts:
            try:
                t = benchmark_fn(fn, in_tensor, N, c0, c1, c2, c3, c4, c5, c6)
                results[label] = t
            except Exception:
                results[label] = None

        valid = {k: v for k, v in results.items() if v is not None}
        if not valid:
            continue
        baseline = valid.get("A 朴素")
        max_t = max(valid.values())
        max_label_len = max(len(k) for k in valid)

        for label, t in results.items():
            if t is None:
                print(f"  {label}: 跳过")
                continue
            speedup = f"  ({baseline/t:.2f}x)" if baseline and label != "A 朴素" else ""
            bar = "█" * int(t / max_t * 28)
            color = "green" if (baseline and t < baseline * 0.98) else (
                "red" if (baseline and t > baseline * 1.02) else "yellow")
            print(f"  {label:{max_label_len}}  {colored(f'{t:6.3f} ms', color)}  {bar}{speedup}")


# ============================================================
# 简答题测试
# ============================================================

def test_written_answers():
    print(colored("\n[简答题] 习题 1-2", "bold"))
    try:
        from answers import (
            ex1a, ex1b, ex1c, ex1d,
            ex2a, ex2b, ex2c, ex2d, ex2e,
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

    def check_float(name, answer, expected, tol=0.05):
        if answer is None:
            report(name, False, "未作答")
            return
        ok = abs(float(answer) - expected) < tol
        report(name, ok, f"你的答案={answer}, 正确答案≈{expected}" if not ok else "")

    # 习题 1
    check("1a: 输出网格点数", ex1a, 1643032)
    check("1b: 基本内核线程块数", ex1b, 3375)
    check("1c: 共享内存内核线程块数", ex1c, 8000)
    check("1d: 线程粗化内核线程块数", ex1d, 64)

    # 习题 2
    check("2a: 输入分块大小（元素数）", ex2a, 18432)
    check("2b: 输出分块大小（元素数）", ex2b, 14400)
    check_float("2c: 浮点/内存访问比 (OP/B)", ex2c, 2.54, tol=0.05)
    check("2d: 无寄存器分块时共享内存字节数", ex2d, 12288)
    check("2e: 寄存器分块时共享内存字节数", ex2e, 4096)


# ============================================================
# 主函数
# ============================================================

CPP_DECL = "torch::Tensor {name}(torch::Tensor in, int N, float c0, float c1, float c2, float c3, float c4, float c5, float c6);"


def main():
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print(colored("警告: 未检测到 CUDA 设备，代码题将跳过", "yellow"))

    print(colored("=" * 55, "bold"))
    print(colored("  第八章 CUDA 3D 模板计算 - 正确性验证 & 性能对比", "bold"))
    print(colored("=" * 55, "bold"))

    test_sizes = [10, 16, 32, 48]
    bench_fns = []

    if cuda_available:
        # A: 朴素
        try:
            ext_basic = compile_kernel(
                "stencil_basic.cu",
                [CPP_DECL.format(name="stencil3dBasic")],
                ["stencil3dBasic"], "ch08_basic")
            test_correctness("朴素 3D 模板 (stencil_basic.cu)", ext_basic.stencil3dBasic, test_sizes)
            bench_fns.append(("A 朴素", ext_basic.stencil3dBasic))
        except Exception as e:
            print(colored(f"\n[A] 编译失败: {e}", "red"))

        # B: 共享内存
        try:
            ext_shared = compile_kernel(
                "stencil_shared_mem.cu",
                [CPP_DECL.format(name="stencil3dSharedMem")],
                ["stencil3dSharedMem"], "ch08_shared")
            test_correctness("共享内存 3D 模板 (stencil_shared_mem.cu)", ext_shared.stencil3dSharedMem, test_sizes)
            bench_fns.append(("B 共享内存", ext_shared.stencil3dSharedMem))
        except Exception as e:
            print(colored(f"\n[B] 编译失败: {e}", "red"))

        # C: 线程粗化
        try:
            ext_coarse = compile_kernel(
                "stencil_thread_coarsening.cu",
                [CPP_DECL.format(name="stencil3dCoarsening")],
                ["stencil3dCoarsening"], "ch08_coarse")
            test_correctness("线程粗化 3D 模板 (stencil_thread_coarsening.cu)", ext_coarse.stencil3dCoarsening, test_sizes)
            bench_fns.append(("C 线程粗化", ext_coarse.stencil3dCoarsening))
        except Exception as e:
            print(colored(f"\n[C] 编译失败: {e}", "red"))

        # D: 寄存器分块
        try:
            ext_reg = compile_kernel(
                "stencil_register_tiling.cu",
                [CPP_DECL.format(name="stencil3dRegisterTiling")],
                ["stencil3dRegisterTiling"], "ch08_reg")
            test_correctness("寄存器分块 3D 模板 (stencil_register_tiling.cu)", ext_reg.stencil3dRegisterTiling, test_sizes)
            bench_fns.append(("D 寄存器分块", ext_reg.stencil3dRegisterTiling))
        except Exception as e:
            print(colored(f"\n[D] 编译失败: {e}", "red"))

        if bench_fns:
            run_benchmark(bench_fns)

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
