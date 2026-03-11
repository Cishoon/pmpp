#!/usr/bin/env python3
"""
第十二章 一键判题脚本
用法: python run_tests.py
"""
import os
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

import torch
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


def make_sorted_pair(m, n, device="cuda"):
    """生成两个已排序的浮点张量"""
    A = torch.sort(torch.randn(m, device=device))[0]
    B = torch.sort(torch.randn(n, device=device))[0]
    return A, B


def reference_merge(A, B):
    """用 PyTorch 实现参考归并"""
    C = torch.cat([A, B])
    return torch.sort(C)[0]


# ============================================================
# 正确性测试
# ============================================================

CPP_DECL = "torch::Tensor {name}(torch::Tensor A, torch::Tensor B);"


def test_merge_correctness(name, fn, test_cases):
    """测试归并的正确性，并对每个用例计时"""
    print(colored(f"\n[代码题] {name}", "bold"))
    for m, n in test_cases:
        A, B = make_sorted_pair(m, n)
        result = fn(A, B)
        expected = reference_merge(A, B)
        ok = torch.allclose(result, expected, rtol=1e-5, atol=1e-8)
        if not ok:
            max_diff = (result - expected).abs().max().item()
            report(f"m={m}, n={n}", False, f"max_diff={max_diff:.6f}")
        else:
            t = benchmark_fn(fn, A, B, warmup=10)
            report(f"m={m:<8} n={n:<8}  {t:.4f} ms", True)


# ============================================================
# 性能对比
# ============================================================

def run_benchmark(exts):
    configs = [
        ((1024, 1024), "2K 元素 (1K+1K)"),
        ((65536, 65536), "128K 元素 (64K+64K)"),
        ((500000, 500000), "1M 元素 (500K+500K)"),
        ((1000000, 638400), "1.6M 元素 (1M+638K)"),
        ((10000000, 6384000), "16M 元素 (10M+6380K)"),
        ((100000000, 63840000), "160M 元素 (100M+63800K)"),
    ]

    for (m, n), title in configs:
        print(colored(f"\n性能对比 {title}", "bold"))
        A, B = make_sorted_pair(m, n)

        results = {}
        for label, fn in exts:
            try:
                t = benchmark_fn(fn, A, B)
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

def main():
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print(colored("警告: 未检测到 CUDA 设备，代码题将跳过", "yellow"))

    print(colored("=" * 55, "bold"))
    print(colored("  第十二章 CUDA 并行归并 - 正确性验证 & 性能对比", "bold"))
    print(colored("=" * 55, "bold"))

    # 测试用例：(m, n) 对
    test_cases = [
        (128, 128),
        (1024, 512),
        (10000, 8000),
        (65536, 65536),
        (500000, 500000),
    ]

    bench_fns = []

    if cuda_available:
        # A: 基础并行归并
        try:
            ext_basic = compile_kernel(
                "merge_basic.cu",
                [CPP_DECL.format(name="merge_basic")],
                ["merge_basic"], "ch12_basic")
            test_merge_correctness(
                "基础并行归并 (merge_basic.cu)",
                ext_basic.merge_basic, test_cases)
            bench_fns.append(("A 基础并行归并", ext_basic.merge_basic))
        except Exception as e:
            print(colored(f"\n[A] 编译失败: {e}", "red"))

        # B: 分块并行归并
        try:
            ext_tiled = compile_kernel(
                "merge_tiled.cu",
                [CPP_DECL.format(name="merge_tiled")],
                ["merge_tiled"], "ch12_tiled")
            test_merge_correctness(
                "分块并行归并 (merge_tiled.cu)",
                ext_tiled.merge_tiled, test_cases)
            bench_fns.append(("B 分块并行归并", ext_tiled.merge_tiled))
        except Exception as e:
            print(colored(f"\n[B] 编译失败: {e}", "red"))

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
