#!/usr/bin/env python3
"""
第十三章 一键判题脚本
用法: python run_tests.py
"""
import os
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

import torch
from torch.utils.cpp_extension import load_inline

torch.cuda.set_device(3)  

KERNELS_DIR = Path(__file__).parent / "kernels"
SCORE = {"total": 0, "passed": 0}
REPS = 50


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


def make_random_ints(n, device="cuda"):
    """生成随机无符号整数（用 int32 存储，值域 [0, 2^31-1]）"""
    return torch.randint(0, 2**31, (n,), dtype=torch.int32, device=device)


def reference_sort(x):
    """用 PyTorch 排序作为参考"""
    return torch.sort(x)[0]


# ============================================================
# 正确性测试
# ============================================================

CPP_DECL = "torch::Tensor {name}(torch::Tensor input);"


def test_sort_correctness(name, fn, test_cases):
    """测试排序的正确性，并对每个用例计时"""
    print(colored(f"\n[代码题] {name}", "bold"))
    for n in test_cases:
        data = make_random_ints(n)
        result = fn(data.clone())
        expected = reference_sort(data)
        ok = torch.equal(result, expected)
        if not ok:
            # 找出第一个不匹配的位置
            diff_mask = result != expected
            first_diff = diff_mask.nonzero(as_tuple=True)[0][0].item() if diff_mask.any() else -1
            report(f"N={n}", False, f"first_diff_at={first_diff}")
        else:
            t = benchmark_fn(fn, data.clone(), warmup=10)
            report(f"N={n:<8}  {t:.4f} ms", True)


# ============================================================
# 性能对比
# ============================================================

def run_benchmark(exts):
    configs = [
        (10000, "10K 元素"),
        (100000, "100K 元素"),
        (1000000, "1M 元素"),
        (5000000, "5M 元素"),
    ]

    for n, title in configs:
        print(colored(f"\n性能对比 {title}", "bold"))
        data = make_random_ints(n)

        results = {}
        for label, fn in exts:
            try:
                t = benchmark_fn(fn, data.clone())
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
    print(colored("  第十三章 CUDA 排序 - 正确性验证 & 性能对比", "bold"))
    print(colored("=" * 55, "bold"))

    test_cases = [128, 1024, 10000, 65536, 500000]

    bench_fns = []

    if cuda_available:
        # A: 朴素并行基数排序
        try:
            ext_naive = compile_kernel(
                "radix_sort_naive.cu",
                [CPP_DECL.format(name="radix_sort_naive")],
                ["radix_sort_naive"], "ch13_radix_naive")
            test_sort_correctness(
                "朴素并行基数排序 (radix_sort_naive.cu)",
                ext_naive.radix_sort_naive, test_cases)
            bench_fns.append(("A 朴素基数排序", ext_naive.radix_sort_naive))
        except Exception as e:
            print(colored(f"\n[A] 编译失败: {e}", "red"))

        # B: 内存合并基数排序
        try:
            ext_coalesced = compile_kernel(
                "radix_sort_coalesced.cu",
                [CPP_DECL.format(name="radix_sort_coalesced")],
                ["radix_sort_coalesced"], "ch13_radix_coalesced")
            test_sort_correctness(
                "内存合并基数排序 (radix_sort_coalesced.cu)",
                ext_coalesced.radix_sort_coalesced, test_cases)
            bench_fns.append(("B 内存合并基数排序", ext_coalesced.radix_sort_coalesced))
        except Exception as e:
            print(colored(f"\n[B] 编译失败: {e}", "red"))

        # C: 多位基数排序
        try:
            ext_multibit = compile_kernel(
                "radix_sort_multibit.cu",
                [CPP_DECL.format(name="radix_sort_multibit")],
                ["radix_sort_multibit"], "ch13_radix_multibit")
            test_sort_correctness(
                "多位基数排序 (radix_sort_multibit.cu)",
                ext_multibit.radix_sort_multibit, test_cases)
            bench_fns.append(("C 多位基数排序", ext_multibit.radix_sort_multibit))
        except Exception as e:
            print(colored(f"\n[C] 编译失败: {e}", "red"))

        # D: 并行归并排序
        try:
            ext_merge = compile_kernel(
                "merge_sort.cu",
                [CPP_DECL.format(name="merge_sort")],
                ["merge_sort"], "ch13_merge_sort")
            test_sort_correctness(
                "并行归并排序 (merge_sort.cu)",
                ext_merge.merge_sort, test_cases)
            bench_fns.append(("D 并行归并排序", ext_merge.merge_sort))
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
