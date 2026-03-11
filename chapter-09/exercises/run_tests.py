#!/usr/bin/env python3
"""
第九章 一键判题脚本
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


def histogram_ref(data, num_bins):
    """PyTorch 参考实现"""
    return torch.bincount(data, minlength=num_bins).to(torch.int32)


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

CPP_DECL = "torch::Tensor {name}(torch::Tensor data, int num_bins);"


def test_correctness(name, fn, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    ok_all = True
    for length, num_bins in test_cases:
        data = torch.randint(0, num_bins, (length,), device="cuda", dtype=torch.int32)
        result = fn(data, num_bins)
        expected = histogram_ref(data, num_bins)
        ok = torch.equal(result, expected)
        if not ok:
            ok_all = False
            diff = torch.max(torch.abs(result - expected)).item()
        report(f"length={length}, bins={num_bins}", ok,
               f"max diff={diff}" if not ok else "")
    return ok_all


# ============================================================
# 性能对比
# ============================================================

def run_benchmark(exts):
    configs = [
        (1_000_000, 128, "1M 元素, 128 bins"),
        (10_000_000, 128, "10M 元素, 128 bins"),
        (100_000_000, 128, "100M 元素, 128 bins"),
        (10_000_000, 1024, "10M 元素, 1024 bins"),
    ]

    for length, num_bins, title in configs:
        print(colored(f"\n性能对比 {title}", "bold"))
        data = torch.randint(0, num_bins, (length,), device="cuda", dtype=torch.int32)

        results = {}
        for label, fn in exts:
            try:
                t = benchmark_fn(fn, data, num_bins)
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
    print(colored("\n[简答题] 习题 1-6", "bold"))
    try:
        from answers import (
            ex1, ex2, ex3, ex4, ex5,
            ex6a, ex6b, ex6c,
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

    def norm_throughput(x):
        """归一化吞吐量答案：去空格、大写"""
        return str(x).strip().upper().replace(" ", "")

    check("1: DRAM 原子操作吞吐量", ex1, "10M", norm_throughput)
    check("2: L2+DRAM 混合原子操作吞吐量", ex2, "73M", norm_throughput)
    check("3: 浮点吞吐量（无私有化）", ex3, "50M", norm_throughput)
    check("4: 浮点吞吐量（共享内存私有化）", ex4, "4.55G", norm_throughput)
    check("5: atomicAdd 正确语句", ex5, "D", lambda x: str(x).upper().strip())
    check("6a: 无私有化原子操作数", ex6a, 524288)
    check("6b: 共享内存私有化原子操作数", ex6b, 65536)
    check("6c: 线程粗化原子操作数", ex6c, 16384)


# ============================================================
# 主函数
# ============================================================

def main():
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print(colored("警告: 未检测到 CUDA 设备，代码题将跳过", "yellow"))

    print(colored("=" * 55, "bold"))
    print(colored("  第九章 CUDA 直方图 - 正确性验证 & 性能对比", "bold"))
    print(colored("=" * 55, "bold"))

    test_cases = [
        (1024, 16),
        (10000, 128),
        (100000, 256),
        (524288, 128),
        (1000000, 1024),
    ]
    bench_fns = []

    if cuda_available:
        # A: 朴素
        try:
            ext_basic = compile_kernel(
                "histo_basic.cu",
                [CPP_DECL.format(name="histoBasic")],
                ["histoBasic"], "ch09_basic")
            test_correctness("朴素直方图 (histo_basic.cu)", ext_basic.histoBasic, test_cases)
            bench_fns.append(("A 朴素", ext_basic.histoBasic))
        except Exception as e:
            print(colored(f"\n[A] 编译失败: {e}", "red"))

        # B: 共享内存
        try:
            ext_shared = compile_kernel(
                "histo_shared_mem.cu",
                [CPP_DECL.format(name="histoSharedMem")],
                ["histoSharedMem"], "ch09_shared")
            test_correctness("共享内存直方图 (histo_shared_mem.cu)", ext_shared.histoSharedMem, test_cases)
            bench_fns.append(("B 共享内存", ext_shared.histoSharedMem))
        except Exception as e:
            print(colored(f"\n[B] 编译失败: {e}", "red"))

        # C: 线程粗化
        try:
            ext_coarse = compile_kernel(
                "histo_coarsening.cu",
                [CPP_DECL.format(name="histoCoarsening")],
                ["histoCoarsening"], "ch09_coarse")
            test_correctness("线程粗化直方图 (histo_coarsening.cu)", ext_coarse.histoCoarsening, test_cases)
            bench_fns.append(("C 线程粗化", ext_coarse.histoCoarsening))
        except Exception as e:
            print(colored(f"\n[C] 编译失败: {e}", "red"))

        # D: 线程粗化 + 合并访存
        try:
            ext_coal = compile_kernel(
                "histo_coarsening_coalesced.cu",
                [CPP_DECL.format(name="histoCoarsenedCoalesced")],
                ["histoCoarsenedCoalesced"], "ch09_coalesced")
            test_correctness("线程粗化+合并访存直方图 (histo_coarsening_coalesced.cu)",
                             ext_coal.histoCoarsenedCoalesced, test_cases)
            bench_fns.append(("D 粗化+合并", ext_coal.histoCoarsenedCoalesced))
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
