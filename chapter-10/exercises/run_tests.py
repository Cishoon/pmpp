#!/usr/bin/env python3
"""
第十章 一键判题脚本
用法: python run_tests.py
"""
import os
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

import torch
from torch.utils.cpp_extension import load_inline

torch.cuda.set_device(6)

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
# 正确性测试
# ============================================================

CPP_DECL = "torch::Tensor {name}(torch::Tensor input);"


def test_sum_correctness(name, fn, test_cases):
    """测试求和归约的正确性"""
    print(colored(f"\n[代码题] {name}", "bold"))
    ok_all = True
    for length in test_cases:
        data = torch.randn(length, device="cuda", dtype=torch.float32)
        result = fn(data)
        expected = data.sum()
        ok = torch.isclose(result, expected, rtol=1e-2, atol=1e-1).item()
        if not ok:
            ok_all = False
            diff = abs(result.item() - expected.item())
        report(f"length={length}", ok,
               f"got={result.item():.4f}, expected={expected.item():.4f}" if not ok else "")
    return ok_all


def test_max_correctness(name, fn, test_cases):
    """测试求最大值归约的正确性"""
    print(colored(f"\n[代码题] {name}", "bold"))
    ok_all = True
    for length in test_cases:
        data = torch.randn(length, device="cuda", dtype=torch.float32)
        result = fn(data)
        expected = data.max()
        ok = torch.isclose(result, expected, rtol=1e-5, atol=1e-5).item()
        if not ok:
            ok_all = False
        report(f"length={length}", ok,
               f"got={result.item():.4f}, expected={expected.item():.4f}" if not ok else "")
    return ok_all


# ============================================================
# 性能对比
# ============================================================

def run_benchmark(exts):
    configs = [
        (1_000_000, "1M 元素"),
        (10_000_000, "10M 元素"),
        (100_000_000, "100M 元素"),
    ]

    for length, title in configs:
        print(colored(f"\n性能对比 {title}", "bold"))
        data = torch.randn(length, device="cuda", dtype=torch.float32)

        results = {}
        for label, fn in exts:
            try:
                t = benchmark_fn(fn, data)
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
# 简答题测试
# ============================================================

def test_written_answers():
    print(colored("\n[简答题] 习题 1-6", "bold"))
    try:
        from answers import (
            ex1, ex2, ex3, ex4, ex5,
            ex6a, ex6b,
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

    check("1: 朴素归约第五次迭代分歧 warp 数", ex1, 16)
    check("2: 收敛型归约第五次迭代分歧 warp 数", ex2, 0)
    check("3: 反向收敛型归约结果线程", ex3, "B", lambda x: str(x).upper().strip())
    check("4: 求和改求最大值需替换的操作", ex4, "B", lambda x: str(x).upper().strip())
    check("5: 支持任意长度需添加的检查", ex5, "B", lambda x: str(x).upper().strip())

    # 习题 6a: 朴素内核 [6,2,7,4,5,8,3,1] 第一次迭代后
    # stride=1, threadIdx.x % 1 == 0 对所有线程成立
    # i=2*threadIdx.x: thread0: input[0]+=input[1] -> 8, thread1: input[2]+=input[3] -> 11,
    #                   thread2: input[4]+=input[5] -> 13, thread3: input[6]+=input[7] -> 4
    expected_6a = [8, 2, 11, 4, 13, 8, 4, 1]
    if ex6a is not None:
        ok = list(ex6a) == expected_6a
        report("6a: 朴素内核第一次迭代后数组", ok,
               f"你的答案={list(ex6a)}, 正确答案={expected_6a}" if not ok else "")
    else:
        report("6a: 朴素内核第一次迭代后数组", False, "未作答")

    # 习题 6b: 收敛型内核 [6,2,7,4,5,8,3,1] 第一次迭代后
    # stride=4 (blockDim.x=4), threadIdx.x < 4 对所有线程成立
    # thread0: input[0]+=input[4] -> 11, thread1: input[1]+=input[5] -> 10,
    # thread2: input[2]+=input[6] -> 10, thread3: input[3]+=input[7] -> 5
    expected_6b = [11, 10, 10, 5, 5, 8, 3, 1]
    if ex6b is not None:
        ok = list(ex6b) == expected_6b
        report("6b: 收敛型内核第一次迭代后数组", ok,
               f"你的答案={list(ex6b)}, 正确答案={expected_6b}" if not ok else "")
    else:
        report("6b: 收敛型内核第一次迭代后数组", False, "未作答")


# ============================================================
# 主函数
# ============================================================

def main():
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print(colored("警告: 未检测到 CUDA 设备，代码题将跳过", "yellow"))

    print(colored("=" * 55, "bold"))
    print(colored("  第十章 CUDA 并行归约 - 正确性验证 & 性能对比", "bold"))
    print(colored("=" * 55, "bold"))

    # 单 block 内核测试用例（必须是 2048 = 2*1024）
    single_block_cases = [2048]
    # 任意长度内核测试用例
    arbitrary_cases = [2048, 10000, 100000, 524288, 1000000]

    bench_fns = []

    if cuda_available:
        # A: 朴素归约
        try:
            ext_simple = compile_kernel(
                "reduce_simple.cu",
                [CPP_DECL.format(name="reduceSimple")],
                ["reduceSimple"], "ch10_simple")
            test_sum_correctness("朴素归约 (reduce_simple.cu)", ext_simple.reduceSimple, single_block_cases)
        except Exception as e:
            print(colored(f"\n[A] 编译失败: {e}", "red"))

        # B: 收敛型归约
        try:
            ext_conv = compile_kernel(
                "reduce_convergent.cu",
                [CPP_DECL.format(name="reduceConvergent")],
                ["reduceConvergent"], "ch10_convergent")
            test_sum_correctness("收敛型归约 (reduce_convergent.cu)", ext_conv.reduceConvergent, single_block_cases)
        except Exception as e:
            print(colored(f"\n[B] 编译失败: {e}", "red"))

        # C: 共享内存归约
        try:
            ext_shared = compile_kernel(
                "reduce_shared_mem.cu",
                [CPP_DECL.format(name="reduceSharedMem")],
                ["reduceSharedMem"], "ch10_shared")
            test_sum_correctness("共享内存归约 (reduce_shared_mem.cu)", ext_shared.reduceSharedMem, single_block_cases)
        except Exception as e:
            print(colored(f"\n[C] 编译失败: {e}", "red"))

        # D: 线程粗化求和归约
        try:
            ext_coarse_sum = compile_kernel(
                "reduce_coarsened_sum.cu",
                [CPP_DECL.format(name="reduceCoarsenedSum")],
                ["reduceCoarsenedSum"], "ch10_coarse_sum")
            test_sum_correctness("线程粗化求和归约 (reduce_coarsened_sum.cu)",
                                ext_coarse_sum.reduceCoarsenedSum, arbitrary_cases)
            bench_fns.append(("D 粗化求和", ext_coarse_sum.reduceCoarsenedSum))
        except Exception as e:
            print(colored(f"\n[D] 编译失败: {e}", "red"))

        # E: 线程粗化求最大值归约
        try:
            ext_coarse_max = compile_kernel(
                "reduce_coarsened_max.cu",
                [CPP_DECL.format(name="reduceCoarsenedMax")],
                ["reduceCoarsenedMax"], "ch10_coarse_max")
            test_max_correctness("线程粗化求最大值归约 (reduce_coarsened_max.cu)",
                                ext_coarse_max.reduceCoarsenedMax, arbitrary_cases)
        except Exception as e:
            print(colored(f"\n[E] 编译失败: {e}", "red"))

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
