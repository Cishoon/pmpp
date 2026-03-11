#!/usr/bin/env python3
"""
第十一章 一键判题脚本
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


def test_scan_correctness(name, fn, test_cases):
    """测试前缀扫描的正确性，并对每个用例计时"""
    print(colored(f"\n[代码题] {name}", "bold"))
    for length in test_cases:
        data = torch.randn(length, device="cuda", dtype=torch.float32)
        result = fn(data)
        expected = torch.cumsum(data, dim=0)
        ok = torch.allclose(result, expected, rtol=1e-2, atol=1e-1)
        if not ok:
            max_diff = (result - expected).abs().max().item()
            report(f"length={length}", False, f"max_diff={max_diff:.6f}")
        else:
            t = benchmark_fn(fn, data, warmup=10)
            report(f"length={length:<8}  {t:.4f} ms", True)


# ============================================================
# 性能对比
# ============================================================

def run_benchmark(exts):
    configs = [
        (1024, "1K 元素"),
        (65536, "64K 元素"),
        (1_000_000, "1M 元素"),
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
# 主函数
# ============================================================

def main():
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print(colored("警告: 未检测到 CUDA 设备，代码题将跳过", "yellow"))

    print(colored("=" * 55, "bold"))
    print(colored("  第十一章 CUDA 前缀扫描 - 正确性验证 & 性能对比", "bold"))
    print(colored("=" * 55, "bold"))

    # 单 block 内核测试用例（<= 1024）
    single_block_cases = [8, 64, 256, 512, 1024]
    # Brent-Kung 测试用例（<= 2048，需为 2 的幂）
    brent_kung_cases = [8, 64, 256, 512, 1024, 2048]
    # 分层扫描测试用例（任意长度）
    hierarchical_cases = [1024, 2048, 10000, 65536, 100000, 524288]

    bench_fns = []

    if cuda_available:
        # A: Kogge-Stone 基础扫描
        try:
            ext_ks = compile_kernel(
                "kogge_stone_scan.cu",
                [CPP_DECL.format(name="kogge_stone_scan")],
                ["kogge_stone_scan"], "ch11_ks")
            test_scan_correctness(
                "Kogge-Stone 扫描 (kogge_stone_scan.cu)",
                ext_ks.kogge_stone_scan, single_block_cases)
            bench_fns.append(("A Kogge-Stone", ext_ks.kogge_stone_scan))
        except Exception as e:
            print(colored(f"\n[A] 编译失败: {e}", "red"))

        # B: Kogge-Stone 双缓冲扫描
        try:
            ext_ks_db = compile_kernel(
                "kogge_stone_double_buffer_scan.cu",
                [CPP_DECL.format(name="kogge_stone_double_buffer_scan")],
                ["kogge_stone_double_buffer_scan"], "ch11_ks_db")
            test_scan_correctness(
                "Kogge-Stone 双缓冲扫描 (kogge_stone_double_buffer_scan.cu)",
                ext_ks_db.kogge_stone_double_buffer_scan, single_block_cases)
            bench_fns.append(("B Kogge-Stone 双缓冲", ext_ks_db.kogge_stone_double_buffer_scan))
        except Exception as e:
            print(colored(f"\n[B] 编译失败: {e}", "red"))

        # C: Brent-Kung 扫描
        try:
            ext_bk = compile_kernel(
                "brent_kung_scan.cu",
                [CPP_DECL.format(name="brent_kung_scan")],
                ["brent_kung_scan"], "ch11_bk")
            test_scan_correctness(
                "Brent-Kung 扫描 (brent_kung_scan.cu)",
                ext_bk.brent_kung_scan, brent_kung_cases)
            bench_fns.append(("C Brent-Kung", ext_bk.brent_kung_scan))
        except Exception as e:
            print(colored(f"\n[C] 编译失败: {e}", "red"))

        # D: 三阶段扫描
        try:
            ext_tp = compile_kernel(
                "three_phase_scan.cu",
                [CPP_DECL.format(name="three_phase_scan")],
                ["three_phase_scan"], "ch11_tp")
            test_scan_correctness(
                "三阶段扫描 (three_phase_scan.cu)",
                ext_tp.three_phase_scan, single_block_cases)
            bench_fns.append(("D 三阶段扫描", ext_tp.three_phase_scan))
        except Exception as e:
            print(colored(f"\n[D] 编译失败: {e}", "red"))

        # E: 分层扫描（任意长度）
        try:
            ext_hier = compile_kernel(
                "hierarchical_scan.cu",
                [CPP_DECL.format(name="hierarchical_scan")],
                ["hierarchical_scan"], "ch11_hier")
            test_scan_correctness(
                "分层 Kogge-Stone 扫描 (hierarchical_scan.cu)",
                ext_hier.hierarchical_scan, hierarchical_cases)
            bench_fns.append(("E 分层扫描", ext_hier.hierarchical_scan))
        except Exception as e:
            print(colored(f"\n[E] 编译失败: {e}", "red"))

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
