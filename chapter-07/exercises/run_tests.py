#!/usr/bin/env python3
"""
第七章 一键判题脚本
用法: python run_tests.py
"""
import os
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

torch.cuda.set_device(5)

KERNELS_DIR = Path(__file__).parent / "kernels"
SCORE = {"total": 0, "passed": 0}
REPS = 200
BENCH_H, BENCH_W = 4096, 4096
BENCH_R = 2


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


def torch_conv2d_ref(N, F_kernel):
    r = F_kernel.shape[0] // 2
    N_pad = F.pad(N.unsqueeze(0).unsqueeze(0), [r, r, r, r], value=0)
    patches = N_pad.unfold(2, 2*r+1, 1).unfold(3, 2*r+1, 1)
    return (patches * F_kernel).sum(dim=(-1, -2)).squeeze(0).squeeze(0).contiguous()


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

def test_correctness(name, fn, r, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    fsize = 2 * r + 1
    ok_all = True
    for H, W in test_cases:
        N = torch.randn(H, W, device="cuda", dtype=torch.float32)
        Fk = torch.randn(fsize, fsize, device="cuda", dtype=torch.float32)
        result = fn(N, Fk)
        expected = torch_conv2d_ref(N, Fk)
        ok = torch.allclose(result, expected, rtol=1e-3, atol=1e-3)
        if not ok:
            ok_all = False
        report(f"H={H} W={W} r={r}", ok,
               f"max diff={torch.max(torch.abs(result-expected)).item():.6f}" if not ok else "")
    return ok_all


# ============================================================
# 性能对比
# ============================================================

def run_benchmark(exts):
    configs = [
        (4096, 4096, 2, "4096×4096 r=2 (基准)"),
        (4096, 4096, 4, "4096×4096 r=4 (大滤波器)"),
        (4096, 4096, 8, "4096×4096 r=8 (超大滤波器)"),
        (1024, 1024, 2, "1024×1024 r=2 (小图)"),
        (8192, 8192, 2, "8192×8192 r=2 (大图)"),
    ]

    for H, W, r, title in configs:
        print(colored(f"\n性能对比 {title}", "bold"))
        fsize = 2 * r + 1
        N  = torch.randn(H, W, device="cuda", dtype=torch.float32)
        Fk = torch.randn(fsize, fsize, device="cuda", dtype=torch.float32)

        results = {}
        for label, fn in exts:
            try:
                t = benchmark_fn(fn, N, Fk)
                results[label] = t
            except Exception as e:
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
            color = "green" if (baseline and t < baseline * 0.98) else ("red" if (baseline and t > baseline * 1.02) else "yellow")
            print(f"  {label:{max_label_len}}  {colored(f'{t:6.3f} ms', color)}  {bar}{speedup}")


# ============================================================
# 主函数
# ============================================================

def main():
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print(colored("警告: 未检测到 CUDA 设备", "yellow"))
        return

    print(colored("=" * 55, "bold"))
    print(colored("  第七章 CUDA 卷积 - 正确性验证 & 性能对比", "bold"))
    print(colored("=" * 55, "bold"))

    cases = [(32, 32), (64, 64), (100, 80)]
    bench_fns = []

    # A: 朴素
    try:
        ext_naive = compile_kernel(
            "conv2d_naive.cu",
            ["torch::Tensor conv2dNaive(torch::Tensor N, torch::Tensor F_kernel);"],
            ["conv2dNaive"], "ch07_naive")
        test_correctness("朴素卷积 (conv2d_naive.cu)", ext_naive.conv2dNaive, 2, cases)
        bench_fns.append(("A 朴素", ext_naive.conv2dNaive))
    except Exception as e:
        print(colored(f"\n[A] 编译失败: {e}", "red"))

    # B: 常量内存
    try:
        ext_const = compile_kernel(
            "conv2d_const_mem.cu",
            ["torch::Tensor conv2dConstMem(torch::Tensor N, torch::Tensor F_kernel);"],
            ["conv2dConstMem"], "ch07_const")
        test_correctness("常量内存卷积 (conv2d_const_mem.cu)", ext_const.conv2dConstMem, 2, cases)
        bench_fns.append(("B 常量", ext_const.conv2dConstMem))
    except Exception as e:
        print(colored(f"\n[B] 编译失败: {e}", "red"))

    # C: 分块
    try:
        ext_tiled = compile_kernel(
            "conv2d_tiled.cu",
            ["torch::Tensor conv2dTiled(torch::Tensor N, torch::Tensor F_kernel);"],
            ["conv2dTiled"], "ch07_tiled")
        test_correctness("分块卷积 (conv2d_tiled.cu)", ext_tiled.conv2dTiled, 2, cases)
        bench_fns.append(("C 分块", ext_tiled.conv2dTiled))
    except Exception as e:
        print(colored(f"\n[C] 编译失败: {e}", "red"))

    # D: 分块+L2
    try:
        ext_l2 = compile_kernel(
            "conv2d_tiled_l2.cu",
            ["torch::Tensor conv2dTiledL2(torch::Tensor N, torch::Tensor F_kernel);"],
            ["conv2dTiledL2"], "ch07_l2")
        test_correctness("分块+L2 卷积 (conv2d_tiled_l2.cu)", ext_l2.conv2dTiledL2, 2, cases)
        bench_fns.append(("D 分块+L2", ext_l2.conv2dTiledL2))
    except Exception as e:
        print(colored(f"\n[D] 编译失败: {e}", "red"))

    if bench_fns:
        run_benchmark(bench_fns)

    print(colored("\n" + "=" * 55, "bold"))
    total = SCORE["total"]
    passed = SCORE["passed"]
    pct = (passed / total * 100) if total > 0 else 0
    color = "green" if pct >= 80 else ("yellow" if pct >= 50 else "red")
    print(colored(f"  正确性: {passed}/{total} ({pct:.0f}%)", color))
    print(colored("=" * 55, "bold"))


if __name__ == "__main__":
    main()
