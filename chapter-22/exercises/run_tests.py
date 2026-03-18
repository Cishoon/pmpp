#!/usr/bin/env python3
"""
第二十二章 FlashAttention 一键判题脚本
用法: python run_tests.py
"""
import os
import math
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


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


def reference_attention(Q, K, V, causal=False):
    """PyTorch 参考实现"""
    d = Q.shape[-1]
    S = Q @ K.T / math.sqrt(d)
    if causal:
        N = Q.shape[0]
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S.masked_fill_(mask, float("-inf"))
    P = F.softmax(S, dim=-1)
    return P @ V


def benchmark_fn(fn, *args, warmup=10):
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

CPP_DECL_3ARG = "torch::Tensor {name}(torch::Tensor Q, torch::Tensor K, torch::Tensor V);"


def test_attention_correctness(name, fn, test_cases, causal=False):
    """测试 Attention 内核的正确性"""
    print(colored(f"\n[代码题] {name}", "bold"))
    ok_all = True
    for N, d in test_cases:
        Q = torch.randn(N, d, device="cuda", dtype=torch.float32)
        K = torch.randn(N, d, device="cuda", dtype=torch.float32)
        V = torch.randn(N, d, device="cuda", dtype=torch.float32)

        result = fn(Q, K, V)
        expected = reference_attention(Q, K, V, causal=causal)

        # 使用相对宽松的容差（浮点累加顺序不同会导致微小差异）
        ok = torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
        if not ok:
            max_diff = (result - expected).abs().max().item()
            ok_all = False
        report(
            f"N={N}, d={d}",
            ok,
            f"max_diff={max_diff:.6f}" if not ok else "",
        )
    return ok_all


# ============================================================
# 性能对比
# ============================================================

def run_benchmark(exts):
    configs = [
        (256, 64, "N=256, d=64"),
        (512, 64, "N=512, d=64"),
        (1024, 64, "N=1024, d=64"),
    ]

    for N, d, title in configs:
        print(colored(f"\n性能对比 {title}", "bold"))
        Q = torch.randn(N, d, device="cuda", dtype=torch.float32)
        K = torch.randn(N, d, device="cuda", dtype=torch.float32)
        V = torch.randn(N, d, device="cuda", dtype=torch.float32)

        # PyTorch 基线
        def pytorch_attn(Q, K, V):
            return reference_attention(Q, K, V)

        results = {}
        results["PyTorch 基线"] = benchmark_fn(pytorch_attn, Q, K, V)

        for label, fn in exts:
            try:
                t = benchmark_fn(fn, Q, K, V)
                results[label] = t
            except Exception:
                results[label] = None

        valid = {k: v for k, v in results.items() if v is not None}
        if not valid:
            continue
        baseline_key = "PyTorch 基线"
        baseline = valid.get(baseline_key, next(iter(valid.values())))
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
    print(colored("\n[简答题] 习题 1-7", "bold"))
    try:
        from answers import ex1, ex2, ex3, ex4, ex5, ex6, ex7
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

    # 习题 1: 朴素 Attention HBM 读写量
    # S = Q@K^T: 读 Nd + Nd, 写 N² => 2Nd + N²
    # P = softmax(S): 读 N², 写 N² => 2N²
    # O = P@V: 读 N² + Nd, 写 Nd => N² + 2Nd
    # 总计: 4Nd + 4N² => 但选项中最接近的是 C: O(3Nd + 4N²)
    # 更精确: 读 Q(Nd) + K(Nd) + S(N²) + P(N²) + V(Nd) = 3Nd + 2N²
    #         写 S(N²) + P(N²) + O(Nd) = Nd + 2N²
    #         总计 = 4Nd + 4N² => C
    check("1: 朴素 Attention HBM 读写量", ex1, "C",
          lambda x: str(x).upper().strip())

    # 习题 2: FlashAttention HBM 读写量
    # Q 被读 T_c 次（每个 Q 块遍历所有 K/V 块）=> T_c * Nd = (N/B_c) * Nd = N²d/B_c
    # K,V 被读 T_r 次 => T_r * Nd * 2
    # 总计 O(N²d / M) 其中 M ~ B_c * d（SRAM 大小）
    check("2: FlashAttention HBM 读写量", ex2, "C",
          lambda x: str(x).upper().strip())

    # 习题 3: 在线 softmax 更新
    # m_new = max(2.0, 3.0) = 3.0
    # l_new = 10.0 * exp(2.0 - 3.0) + 5.0 = 10.0 * exp(-1) + 5.0
    #       = 10.0 * 0.3679 + 5.0 = 3.679 + 5.0 = 8.68
    if ex3 is not None:
        m_new_expected = 3.0
        l_new_expected = 8.68
        try:
            m_ans, l_ans = ex3
            m_ok = abs(m_ans - m_new_expected) < 0.01
            l_ok = abs(l_ans - l_new_expected) < 0.05
            ok = m_ok and l_ok
            report("3: 在线 softmax 更新 (m_new, l_new)", ok,
                   f"你的答案=({m_ans}, {l_ans}), 正确答案=({m_new_expected}, {l_new_expected})"
                   if not ok else "")
        except (TypeError, ValueError):
            report("3: 在线 softmax 更新", False, f"格式错误，应为元组: {ex3!r}")
    else:
        report("3: 在线 softmax 更新 (m_new, l_new)", False, "未作答")

    # 习题 4: 分块大小与共享内存
    # (3*B*64 + B²) * 4 ≤ 49152
    # 192B + B² ≤ 12288
    # B² + 192B - 12288 ≤ 0
    # B = (-192 + sqrt(192² + 4*12288)) / 2 = (-192 + sqrt(36864 + 49152)) / 2
    #   = (-192 + sqrt(86016)) / 2 = (-192 + 293.28) / 2 = 50.64
    # 所以 B_max = 50
    check("4: 最大分块大小 B", ex4, 50)

    # 习题 5: 数值等价性
    check("5: FlashAttention 数值等价性", ex5, "A",
          lambda x: str(x).upper().strip())

    # 习题 6: FlashAttention-2 关键优化
    check("6: FlashAttention-2 关键优化", ex6, "B",
          lambda x: str(x).upper().strip())

    # 习题 7: 因果掩码计算节省
    # N=1024, B=64 => 16 个块
    # 总块对: 16 * 16 = 256
    # 因果掩码跳过的块: 上三角（不含对角线）= 16*15/2 = 120
    # 跳过比例: 120/256 = 46.875% ≈ 47%
    check("7: 因果掩码跳过比例", ex7, "C",
          lambda x: str(x).upper().strip())


# ============================================================
# 主函数
# ============================================================

def main():
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print(colored("警告: 未检测到 CUDA 设备，代码题将跳过", "yellow"))

    print(colored("=" * 60, "bold"))
    print(colored("  第二十二章 FlashAttention - 正确性验证 & 性能对比", "bold"))
    print(colored("=" * 60, "bold"))

    # 测试用例: (N, d)
    small_cases = [(64, 32), (128, 64)]
    full_cases = [(64, 32), (128, 64), (256, 64), (512, 64)]

    bench_fns = []

    if cuda_available:
        # A: 朴素 Attention
        try:
            ext_naive = compile_kernel(
                "naive_attention.cu",
                [CPP_DECL_3ARG.format(name="naiveAttention")],
                ["naiveAttention"], "ch22_naive")
            test_attention_correctness(
                "朴素 Attention (naive_attention.cu)",
                ext_naive.naiveAttention, full_cases)
            bench_fns.append(("A 朴素", ext_naive.naiveAttention))
        except Exception as e:
            print(colored(f"\n[A] 编译失败: {e}", "red"))

        # B: FlashAttention 单 block（1-pass）
        try:
            ext_1pass = compile_kernel(
                "flash_attention_1pass.cu",
                [CPP_DECL_3ARG.format(name="flashAttention1Pass")],
                ["flashAttention1Pass"], "ch22_1pass")
            test_attention_correctness(
                "FlashAttention 1-pass (flash_attention_1pass.cu)",
                ext_1pass.flashAttention1Pass, full_cases)
            bench_fns.append(("B 1-pass", ext_1pass.flashAttention1Pass))
        except Exception as e:
            print(colored(f"\n[B] 编译失败: {e}", "red"))

        # C: FlashAttention 分块版
        try:
            ext_tiled = compile_kernel(
                "flash_attention_tiled.cu",
                [CPP_DECL_3ARG.format(name="flashAttentionTiled")],
                ["flashAttentionTiled"], "ch22_tiled")
            test_attention_correctness(
                "FlashAttention 分块版 (flash_attention_tiled.cu)",
                ext_tiled.flashAttentionTiled, full_cases)
            bench_fns.append(("C 分块", ext_tiled.flashAttentionTiled))
        except Exception as e:
            print(colored(f"\n[C] 编译失败: {e}", "red"))

        # D: FlashAttention 因果掩码版
        try:
            ext_causal = compile_kernel(
                "flash_attention_causal.cu",
                [CPP_DECL_3ARG.format(name="flashAttentionCausal")],
                ["flashAttentionCausal"], "ch22_causal")
            test_attention_correctness(
                "FlashAttention 因果掩码版 (flash_attention_causal.cu)",
                ext_causal.flashAttentionCausal, full_cases, causal=True)
            bench_fns.append(("D 因果", ext_causal.flashAttentionCausal))
        except Exception as e:
            print(colored(f"\n[D] 编译失败: {e}", "red"))

        if bench_fns:
            run_benchmark(bench_fns)

    test_written_answers()

    print(colored("\n" + "=" * 60, "bold"))
    total = SCORE["total"]
    passed = SCORE["passed"]
    pct = (passed / total * 100) if total > 0 else 0
    color = "green" if pct >= 80 else ("yellow" if pct >= 50 else "red")
    print(colored(f"  总分: {passed}/{total} ({pct:.0f}%)", color))
    print(colored("=" * 60, "bold"))


if __name__ == "__main__":
    main()
