#!/usr/bin/env python3
"""
第十六章 一键判题脚本
用法: python run_tests.py
"""
import os
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

import torch
import torch.nn.functional as F
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


# ============================================================
# 参考实现（使用 PyTorch）
# ============================================================

def ref_conv2d_forward(input, weights, bias, pad_h, pad_w, stride_h, stride_w):
    return F.conv2d(input, weights, bias, stride=(stride_h, stride_w), padding=(pad_h, pad_w))


def ref_conv2d_backward_input(weights, grad_output, input_shape, pad_h, pad_w, stride_h, stride_w):
    """用 PyTorch autograd 计算输入梯度"""
    dummy = torch.randn(input_shape, device=weights.device, requires_grad=True)
    out = F.conv2d(dummy, weights, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    out.backward(grad_output)
    return dummy.grad.clone()


def ref_conv2d_backward_weights(input, grad_output, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w):
    """用 PyTorch autograd 计算权重和偏置梯度"""
    out_channels = grad_output.shape[1]
    in_channels = input.shape[1]
    w = torch.randn(out_channels, in_channels, kernel_h, kernel_w, device=input.device, requires_grad=True)
    b = torch.randn(out_channels, device=input.device, requires_grad=True)
    out = F.conv2d(input, w, b, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    out.backward(grad_output)
    return w.grad.clone(), b.grad.clone()


def ref_maxpool2d_forward(input, kernel_h, kernel_w, stride_h, stride_w):
    return F.max_pool2d(input, (kernel_h, kernel_w), (stride_h, stride_w), return_indices=True)


def ref_maxpool2d_backward(input, kernel_h, kernel_w, stride_h, stride_w):
    """用 PyTorch autograd 计算 maxpool 输入梯度"""
    inp = input.clone().requires_grad_(True)
    out = F.max_pool2d(inp, (kernel_h, kernel_w), (stride_h, stride_w))
    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    return grad_output, inp.grad.clone()


# ============================================================
# 正确性测试
# ============================================================

def test_conv2d_forward(name, fn, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    for bs, ic, h, w, oc, kh, kw, ph, pw, sh, sw in test_cases:
        input = torch.randn(bs, ic, h, w, device="cuda")
        weights = torch.randn(oc, ic, kh, kw, device="cuda")
        bias = torch.randn(oc, device="cuda")
        expected = ref_conv2d_forward(input, weights, bias, ph, pw, sh, sw)
        result = fn(input, weights, bias, ph, pw, sh, sw)
        ok = torch.allclose(result, expected, rtol=1e-3, atol=1e-4)
        desc = f"B={bs} C={ic}->{oc} {h}x{w} K={kh}x{kw} P={ph} S={sh}"
        if not ok:
            max_diff = (result - expected).abs().max().item()
            report(desc, False, f"max_diff={max_diff:.6f}")
        else:
            t = benchmark_fn(fn, input, weights, bias, ph, pw, sh, sw, warmup=10)
            report(f"{desc}  {t:.4f} ms", True)


def test_conv2d_backward_input(name, fn, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    for bs, ic, h, w, oc, kh, kw, ph, pw, sh, sw in test_cases:
        weights = torch.randn(oc, ic, kh, kw, device="cuda")
        oh = (h + 2 * ph - kh) // sh + 1
        ow = (w + 2 * pw - kw) // sw + 1
        grad_output = torch.randn(bs, oc, oh, ow, device="cuda")
        expected = ref_conv2d_backward_input(weights, grad_output, (bs, ic, h, w), ph, pw, sh, sw)
        result = fn(weights, grad_output, h, w, ph, pw, sh, sw)
        ok = torch.allclose(result, expected, rtol=1e-3, atol=1e-4)
        desc = f"B={bs} C={ic}->{oc} {h}x{w} K={kh}x{kw} P={ph} S={sh}"
        if not ok:
            max_diff = (result - expected).abs().max().item()
            report(desc, False, f"max_diff={max_diff:.6f}")
        else:
            t = benchmark_fn(fn, weights, grad_output, h, w, ph, pw, sh, sw, warmup=10)
            report(f"{desc}  {t:.4f} ms", True)


def test_conv2d_backward_weights(name, fn, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    for bs, ic, h, w, oc, kh, kw, ph, pw, sh, sw in test_cases:
        input = torch.randn(bs, ic, h, w, device="cuda")
        oh = (h + 2 * ph - kh) // sh + 1
        ow = (w + 2 * pw - kw) // sw + 1
        grad_output = torch.randn(bs, oc, oh, ow, device="cuda")
        expected_w, expected_b = ref_conv2d_backward_weights(input, grad_output, kh, kw, ph, pw, sh, sw)
        result_w, result_b = fn(input, grad_output, kh, kw, ph, pw, sh, sw)
        ok_w = torch.allclose(result_w, expected_w, rtol=1e-3, atol=1e-4)
        ok_b = torch.allclose(result_b, expected_b, rtol=1e-3, atol=1e-4)
        desc = f"B={bs} C={ic}->{oc} {h}x{w} K={kh}x{kw} P={ph} S={sh}"
        if not (ok_w and ok_b):
            dw = (result_w - expected_w).abs().max().item()
            db = (result_b - expected_b).abs().max().item()
            report(desc, False, f"max_diff_w={dw:.6f} max_diff_b={db:.6f}")
        else:
            t = benchmark_fn(fn, input, grad_output, kh, kw, ph, pw, sh, sw, warmup=10)
            report(f"{desc}  {t:.4f} ms", True)


def test_maxpool2d_forward(name, fn, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    for bs, c, h, w, kh, kw, sh, sw in test_cases:
        input = torch.randn(bs, c, h, w, device="cuda")
        expected_out, _ = ref_maxpool2d_forward(input, kh, kw, sh, sw)
        result_out, result_idx = fn(input, kh, kw, sh, sw)
        # 只比较输出值（indices 格式不同：我们用窗口内相对索引，PyTorch 用平面索引）
        ok = torch.allclose(result_out, expected_out, rtol=1e-5, atol=1e-6)
        desc = f"B={bs} C={c} {h}x{w} K={kh}x{kw} S={sh}"
        if not ok:
            max_diff = (result_out - expected_out).abs().max().item()
            report(desc, False, f"max_diff={max_diff:.6f}")
        else:
            t = benchmark_fn(fn, input, kh, kw, sh, sw, warmup=10)
            report(f"{desc}  {t:.4f} ms", True)


def _pytorch_indices_to_relative(flat_indices, h, w, kh, kw, sh, sw):
    """将 PyTorch 的平面索引转换为窗口内相对索引 (kh_pos * kw + kw_pos)"""
    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1
    # flat_indices 是在 H*W 平面内的索引
    h_in = flat_indices // w  # 输入行
    w_in = flat_indices % w   # 输入列
    # 计算输出位置对应的窗口起始位置
    bs_dim, c_dim = flat_indices.shape[0], flat_indices.shape[1]
    h_out_idx = torch.arange(oh, device=flat_indices.device).view(1, 1, oh, 1).expand(bs_dim, c_dim, oh, ow)
    w_out_idx = torch.arange(ow, device=flat_indices.device).view(1, 1, 1, ow).expand(bs_dim, c_dim, oh, ow)
    h_start = h_out_idx * sh
    w_start = w_out_idx * sw
    kh_pos = h_in - h_start
    kw_pos = w_in - w_start
    return (kh_pos * kw + kw_pos).to(torch.int32)


def test_maxpool2d_backward(name, fn, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    for bs, c, h, w, kh, kw, sh, sw in test_cases:
        input = torch.randn(bs, c, h, w, device="cuda")
        oh = (h - kh) // sh + 1
        ow = (w - kw) // sw + 1
        grad_output = torch.randn(bs, c, oh, ow, device="cuda")
        # 参考：用 PyTorch autograd
        inp_ag = input.clone().requires_grad_(True)
        out_ag = F.max_pool2d(inp_ag, (kh, kw), (sh, sw))
        out_ag.backward(grad_output)
        expected = inp_ag.grad.clone()
        # 获取 PyTorch 的 flat indices 并转换为窗口内相对索引
        _, pt_indices = F.max_pool2d(input, (kh, kw), (sh, sw), return_indices=True)
        rel_indices = _pytorch_indices_to_relative(pt_indices, h, w, kh, kw, sh, sw)
        result = fn(grad_output, rel_indices, h, w, kh, kw, sh, sw)
        ok = torch.allclose(result, expected, rtol=1e-4, atol=1e-5)
        desc = f"B={bs} C={c} {h}x{w} K={kh}x{kw} S={sh}"
        if not ok:
            max_diff = (result - expected).abs().max().item()
            report(desc, False, f"max_diff={max_diff:.6f}")
        else:
            t = benchmark_fn(fn, grad_output, rel_indices, h, w, kh, kw, sh, sw, warmup=10)
            report(f"{desc}  {t:.4f} ms", True)


# ============================================================
# 性能对比
# ============================================================

def run_benchmark(exts):
    conv_configs = [
        ((2, 3, 32, 32, 16, 3, 3, 1, 1, 1, 1), "Conv2D 2x3x32x32 -> 16 K=3 P=1 S=1"),
        ((4, 3, 64, 64, 32, 3, 3, 1, 1, 1, 1), "Conv2D 4x3x64x64 -> 32 K=3 P=1 S=1"),
        ((8, 16, 128, 128, 32, 5, 5, 2, 2, 1, 1), "Conv2D 8x16x128x128 -> 32 K=5 P=2 S=1"),
    ]

    pool_configs = [
        ((4, 16, 32, 32, 2, 2, 2, 2), "MaxPool2D 4x16x32x32 K=2 S=2"),
        ((8, 32, 64, 64, 2, 2, 2, 2), "MaxPool2D 8x32x64x64 K=2 S=2"),
    ]

    for params, title in conv_configs:
        bs, ic, h, w, oc, kh, kw, ph, pw, sh, sw = params
        print(colored(f"\n性能对比 {title}", "bold"))
        input = torch.randn(bs, ic, h, w, device="cuda")
        weights = torch.randn(oc, ic, kh, kw, device="cuda")
        bias = torch.randn(oc, device="cuda")

        results = {}
        # PyTorch 参考
        try:
            t = benchmark_fn(ref_conv2d_forward, input, weights, bias, ph, pw, sh, sw)
            results["PyTorch Conv2D"] = t
        except Exception:
            pass

        for label, fn, kind in exts:
            if kind != "conv_fwd":
                continue
            try:
                t = benchmark_fn(fn, input, weights, bias, ph, pw, sh, sw)
                results[label] = t
            except Exception:
                results[label] = None

        _print_benchmark_results(results)

    for params, title in pool_configs:
        bs, c, h, w, kh, kw, sh, sw = params
        print(colored(f"\n性能对比 {title}", "bold"))
        input = torch.randn(bs, c, h, w, device="cuda")

        results = {}
        try:
            t = benchmark_fn(ref_maxpool2d_forward, input, kh, kw, sh, sw)
            results["PyTorch MaxPool2D"] = t
        except Exception:
            pass

        for label, fn, kind in exts:
            if kind != "pool_fwd":
                continue
            try:
                t = benchmark_fn(fn, input, kh, kw, sh, sw)
                results[label] = t
            except Exception:
                results[label] = None

        _print_benchmark_results(results)


def _print_benchmark_results(results):
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        return
    baseline_key = next(iter(valid))
    baseline = valid[baseline_key]
    max_t = max(valid.values())
    max_label_len = max(len(k) for k in valid)

    for label, t in results.items():
        if t is None:
            print(f"  {label}: 跳过")
            continue
        speedup = f"  ({baseline / t:.2f}x)" if label != baseline_key else ""
        bar = "█" * int(t / max_t * 28)
        color = "green" if (t < baseline * 0.98) else (
            "red" if (t > baseline * 1.02) else "yellow")
        if label == baseline_key:
            color = "yellow"
        print(f"  {label:{max_label_len}}  {colored(f'{t:6.3f} ms', color)}  {bar}{speedup}")


# ============================================================
# C++ 接口声明
# ============================================================

CPP_CONV2D_FWD = """
torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weights, torch::Tensor bias,
                              int pad_h, int pad_w, int stride_h, int stride_w);
"""

CPP_CONV2D_BWD_INPUT = """
torch::Tensor conv2d_backward_input(torch::Tensor weights, torch::Tensor grad_output,
                                     int height, int width, int pad_h, int pad_w,
                                     int stride_h, int stride_w);
"""

CPP_CONV2D_BWD_WEIGHTS = """
std::vector<torch::Tensor> conv2d_backward_weights(torch::Tensor input, torch::Tensor grad_output,
                                                     int kernel_h, int kernel_w,
                                                     int pad_h, int pad_w,
                                                     int stride_h, int stride_w);
"""

CPP_MAXPOOL_FWD = """
std::vector<torch::Tensor> maxpool2d_forward(torch::Tensor input,
                                              int kernel_h, int kernel_w,
                                              int stride_h, int stride_w);
"""

CPP_MAXPOOL_BWD = """
torch::Tensor maxpool2d_backward(torch::Tensor grad_output, torch::Tensor indices,
                                  int height, int width,
                                  int kernel_h, int kernel_w,
                                  int stride_h, int stride_w);
"""


# ============================================================
# 主函数
# ============================================================

def main():
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print(colored("警告: 未检测到 CUDA 设备，代码题将跳过", "yellow"))

    print(colored("=" * 55, "bold"))
    print(colored("  第十六章 CUDA DNN 前向/反向传播 - 正确性验证 & 性能对比", "bold"))
    print(colored("=" * 55, "bold"))

    conv_test_cases = [
        # (bs, ic, h, w, oc, kh, kw, ph, pw, sh, sw)
        (1, 1, 8, 8, 1, 3, 3, 0, 0, 1, 1),
        (2, 3, 16, 16, 8, 3, 3, 1, 1, 1, 1),
        (4, 3, 28, 28, 16, 5, 5, 2, 2, 1, 1),
        (2, 16, 32, 32, 32, 3, 3, 1, 1, 2, 2),
    ]

    pool_test_cases = [
        # (bs, c, h, w, kh, kw, sh, sw)
        (1, 1, 8, 8, 2, 2, 2, 2),
        (2, 3, 16, 16, 2, 2, 2, 2),
        (4, 16, 32, 32, 2, 2, 2, 2),
        (2, 32, 28, 28, 3, 3, 3, 3),
    ]

    bench_fns = []

    if cuda_available:
        # A: Conv2D 前向
        try:
            ext_conv_fwd = compile_kernel(
                "conv2d_forward.cu", [CPP_CONV2D_FWD],
                ["conv2d_forward"], "ch16_conv2d_fwd")
            test_conv2d_forward(
                "Conv2D 前向传播 (conv2d_forward.cu)",
                ext_conv_fwd.conv2d_forward, conv_test_cases)
            bench_fns.append(("A Conv2D 前向", ext_conv_fwd.conv2d_forward, "conv_fwd"))
        except Exception as e:
            print(colored(f"\n[A] 编译失败: {e}", "red"))

        # B: Conv2D 反向 - 输入梯度
        try:
            ext_conv_bwd_input = compile_kernel(
                "conv2d_backward_input.cu", [CPP_CONV2D_BWD_INPUT],
                ["conv2d_backward_input"], "ch16_conv2d_bwd_input")
            test_conv2d_backward_input(
                "Conv2D 反向传播 - 输入梯度 (conv2d_backward_input.cu)",
                ext_conv_bwd_input.conv2d_backward_input, conv_test_cases)
        except Exception as e:
            print(colored(f"\n[B] 编译失败: {e}", "red"))

        # C: Conv2D 反向 - 权重梯度
        try:
            ext_conv_bwd_w = compile_kernel(
                "conv2d_backward_weights.cu", [CPP_CONV2D_BWD_WEIGHTS],
                ["conv2d_backward_weights"], "ch16_conv2d_bwd_w")
            test_conv2d_backward_weights(
                "Conv2D 反向传播 - 权重梯度 (conv2d_backward_weights.cu)",
                ext_conv_bwd_w.conv2d_backward_weights, conv_test_cases)
        except Exception as e:
            print(colored(f"\n[C] 编译失败: {e}", "red"))

        # D: MaxPool2D 前向
        try:
            ext_pool_fwd = compile_kernel(
                "maxpool2d_forward.cu", [CPP_MAXPOOL_FWD],
                ["maxpool2d_forward"], "ch16_maxpool_fwd")
            test_maxpool2d_forward(
                "MaxPool2D 前向传播 (maxpool2d_forward.cu)",
                ext_pool_fwd.maxpool2d_forward, pool_test_cases)
            bench_fns.append(("D MaxPool2D 前向", ext_pool_fwd.maxpool2d_forward, "pool_fwd"))
        except Exception as e:
            print(colored(f"\n[D] 编译失败: {e}", "red"))

        # E: MaxPool2D 反向
        try:
            ext_pool_bwd = compile_kernel(
                "maxpool2d_backward.cu", [CPP_MAXPOOL_BWD],
                ["maxpool2d_backward"], "ch16_maxpool_bwd")
            test_maxpool2d_backward(
                "MaxPool2D 反向传播 (maxpool2d_backward.cu)",
                ext_pool_bwd.maxpool2d_backward, pool_test_cases)
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
