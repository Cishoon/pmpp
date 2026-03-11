#!/usr/bin/env python3
"""
第三章 一键判题脚本
用法: python run_tests.py
"""
import os
import sys
from pathlib import Path

os.environ["CC"] = "x86_64-conda-linux-gnu-gcc"
os.environ["CXX"] = "x86_64-conda-linux-gnu-g++"
os.environ["CUDA_HOME"] = "/usr/local/cuda"

import torch
from torch.utils.cpp_extension import load_inline

KERNELS_DIR = Path(__file__).parent / "kernels"
SCORE = {"total": 0, "passed": 0}


def colored(text, color):
    colors = {"green": "\033[92m", "red": "\033[91m", "yellow": "\033[93m", "bold": "\033[1m", "end": "\033[0m"}
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
    )


# ============================================================
# 代码题测试
# ============================================================

def test_matmul_row():
    print(colored("\n[习题 1a] 按行矩阵乘法内核", "bold"))
    try:
        ext = compile_kernel(
            "matmul_row.cu",
            ["torch::Tensor matrixRowMul(torch::Tensor M, torch::Tensor N);"],
            ["matrixRowMul"],
            "test_matmul_row",
        )
        for size in [4, 64, 128]:
            M = torch.randn(size, size, device="cuda", dtype=torch.float32)
            N = torch.randn(size, size, device="cuda", dtype=torch.float32)
            result = ext.matrixRowMul(M, N)
            expected = torch.matmul(M, N)
            ok = torch.allclose(result, expected, rtol=1e-3, atol=1e-3)
            report(f"size={size}", ok, f"max diff={torch.max(torch.abs(result - expected)).item():.6f}")
    except Exception as e:
        report("编译/运行失败", False, str(e)[:120])


def test_matmul_col():
    print(colored("\n[习题 1b] 按列矩阵乘法内核", "bold"))
    try:
        ext = compile_kernel(
            "matmul_col.cu",
            ["torch::Tensor matrixColMul(torch::Tensor M, torch::Tensor N);"],
            ["matrixColMul"],
            "test_matmul_col",
        )
        for size in [4, 64, 128]:
            M = torch.randn(size, size, device="cuda", dtype=torch.float32)
            N = torch.randn(size, size, device="cuda", dtype=torch.float32)
            result = ext.matrixColMul(M, N)
            expected = torch.matmul(M, N)
            ok = torch.allclose(result, expected, rtol=1e-3, atol=1e-3)
            report(f"size={size}", ok, f"max diff={torch.max(torch.abs(result - expected)).item():.6f}")
    except Exception as e:
        report("编译/运行失败", False, str(e)[:120])


def test_mat_vec_mul():
    print(colored("\n[习题 2] 矩阵-向量乘法内核", "bold"))
    try:
        ext = compile_kernel(
            "mat_vec_mul.cu",
            ["torch::Tensor matrix_vector_multiplication(torch::Tensor B, torch::Tensor c);"],
            ["matrix_vector_multiplication"],
            "test_mat_vec_mul",
        )
        for rows, cols in [(100, 256), (1000, 64), (1, 512)]:
            B = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
            c = torch.randn(cols, device="cuda", dtype=torch.float32)
            result = ext.matrix_vector_multiplication(B, c)
            expected = torch.matmul(B, c)
            ok = torch.allclose(result, expected, rtol=1e-3, atol=1e-3)
            report(f"B={rows}x{cols}", ok, f"max diff={torch.max(torch.abs(result - expected)).item():.6f}")
    except Exception as e:
        report("编译/运行失败", False, str(e)[:120])


def test_rgb_to_grayscale():
    print(colored("\n[附加题] RGB 转灰度内核", "bold"))
    try:
        ext = compile_kernel(
            "rgb_to_grayscale.cu",
            ["torch::Tensor rgb_to_gray(torch::Tensor img);"],
            ["rgb_to_gray"],
            "test_rgb_gray",
        )
        # 用随机图像测试
        for h, w in [(64, 64), (100, 200), (13, 7)]:
            img = torch.randint(0, 256, (h, w, 3), device="cuda", dtype=torch.uint8)
            result = ext.rgb_to_gray(img)
            # CPU 参考实现
            img_f = img.float()
            expected = (0.21 * img_f[:, :, 0] + 0.71 * img_f[:, :, 1] + 0.07 * img_f[:, :, 2]).to(torch.uint8)
            ok = torch.allclose(result.squeeze(-1), expected, atol=1)
            report(f"{h}x{w}", ok)
    except Exception as e:
        report("编译/运行失败", False, str(e)[:120])


def test_gaussian_blur():
    print(colored("\n[附加题] 高斯模糊内核", "bold"))
    try:
        ext = compile_kernel(
            "gaussian_blur.cu",
            ["torch::Tensor gaussian_blur(torch::Tensor img, int blurSize);"],
            ["gaussian_blur"],
            "test_blur",
        )
        # channels x height x width
        img = torch.randint(0, 256, (3, 32, 32), device="cuda", dtype=torch.uint8)
        blur_size = 1
        result = ext.gaussian_blur(img, blur_size)

        # 用 unfold 做参考实现（避免慢循环）
        import torch.nn.functional as F
        img_f = img.float().unsqueeze(0)  # 1 x C x H x W
        k = 2 * blur_size + 1
        padded = F.pad(img_f, [blur_size] * 4, mode="constant", value=0)
        # 计算每个位置的邻域和与邻域计数
        ones = torch.ones(1, 1, k, k, device="cpu")
        sum_vals = F.conv2d(padded.cpu(), ones.expand(3, 1, k, k), groups=3)
        # 计算有效像素数（处理边界）
        mask = F.pad(torch.ones_like(img_f), [blur_size] * 4, mode="constant", value=0)
        count = F.conv2d(mask.cpu(), ones.expand(3, 1, k, k), groups=3)
        expected = (sum_vals / count).to(torch.uint8).squeeze(0).cuda()
        ok = torch.allclose(result, expected, atol=1)
        report("3x32x32, blur=1", ok)
    except Exception as e:
        report("编译/运行失败", False, str(e)[:120])


# ============================================================
# 选择题/简答题测试
# ============================================================

def test_written_answers():
    print(colored("\n[习题 3-5] 选择题 & 计算题", "bold"))
    try:
        from answers import ex3a, ex3b, ex3c, ex3d, ex4a, ex4b, ex5
    except ImportError:
        report("无法导入 answers.py", False)
        return

    checks = [
        ("3a: 每块线程数", ex3a, 512),
        ("3b: 网格线程总数", ex3b, 48640),
        ("3c: 网格块数", ex3c, 95),
        ("3d: 执行第05行的线程数", ex3d, 45000),
        ("4a: 行优先索引", ex4a, 8010),
        ("4b: 列优先索引", ex4b, 5020),
        ("5:  3D张量索引", ex5, 1008010),
    ]

    for name, answer, expected in checks:
        if answer is None:
            report(name, False, "未作答")
        else:
            report(name, answer == expected, f"你的答案={answer}, 正确答案={expected}" if answer != expected else "")


# ============================================================
# 主函数
# ============================================================

def main():
    print(colored("=" * 50, "bold"))
    print(colored("  第三章 CUDA 编程练习 - 自动判题", "bold"))
    print(colored("=" * 50, "bold"))

    test_matmul_row()
    test_matmul_col()
    test_mat_vec_mul()
    test_rgb_to_grayscale()
    test_gaussian_blur()
    test_written_answers()

    print(colored("\n" + "=" * 50, "bold"))
    total = SCORE["total"]
    passed = SCORE["passed"]
    pct = (passed / total * 100) if total > 0 else 0
    color = "green" if pct >= 80 else ("yellow" if pct >= 50 else "red")
    print(colored(f"  总分: {passed}/{total} ({pct:.0f}%)", color))
    print(colored("=" * 50, "bold"))


if __name__ == "__main__":
    main()
