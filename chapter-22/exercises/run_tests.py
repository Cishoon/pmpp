#!/usr/bin/env python3
"""
第二十二章 FlashAttention 一键判题脚本
用法: python run_tests.py

代码题：调用 nvcc 编译纯 CUDA 内核，运行可执行文件验证正确性和性能
简答题：直接在 Python 中判题
"""
import os
import subprocess
import sys
from pathlib import Path

EXERCISES_DIR = Path(__file__).parent
SCORE = {"total": 0, "passed": 0}


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


# ============================================================
# 代码题：编译 + 运行
# ============================================================

EXERCISES = [
    ("A 朴素 Attention",       "test_naive",  "naive_attention.cu"),
    ("B FlashAttention 1-pass", "test_1pass",  "flash_attention_1pass.cu"),
    ("C FlashAttention 分块",   "test_tiled",  "flash_attention_tiled.cu"),
    ("D FlashAttention 因果",   "test_causal", "flash_attention_causal.cu"),
]


def run_code_exercises():
    print(colored("\n[代码题] 编译 & 运行", "bold"))

    for label, target, cu_file in EXERCISES:
        print(colored(f"\n  --- {label} ({cu_file}) ---", "bold"))

        # 编译
        compile_result = subprocess.run(
            ["make", target],
            cwd=str(EXERCISES_DIR),
            capture_output=True, text=True, timeout=120,
        )

        if compile_result.returncode != 0:
            print(colored(f"  编译失败:", "red"))
            # 只打印 stderr 的最后 20 行，避免刷屏
            err_lines = compile_result.stderr.strip().split("\n")
            for line in err_lines[-20:]:
                print(f"    {line}")
            report(f"{label} 编译", False, "nvcc 编译错误")
            continue

        report(f"{label} 编译", True)

        # 运行
        exe_path = EXERCISES_DIR / target
        try:
            run_result = subprocess.run(
                [str(exe_path)],
                cwd=str(EXERCISES_DIR),
                capture_output=True, text=True, timeout=60,
            )
            print(run_result.stdout)
            if run_result.stderr:
                print(run_result.stderr)

            # 解析输出中的 PASS/FAIL
            pass_count = run_result.stdout.count("✓ PASS")
            fail_count = run_result.stdout.count("✗ FAIL")
            for _ in range(pass_count):
                SCORE["total"] += 1
                SCORE["passed"] += 1
            for _ in range(fail_count):
                SCORE["total"] += 1

        except subprocess.TimeoutExpired:
            report(f"{label} 运行", False, "超时 (60s)")
        except Exception as e:
            report(f"{label} 运行", False, str(e))


# ============================================================
# 简答题
# ============================================================

def test_written_answers():
    print(colored("\n[简答题] 习题 1-7", "bold"))
    try:
        sys.path.insert(0, str(EXERCISES_DIR))
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

    check("1: 朴素 Attention HBM 读写量", ex1, "C",
          lambda x: str(x).upper().strip())

    check("2: FlashAttention HBM 读写量", ex2, "C",
          lambda x: str(x).upper().strip())

    # 习题 3: 在线 softmax
    if ex3 is not None:
        try:
            m_ans, l_ans = ex3
            m_ok = abs(m_ans - 3.0) < 0.01
            l_ok = abs(l_ans - 8.68) < 0.05
            ok = m_ok and l_ok
            report("3: 在线 softmax 更新 (m_new, l_new)", ok,
                   f"你的答案=({m_ans}, {l_ans}), 正确答案=(3.0, 8.68)"
                   if not ok else "")
        except (TypeError, ValueError):
            report("3: 在线 softmax 更新", False, f"格式错误: {ex3!r}")
    else:
        report("3: 在线 softmax 更新 (m_new, l_new)", False, "未作答")

    check("4: 最大分块大小 B", ex4, 50)

    check("5: FlashAttention 数值等价性", ex5, "A",
          lambda x: str(x).upper().strip())

    check("6: FlashAttention-2 关键优化", ex6, "B",
          lambda x: str(x).upper().strip())

    check("7: 因果掩码跳过比例", ex7, "C",
          lambda x: str(x).upper().strip())


# ============================================================
# 主函数
# ============================================================

def main():
    print(colored("=" * 60, "bold"))
    print(colored("  第二十二章 FlashAttention - 一键判题", "bold"))
    print(colored("=" * 60, "bold"))

    # 先 clean
    subprocess.run(["make", "clean"], cwd=str(EXERCISES_DIR),
                   capture_output=True, timeout=10)

    run_code_exercises()
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
