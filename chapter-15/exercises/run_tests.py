#!/usr/bin/env python3
"""
第十五章 一键判题脚本
用法: python run_tests.py
"""
import os
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

import torch
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
# 图生成工具
# ============================================================

def make_random_graph_csr(num_vertices, avg_degree=6, device="cuda"):
    """生成随机有向图的 CSR 格式，返回 (srcPtrs, dst, numVertices)"""
    edges_per_vertex = torch.poisson(torch.full((num_vertices,), float(avg_degree)))
    edges_per_vertex = edges_per_vertex.clamp(min=1, max=num_vertices - 1).int()

    dst_list = []
    for v in range(num_vertices):
        k = edges_per_vertex[v].item()
        # 随机选 k 个不同的邻居（排除自身）
        candidates = list(range(num_vertices))
        candidates.remove(v)
        perm = torch.randperm(len(candidates))[:k]
        neighbors = sorted([candidates[p] for p in perm])
        dst_list.extend(neighbors)

    srcPtrs = torch.zeros(num_vertices + 1, dtype=torch.int32)
    for v in range(num_vertices):
        srcPtrs[v + 1] = srcPtrs[v] + edges_per_vertex[v]

    dst_tensor = torch.tensor(dst_list, dtype=torch.int32)
    return srcPtrs.to(device), dst_tensor.to(device), num_vertices


def csr_to_csc(srcPtrs, dst, num_vertices, device="cuda"):
    """将 CSR 转换为 CSC 格式，返回 (dstPtrs, src)"""
    srcPtrs_cpu = srcPtrs.cpu()
    dst_cpu = dst.cpu()

    # 统计每个目标顶点的入边数
    in_degree = torch.zeros(num_vertices, dtype=torch.int32)
    for e in range(dst_cpu.size(0)):
        in_degree[dst_cpu[e].item()] += 1

    dstPtrs = torch.zeros(num_vertices + 1, dtype=torch.int32)
    for v in range(num_vertices):
        dstPtrs[v + 1] = dstPtrs[v] + in_degree[v]

    src_array = torch.zeros(dst_cpu.size(0), dtype=torch.int32)
    offset = torch.zeros(num_vertices, dtype=torch.int32)
    for v in range(num_vertices):
        for e in range(srcPtrs_cpu[v].item(), srcPtrs_cpu[v + 1].item()):
            d = dst_cpu[e].item()
            idx = dstPtrs[d].item() + offset[d].item()
            src_array[idx] = v
            offset[d] += 1

    return dstPtrs.to(device), src_array.to(device)


def csr_to_coo(srcPtrs, dst, num_vertices, device="cuda"):
    """将 CSR 转换为 COO 格式，返回 (edgeSrc, edgeDst)"""
    srcPtrs_cpu = srcPtrs.cpu()
    dst_cpu = dst.cpu()
    num_edges = dst_cpu.size(0)

    edgeSrc = torch.zeros(num_edges, dtype=torch.int32)
    for v in range(num_vertices):
        for e in range(srcPtrs_cpu[v].item(), srcPtrs_cpu[v + 1].item()):
            edgeSrc[e] = v

    return edgeSrc.to(device), dst_cpu.to(device)


def reference_bfs(srcPtrs, dst, num_vertices, start_vertex):
    """CPU 参考 BFS 实现，返回 levels 张量"""
    srcPtrs_cpu = srcPtrs.cpu().tolist()
    dst_cpu = dst.cpu().tolist()

    levels = [-1] * num_vertices
    levels[start_vertex] = 0
    queue = [start_vertex]

    while queue:
        next_queue = []
        for v in queue:
            for e in range(srcPtrs_cpu[v], srcPtrs_cpu[v + 1]):
                neighbor = dst_cpu[e]
                if levels[neighbor] == -1:
                    levels[neighbor] = levels[v] + 1
                    next_queue.append(neighbor)
        queue = next_queue

    return torch.tensor(levels, dtype=torch.int32, device="cuda")


# ============================================================
# 正确性测试
# ============================================================

def test_bfs_correctness(name, fn, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    for num_v, avg_deg in test_cases:
        srcPtrs, dst_arr, nv = make_random_graph_csr(num_v, avg_deg)
        start_vertex = 0
        expected = reference_bfs(srcPtrs, dst_arr, nv, start_vertex)
        try:
            result = fn(srcPtrs, dst_arr, nv, start_vertex)
        except Exception as e:
            report(f"V={num_v} deg={avg_deg}", False, str(e))
            continue

        # 只比较从起始顶点可达的顶点
        reachable = expected >= 0
        ok = torch.equal(result[reachable], expected[reachable])
        if not ok:
            diff_count = (result[reachable] != expected[reachable]).sum().item()
            report(f"V={num_v} deg={avg_deg}", False, f"{diff_count} vertices differ")
        else:
            t = benchmark_fn(fn, srcPtrs, dst_arr, nv, start_vertex, warmup=5)
            report(f"V={num_v} deg={avg_deg:<3}  {t:.4f} ms", True)


def test_bfs_pull_correctness(name, fn, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    for num_v, avg_deg in test_cases:
        srcPtrs, dst_arr, nv = make_random_graph_csr(num_v, avg_deg)
        dstPtrs, src_arr = csr_to_csc(srcPtrs, dst_arr, nv)
        start_vertex = 0
        expected = reference_bfs(srcPtrs, dst_arr, nv, start_vertex)
        try:
            result = fn(dstPtrs, src_arr, nv, start_vertex)
        except Exception as e:
            report(f"V={num_v} deg={avg_deg}", False, str(e))
            continue

        reachable = expected >= 0
        ok = torch.equal(result[reachable], expected[reachable])
        if not ok:
            diff_count = (result[reachable] != expected[reachable]).sum().item()
            report(f"V={num_v} deg={avg_deg}", False, f"{diff_count} vertices differ")
        else:
            t = benchmark_fn(fn, dstPtrs, src_arr, nv, start_vertex, warmup=5)
            report(f"V={num_v} deg={avg_deg:<3}  {t:.4f} ms", True)


def test_bfs_edge_correctness(name, fn, test_cases):
    print(colored(f"\n[代码题] {name}", "bold"))
    for num_v, avg_deg in test_cases:
        srcPtrs, dst_arr, nv = make_random_graph_csr(num_v, avg_deg)
        edgeSrc, edgeDst = csr_to_coo(srcPtrs, dst_arr, nv)
        start_vertex = 0
        expected = reference_bfs(srcPtrs, dst_arr, nv, start_vertex)
        try:
            result = fn(edgeSrc, edgeDst, nv, start_vertex)
        except Exception as e:
            report(f"V={num_v} deg={avg_deg}", False, str(e))
            continue

        reachable = expected >= 0
        ok = torch.equal(result[reachable], expected[reachable])
        if not ok:
            diff_count = (result[reachable] != expected[reachable]).sum().item()
            report(f"V={num_v} deg={avg_deg}", False, f"{diff_count} vertices differ")
        else:
            t = benchmark_fn(fn, edgeSrc, edgeDst, nv, start_vertex, warmup=5)
            report(f"V={num_v} deg={avg_deg:<3}  {t:.4f} ms", True)


# ============================================================
# 性能对比
# ============================================================

def run_benchmark(exts):
    configs = [
        (1000, 10, "1K vertices, avg_deg=10"),
        (5000, 8, "5K vertices, avg_deg=8"),
        (10000, 6, "10K vertices, avg_deg=6"),
    ]

    for num_v, avg_deg, title in configs:
        print(colored(f"\n性能对比 {title}", "bold"))
        srcPtrs, dst_arr, nv = make_random_graph_csr(num_v, avg_deg)
        dstPtrs, src_arr = csr_to_csc(srcPtrs, dst_arr, nv)
        edgeSrc, edgeDst = csr_to_coo(srcPtrs, dst_arr, nv)
        start_vertex = 0

        format_args = {
            "CSR": (srcPtrs, dst_arr, nv, start_vertex),
            "CSC": (dstPtrs, src_arr, nv, start_vertex),
            "COO": (edgeSrc, edgeDst, nv, start_vertex),
        }

        results = {}
        for label, fn, fmt_key in exts:
            if fmt_key not in format_args:
                continue
            try:
                args = format_args[fmt_key]
                t = benchmark_fn(fn, *args)
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
# C++ 绑定声明
# ============================================================

CPP_PUSH = "torch::Tensor bfs_push(torch::Tensor srcPtrs, torch::Tensor dst, torch::Tensor levels, int startVertex);"
CPP_PULL = "torch::Tensor bfs_pull(torch::Tensor dstPtrs, torch::Tensor src, torch::Tensor levels, int startVertex);"
CPP_EDGE = "torch::Tensor bfs_edge_centric(torch::Tensor edgeSrc, torch::Tensor edgeDst, torch::Tensor levels, int startVertex, int numVertices);"
CPP_FRONTIER = "torch::Tensor bfs_frontier(torch::Tensor srcPtrs, torch::Tensor dst, torch::Tensor levels, int startVertex);"
CPP_FRONTIER_PRIV = "torch::Tensor bfs_frontier_private(torch::Tensor srcPtrs, torch::Tensor dst, torch::Tensor levels, int startVertex);"


# ============================================================
# 包装函数（统一接口）
# ============================================================

def wrap_push(ext):
    def fn(srcPtrs, dst, nv, start):
        levels = torch.full((nv,), -1, dtype=torch.int32, device="cuda")
        return ext.bfs_push(srcPtrs, dst, levels, start)
    return fn


def wrap_pull(ext):
    def fn(dstPtrs, src, nv, start):
        levels = torch.full((nv,), -1, dtype=torch.int32, device="cuda")
        return ext.bfs_pull(dstPtrs, src, levels, start)
    return fn


def wrap_edge(ext):
    def fn(edgeSrc, edgeDst, nv, start):
        levels = torch.full((nv,), -1, dtype=torch.int32, device="cuda")
        return ext.bfs_edge_centric(edgeSrc, edgeDst, levels, start, nv)
    return fn


def wrap_frontier(ext):
    def fn(srcPtrs, dst, nv, start):
        levels = torch.full((nv,), -1, dtype=torch.int32, device="cuda")
        return ext.bfs_frontier(srcPtrs, dst, levels, start)
    return fn


def wrap_frontier_priv(ext):
    def fn(srcPtrs, dst, nv, start):
        levels = torch.full((nv,), -1, dtype=torch.int32, device="cuda")
        return ext.bfs_frontier_private(srcPtrs, dst, levels, start)
    return fn


# ============================================================
# 主函数
# ============================================================

def main():
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print(colored("警告: 未检测到 CUDA 设备，代码题将跳过", "yellow"))

    print(colored("=" * 55, "bold"))
    print(colored("  第十五章 CUDA 图遍历 BFS - 正确性验证 & 性能对比", "bold"))
    print(colored("=" * 55, "bold"))

    test_cases = [
        (64, 4),
        (256, 6),
        (1000, 8),
        (5000, 6),
    ]

    bench_fns = []

    if cuda_available:
        # A: Push BFS
        try:
            ext_push = compile_kernel(
                "bfs_push.cu", [CPP_PUSH], ["bfs_push"], "ch15_push")
            fn_push = wrap_push(ext_push)
            test_bfs_correctness(
                "Vertex-Centric Push BFS (bfs_push.cu)", fn_push, test_cases)
            bench_fns.append(("A Push BFS", fn_push, "CSR"))
        except Exception as e:
            print(colored(f"\n[A] 编译失败: {e}", "red"))

        # B: Pull BFS
        try:
            ext_pull = compile_kernel(
                "bfs_pull.cu", [CPP_PULL], ["bfs_pull"], "ch15_pull")
            fn_pull = wrap_pull(ext_pull)
            test_bfs_pull_correctness(
                "Vertex-Centric Pull BFS (bfs_pull.cu)", fn_pull, test_cases)
            bench_fns.append(("B Pull BFS", fn_pull, "CSC"))
        except Exception as e:
            print(colored(f"\n[B] 编译失败: {e}", "red"))

        # C: Edge-Centric BFS
        try:
            ext_edge = compile_kernel(
                "bfs_edge_centric.cu", [CPP_EDGE], ["bfs_edge_centric"], "ch15_edge")
            fn_edge = wrap_edge(ext_edge)
            test_bfs_edge_correctness(
                "Edge-Centric BFS (bfs_edge_centric.cu)", fn_edge, test_cases)
            bench_fns.append(("C Edge BFS", fn_edge, "COO"))
        except Exception as e:
            print(colored(f"\n[C] 编译失败: {e}", "red"))

        # D: Frontier BFS
        try:
            ext_frontier = compile_kernel(
                "bfs_frontier.cu", [CPP_FRONTIER], ["bfs_frontier"], "ch15_frontier")
            fn_frontier = wrap_frontier(ext_frontier)
            test_bfs_correctness(
                "Frontier-based BFS (bfs_frontier.cu)", fn_frontier, test_cases)
            bench_fns.append(("D Frontier BFS", fn_frontier, "CSR"))
        except Exception as e:
            print(colored(f"\n[D] 编译失败: {e}", "red"))

        # E: Frontier BFS with Privatization
        try:
            ext_frontier_priv = compile_kernel(
                "bfs_frontier_private.cu", [CPP_FRONTIER_PRIV], ["bfs_frontier_private"], "ch15_frontier_priv")
            fn_frontier_priv = wrap_frontier_priv(ext_frontier_priv)
            test_bfs_correctness(
                "Frontier BFS + 私有化 (bfs_frontier_private.cu)", fn_frontier_priv, test_cases)
            bench_fns.append(("E Frontier+Priv", fn_frontier_priv, "CSR"))
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
