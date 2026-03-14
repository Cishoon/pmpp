# 第十五章 练习

## 结构

```
exercises/
├── kernels/
│   ├── bfs_push.cu                # 代码题A: Vertex-Centric Push BFS
│   ├── bfs_pull.cu                # 代码题B: Vertex-Centric Pull BFS
│   ├── bfs_edge_centric.cu        # 代码题C: Edge-Centric BFS
│   ├── bfs_frontier.cu            # 代码题D: Frontier-based BFS（atomicCAS）
│   └── bfs_frontier_private.cu    # 代码题E: Frontier-based BFS + 共享内存私有化
└── run_tests.py                    # 一键判题
```

## 使用方法

```bash
cd chapter-15/exercises
python run_tests.py
```
