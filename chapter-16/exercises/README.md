# 第十六章 练习

## 结构

```
exercises/
├── kernels/
│   ├── conv2d_forward.cu            # 代码题A: Conv2D 前向传播内核
│   ├── conv2d_backward_input.cu     # 代码题B: Conv2D 反向传播 - 输入梯度
│   ├── conv2d_backward_weights.cu   # 代码题C: Conv2D 反向传播 - 权重梯度
│   ├── maxpool2d_forward.cu         # 代码题D: MaxPool2D 前向传播内核
│   └── maxpool2d_backward.cu        # 代码题E: MaxPool2D 反向传播内核
└── run_tests.py                     # 一键判题
```

## 使用方法

```bash
cd chapter-16/exercises
python run_tests.py
```
