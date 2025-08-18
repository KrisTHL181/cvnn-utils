import torch

__version__ = "0.1.1"


class ComplexModule(torch.nn.Module):
    """
    所有复数神经网络模块的基类。

    强调：输入输出均为 torch.complex 类型，
    操作应尽量保持复线性或相位等变性。

    设计哲学：
        ℂ 是一等公民，不是 ℝ × ℝ 的语法糖。
    """


class ComplexIsNotRsqWarning(Warning):
    """
    警告：复数 ℂ 不是实数对 ℝ²。

    虽然 ℂ 与 ℝ² 同胚，但其代数结构（乘法、共轭、解析性）完全不同。
    避免将复数视为两个独立实数通道处理，除非明确知道后果。

    特别地：
    - 不要随意拆分 .real 和 .imag
    - 不要对相位直接池化或归一化
    - 注意操作是否满足复线性或相位等变性

    推荐使用：
    - z.mean(dim=...) 替代 F.adaptive_avg_pool2d(z)
    - 模长激活（ModReLU）而非 ReLU(z.real)
    """
