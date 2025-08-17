import torch


class WirtingerAdamW(torch.optim.Optimizer):
    """
    基于 Wirtinger 微积分的 AdamW 优化器
    更新方向：z ← z - η * ∂L/∂z*
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        wirtinger_real_scaling=True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            wirtinger_real_scaling=wirtinger_real_scaling,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | torch.Tensor | None:  # type: ignore
        loss = closure() if closure else None

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            lr = group["lr"]
            eps = group["eps"]
            apply_wirtinger_scale = group["wirtinger_real_scaling"]

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad
                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    # 根据参数类型初始化方差
                    if p.is_complex():
                        state["exp_avg_sq"] = torch.zeros_like(grad.real)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # 更新动量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 统一计算梯度模长平方
                grad_sq = grad.abs().square()

                exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=1 - beta2)

                # 偏差校正
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                denom = exp_avg_sq_corrected.sqrt() + eps

                # 对复数参数，PyTorch自动微分常返回`2 * ∂L/∂z̄`，但这对于数学而言有点反直觉。
                # 因此乘0.5以符合标准Wirtinger更新规则。通过下面的代码来检查：
                # >>> z = torch.tensor(1.0 + 2j, requires_grad=True)
                # >>> loss = z.real.pow(2) + z.imag.pow(2)
                # >>> loss.backward()
                # >>> print("grad = ", z.grad)
                # grad =  tensor(2.+4.j)
                # 2+4j非预期，我们希望这是1+2j。这意味着他们是这样计算的：grad = ∂L/∂x + i ∂L/∂y = 2 * (∂L/∂z*)
                # 请阅读文档：autograd#autograd-for-complex-numbers
                # 他们没有提到这里有2的系数，但实际代码计算证明这确实存在。
                # PyTorch似乎将复数认为成了两个实数分别求导并将其组合，但我们更希望复数优先。
                # 因此添加一个可选的0.5修正。
                if p.is_complex() and apply_wirtinger_scale:
                    exp_avg_corrected *= 0.5

                # 权重衰减 (复数安全)
                if weight_decay != 0.0:
                    p.data.mul_(1 - lr * weight_decay)

                # 应用更新
                update = exp_avg_corrected / denom
                p.data.sub_(update, alpha=lr)

        return loss
