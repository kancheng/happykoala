import torch
import torch.optim as optim

class WPOptim(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, alpha=0.05, **kwargs):
        defaults = dict(alpha=alpha, **kwargs)
        super(WPOptim, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def generate_delta(self):
        device = self.param_groups[0]["params"][0].device
        grads = [p.grad.float() for group in self.param_groups for p in group["params"] if p.grad is not None and p.grad.numel() > 0]
        
        if not grads:  # 如果没有梯度，直接返回
            return

        grad_norm = torch.norm(
            torch.stack([
                grad.norm(p=2).to(device)
                for grad in grads
            ]), p=2
        )

        if grad_norm == 0:
            return  # 如果梯度范数为0，直接返回

        for group in self.param_groups:
            scale = group["alpha"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None or p.grad.numel() == 0:
                    continue
                delta = p.grad.float() * scale.to(p)
                p.add_(delta)
                self.state[p]["delta"] = delta

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.generate_delta()  # 确保每次 step 时都生成 delta

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "delta" not in self.state[p]:
                    continue  # 如果没有 delta，跳过这个参数
                p.sub_(self.state[p]["delta"])
        
        self.base_optimizer.step()

        return loss