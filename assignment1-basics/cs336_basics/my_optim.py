from collections.abc import Callable, Iterable
from typing import Optional, Tuple
import torch,math

class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"invalid lr: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 0)
                print(t)
                p.data -= lr / math.sqrt(t+1) * p.grad.data
                state['t'] = t+1
        return loss
                

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, 
                lr: float = 1e-3,
                *,
                weight_decay: float = 0.01,
                betas: Tuple[float] = (0.9, 0.999),
                eps: float = 1e-8):
        if lr < 0:
            raise ValueError(f"invalid lr: {lr}")

        defaults = {"lr": lr,
                    "weight_decay": weight_decay,
                    "betas": betas,
                    "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']        
            betas, weight_decay, eps = self.defaults['betas'], self.defaults['weight_decay'], self.defaults['eps']
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if not state:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    state['t'] = 1
                t = state['t']
                p.sub_(p, alpha = lr * weight_decay)
                lr *= math.sqrt(1-betas[1]**t)/(1-betas[0]**t)
                state['m'].mul_(betas[0]).add_(p.grad, alpha=1 - betas[0])
                state['v'].mul_(betas[1]).addcmul_(p.grad, p.grad, value = 1 - betas[1])
                p.addcdiv_(state['m'], torch.sqrt(state['v'])+eps, value = -lr)
                state['t'] += 1
        return loss

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def learning_rate_schedule(t: int,
                           alpha_max: float,
                           alpha_min: float,
                           t_w: int,
                           t_c: int):
    if t < t_w:
        return alpha_max * t / t_w
    elif t_w <= t <= t_c:
        return alpha_min + (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w))) * (alpha_max - alpha_min) / 2
    else:
        return alpha_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    norm_sum = torch.sqrt(sum([torch.sum(torch.square(p.grad)) for p in parameters if p.grad is not None]))
    if norm_sum > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(max_l2_norm / (norm_sum + eps))

if __name__ == "__main__":
    weights = torch.nn.Parameter(5*torch.randn((10,10)))
    for i in range(4):
        lr = 10**i
        opt = SGD([weights], lr=lr)
        for i in range(10):
            loss = (weights ** 2).mean()
            # print(f'current lr is {lr}: {loss.cpu().item()}')
            loss.backward()
            opt.step()
