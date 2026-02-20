import torch
from jaxtyping import Float, Int
from torch import Tensor
import math
from collections.abc import Iterable
import numpy.typing as npt
import os
from typing import IO, BinaryIO

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    (batch_size, vocab_size) = inputs.shape
    max_logits = inputs.max(dim=-1, keepdim=True)[0]
    norm_inputs = inputs - max_logits
    molecule = norm_inputs[torch.arange(batch_size), targets]
    denominator = torch.log(torch.exp(norm_inputs).sum(dim=-1, keepdim=False))
    return (-molecule + denominator).mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, defaults={
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
        })

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get('m', torch.zeros_like(p))
                v = state.get('v', torch.zeros_like(p))
                t = state.get('t', 0)
                grad = p.grad

                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad ** 2
                alpha = lr * math.sqrt(1 - betas[1] ** (t + 1)) / (1 - betas[0] ** (t + 1))
                p.data = p.data - alpha * m / (torch.sqrt(v) + eps)
                p.data = p.data * (1 - lr * weight_decay)

                state['m'] = m
                state['v'] = v
                state['t'] = t + 1

def learning_rate_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it <= cosine_cycle_iters:
        return min_learning_rate + (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) / 2
    return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    total_norm_sq = sum(p.grad.data.pow(2).sum().item() for p in parameters if p.grad is not None)
    total_norm = total_norm_sq ** 0.5
    if total_norm <= max_l2_norm:
        return
    scale = max_l2_norm / (total_norm + eps)
    for p in parameters:
        if p.grad is None:
            continue
        p.grad.data.mul_(scale)

def data_loader(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = torch.from_numpy(dataset).to(device)
    sample_indices = torch.randperm(len(dataset) - context_length, device=device)[:batch_size]
    offset = torch.arange(context_length, device=device)
    idx_left = sample_indices.unsqueeze(1) + offset
    idx_right = sample_indices.unsqueeze(1) + offset + 1
    left = dataset[idx_left]
    right = dataset[idx_right]
    return (left, right)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
) -> None:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']

if __name__ == "__main__":
    pass
