from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    (batch_size, _) = inputs.shape
    max_logits = inputs.max(dim=-1, keepdim=True)[0]
    norm_inputs = inputs - max_logits
    numerator = norm_inputs[torch.arange(batch_size), targets]
    denominator = torch.log(torch.exp(norm_inputs).sum(dim=-1, keepdim=False))
    return (-numerator + denominator).mean()


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, defaults={"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t = state.get("t", 0)
                grad = p.grad

                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad**2
                alpha = lr * math.sqrt(1 - betas[1] ** (t + 1)) / (1 - betas[0] ** (t + 1))
                p.data = p.data - alpha * m / (torch.sqrt(v) + eps)
                p.data = p.data * (1 - lr * weight_decay)

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1


def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it <= cosine_cycle_iters:
        return min_learning_rate + (max_learning_rate - min_learning_rate) * (
            1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        ) / 2
    return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    total_norm_sq = sum(p.grad.data.pow(2).sum().item() for p in parameters if p.grad is not None)
    total_norm = total_norm_sq**0.5
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
    return left, right


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


@dataclass
class TrainingConfig:
    batch_size: int
    context_length: int
    max_iters: int
    max_learning_rate: float
    min_learning_rate: float
    warmup_iters: int
    cosine_cycle_iters: int
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    eval_interval: int = 100
    eval_batches: int = 10
    checkpoint_interval: int = 0
    checkpoint_path: str | None = None


def _lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    eval_batches: int,
    device: str,
) -> float:
    model_was_training = model.training
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = data_loader(dataset, batch_size, context_length, device)
        logits = model(x)
        losses.append(_lm_loss(logits, y).item())
    if model_was_training:
        model.train()
    return float(np.mean(losses))


def train_language_model(
    model: torch.nn.Module,
    train_dataset: npt.NDArray,
    val_dataset: npt.NDArray | None,
    config: TrainingConfig,
    device: str,
    *,
    on_log: Callable[[dict], None] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    start_iteration: int = 0,
) -> tuple[list[dict], torch.optim.Optimizer]:
    model.to(device)
    model.train()

    if optimizer is None:
        optimizer = AdamW(
            model.parameters(),
            lr=config.max_learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    history: list[dict] = []

    for it in range(start_iteration, config.max_iters):
        lr = learning_rate_schedule(
            it,
            config.max_learning_rate,
            config.min_learning_rate,
            config.warmup_iters,
            config.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = data_loader(train_dataset, config.batch_size, config.context_length, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = _lm_loss(logits, y)
        loss.backward()
        gradient_clipping(model.parameters(), config.grad_clip)
        optimizer.step()

        should_eval = config.eval_interval > 0 and ((it + 1) % config.eval_interval == 0 or it == start_iteration)
        if should_eval:
            rec = {"iter": it + 1, "lr": lr, "train_loss": float(loss.item())}
            if val_dataset is not None:
                rec["val_loss"] = estimate_loss(
                    model,
                    val_dataset,
                    config.batch_size,
                    config.context_length,
                    config.eval_batches,
                    device,
                )
            history.append(rec)
            if on_log is not None:
                on_log(rec)

        if (
            config.checkpoint_interval > 0
            and config.checkpoint_path is not None
            and (it + 1) % config.checkpoint_interval == 0
        ):
            save_checkpoint(model, optimizer, it + 1, config.checkpoint_path)

    return history, optimizer


if __name__ == "__main__":
    pass
