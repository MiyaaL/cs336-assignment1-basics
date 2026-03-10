from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cs336_basics.lmtrain import AdamW, TrainingConfig, load_checkpoint, save_checkpoint, train_language_model
from cs336_basics.transfomer import TransformerLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal training script for CS336 Assignment 1 model")
    p.add_argument("--train-data", type=Path, required=True, help="Path to train token ids (.npy, 1D int array)")
    p.add_argument("--val-data", type=Path, default=None, help="Path to val token ids (.npy, 1D int array)")
    p.add_argument("--out-dir", type=Path, default=Path("runs/default"), help="Output directory")

    # model
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10000.0)

    # training
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--max-lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--cosine-cycle-iters", type=int, default=2000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--checkpoint-interval", type=int, default=500)
    p.add_argument("--resume", type=Path, default=None, help="Optional checkpoint path to resume from")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_dataset(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D token-id array at {path}, got shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Expected integer dtype at {path}, got {arr.dtype}")
    return arr.astype(np.int64, copy=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ids = _load_dataset(args.train_data)
    val_ids = _load_dataset(args.val_data) if args.val_data is not None else None

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume is not None:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"[resume] loaded checkpoint: {args.resume}, start_iteration={start_iter}")

    ckpt_path = args.out_dir / "checkpoint.pt"
    cfg = TrainingConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        max_iters=args.max_iters,
        max_learning_rate=args.max_lr,
        min_learning_rate=args.min_lr,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=str(ckpt_path),
    )

    def on_log(record: dict) -> None:
        msg = f"iter={record['iter']} lr={record['lr']:.3e} train_loss={record['train_loss']:.4f}"
        if "val_loss" in record:
            msg += f" val_loss={record['val_loss']:.4f}"
        print(msg, flush=True)

    history, optimizer = train_language_model(
        model=model,
        train_dataset=train_ids,
        val_dataset=val_ids,
        config=cfg,
        device=args.device,
        on_log=on_log,
        optimizer=optimizer,
        start_iteration=start_iter,
    )

    save_checkpoint(model, optimizer, args.max_iters, args.out_dir / "final.pt")
    with open(args.out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(args.out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)

    print(f"done. artifacts saved to {args.out_dir}")


if __name__ == "__main__":
    main()
