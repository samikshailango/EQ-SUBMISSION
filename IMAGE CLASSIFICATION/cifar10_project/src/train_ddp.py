import argparse
import copy
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from config import TrainConfig
from data import get_dataloaders_ddp
from models import build_model
from train import build_optimizer, build_scheduler
from utils import count_parameters, save_checkpoint, save_history_json, set_seed


def setup_distributed():

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, local_rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_one_epoch_ddp(model, loader, sampler, optimizer, criterion, device,
                         grad_clip, log_every, epoch, rank, max_batches: int = 0):
    model.train()
    sampler.set_epoch(epoch)  # different shuffle order each epoch across all ranks
    total_loss, total_correct, total_n = 0.0, 0, 0
    t0 = time.time()

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()  # DDP all-reduces gradients across ranks here automatically
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += x.size(0)

        if rank == 0 and log_every and (i + 1) % log_every == 0:
            print(f"  epoch {epoch} batch {i+1}/{len(loader)} "
                  f"local_loss={total_loss/total_n:.4f} local_acc={total_correct/total_n:.4f}")

        if max_batches and i + 1 >= max_batches:
            break

    elapsed = time.time() - t0
    return total_loss, total_correct, total_n, elapsed


@torch.no_grad()
def evaluate_ddp(model, loader, criterion, device, max_batches: int = 0):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += x.size(0)
        if max_batches and i + 1 >= max_batches:
            break
    return total_loss, total_correct, total_n


def all_reduce_stats(loss_sum, correct_sum, n_sum, device):
    stats = torch.tensor([loss_sum, correct_sum, n_sum], device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    g_loss_sum, g_correct_sum, g_n_sum = stats.tolist()
    return g_loss_sum / g_n_sum, g_correct_sum / g_n_sum


def parse_args():
    p = argparse.ArgumentParser(description="Distributed (multi-GPU) CIFAR-10 training via DDP")
    p.add_argument("--model", type=str, default="baseline", choices=["baseline", "resnet"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128, help="Per-GPU batch size.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "sgd", "adamw"])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=["cosine", "step", "none"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    is_main = rank == 0

    cfg = TrainConfig(
        model_name=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, optimizer=args.optimizer, dropout=args.dropout,
        lr_schedule=args.lr_schedule, grad_clip=args.grad_clip, seed=args.seed,
        num_workers=args.num_workers, early_stopping=False,
    )

    set_seed(cfg.seed + rank)  
    if is_main:
        print(f"World size: {world_size} | backend: {'nccl' if torch.cuda.is_available() else 'gloo'}")

    train_loader, val_loader, test_loader, train_sampler = get_dataloaders_ddp(cfg, rank, world_size)
    dist.barrier() 

    model = build_model(cfg.model_name, num_classes=cfg.num_classes, dropout=cfg.dropout).to(device)
    ddp_kwargs = dict(device_ids=[local_rank]) if torch.cuda.is_available() else {}
    model = DDP(model, **ddp_kwargs)

    if is_main:
        print(f"Model: {cfg.model_name} | trainable params: {count_parameters(model.module):,}")

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = -1.0
    best_state = None
    run_name = cfg.run_name() + "_ddp"
    max_batches = args.max_batches

    for epoch in range(1, cfg.epochs + 1):
        loss_sum, correct_sum, n_sum, elapsed = train_one_epoch_ddp(
            model, train_loader, train_sampler, optimizer, criterion, device,
            cfg.grad_clip, cfg.log_every, epoch, rank, max_batches=max_batches,
        )
        train_loss, train_acc = all_reduce_stats(loss_sum, correct_sum, n_sum, device)

        val_loss_sum, val_correct_sum, val_n_sum = evaluate_ddp(
            model, val_loader, criterion, device, max_batches=max_batches)
        val_loss, val_acc = all_reduce_stats(val_loss_sum, val_correct_sum, val_n_sum, device)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        if is_main:
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)
            print(f"[{run_name}] epoch {epoch:03d}/{cfg.epochs} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"lr={current_lr:.6f} ({elapsed:.1f}s, {world_size} ranks)")

            ckpt_last = os.path.join(cfg.checkpoint_dir, f"{run_name}_last.pt")
            save_checkpoint(ckpt_last, model.module, optimizer, scheduler, epoch, best_val_acc, history)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.module.state_dict())
                ckpt_best = os.path.join(cfg.checkpoint_dir, f"{run_name}_best.pt")
                save_checkpoint(ckpt_best, model.module, optimizer, scheduler, epoch, best_val_acc, history)

            save_history_json(history, os.path.join(cfg.output_dir, f"{run_name}_history.json"))

        dist.barrier()

    if is_main and best_state is not None:
        model.module.load_state_dict(best_state)

    test_loss_sum, test_correct_sum, test_n_sum = evaluate_ddp(model, test_loader, criterion, device)
    test_loss, test_acc = all_reduce_stats(test_loss_sum, test_correct_sum, test_n_sum, device)

    if is_main:
        print(f"[{run_name}] FINAL test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
