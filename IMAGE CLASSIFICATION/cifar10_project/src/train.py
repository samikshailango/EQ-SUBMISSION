import argparse
import copy
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TrainConfig
from data import get_dataloaders
from models import build_model
from utils import (EarlyStopper, count_parameters, get_device,
                    load_checkpoint, save_checkpoint, save_history_json, set_seed)

torch.backends.cudnn.enabled = False

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train a CNN on CIFAR-10")
    p.add_argument("--model", type=str, default="baseline", choices=["baseline", "resnet"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "sgd", "adamw"])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=["cosine", "step", "none"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-batches", type=int, default=0,
                    help="If >0, cap batches/epoch (debug/smoke-test only).")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint (_last.pt or _best.pt) to resume training from.")
    p.add_argument("--amp", action="store_true",
                    help="Enable mixed-precision training (torch.amp autocast + GradScaler). "
                         "Speeds up training and reduces memory on CUDA GPUs with Tensor Cores; "
                         "has no effect (safely no-ops) on CPU.")
    a = p.parse_args()

    cfg = TrainConfig(
        model_name=a.model, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        weight_decay=a.weight_decay, optimizer=a.optimizer, dropout=a.dropout,
        lr_schedule=a.lr_schedule, grad_clip=a.grad_clip, patience=a.patience,
        early_stopping=not a.no_early_stopping, augment=not a.no_augment,
        seed=a.seed, num_workers=a.num_workers, device=a.device, amp=a.amp,
    )
    cfg._max_batches = a.max_batches  # debug knob, not part of the dataclass schema
    cfg._resume = a.resume
    return cfg


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay, nesterov=True)
    raise ValueError(cfg.optimizer)


def build_scheduler(cfg: TrainConfig, optimizer: torch.optim.Optimizer):
    if cfg.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.lr_schedule == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    return None


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device, max_batches: int = 0,
             amp: bool = False):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    amp_enabled = amp and device.type == "cuda"
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += x.size(0)
        if max_batches and i + 1 >= max_batches:
            break
    return total_loss / total_n, total_correct / total_n


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip, log_every,
                     epoch, max_batches: int = 0, amp: bool = False, scaler=None):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    t0 = time.time()
    amp_enabled = amp and device.type == "cuda"
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        if amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += x.size(0)

        if log_every and (i + 1) % log_every == 0:
            print(f"  epoch {epoch} batch {i+1}/{len(loader)} "
                  f"loss={total_loss/total_n:.4f} acc={total_correct/total_n:.4f}")

        if max_batches and i + 1 >= max_batches:
            break

    elapsed = time.time() - t0
    return total_loss / total_n, total_correct / total_n, elapsed


def run_training(cfg: TrainConfig, verbose: bool = True) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    if verbose:
        print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    model = build_model(cfg.model_name, num_classes=cfg.num_classes, dropout=cfg.dropout).to(device)
    if verbose:
        print(f"Model: {cfg.model_name} | trainable params: {count_parameters(model):,}")

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopper(patience=cfg.patience) if cfg.early_stopping else None

    amp_enabled = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    if cfg.amp and device.type != "cuda":
        if verbose:
            print("Note: --amp was requested but no CUDA device is available; "
                  "continuing in full precision (AMP only helps on CUDA GPUs).")
    elif amp_enabled and verbose:
        print("Mixed-precision training (AMP) enabled.")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = -1.0
    best_state = None
    max_batches = getattr(cfg, "_max_batches", 0)
    start_epoch = 1

    run_name = cfg.run_name()
    ckpt_best_path = os.path.join(cfg.checkpoint_dir, f"{run_name}_best.pt")
    ckpt_last_path = os.path.join(cfg.checkpoint_dir, f"{run_name}_last.pt")
    history_path = os.path.join(cfg.output_dir, f"{run_name}_history.json")

    resume_path = getattr(cfg, "_resume", None)
    if resume_path:
        ckpt = load_checkpoint(resume_path, model, optimizer, scheduler, map_location=str(device), scaler=scaler)
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", -1.0)
        if cfg.lr_schedule == "cosine" and scheduler is not None:
            saved_t_max = getattr(scheduler, "T_max", None)
            if saved_t_max is not None and saved_t_max != cfg.epochs:
                print(f"WARNING: resuming with --epochs {cfg.epochs} but the cosine schedule "
                      f"was originally set up for {saved_t_max} epochs. The LR curve will not "
                      f"extend smoothly. Re-run with --epochs {saved_t_max} to match the "
                      f"original run, or use --lr-schedule step/none instead.")
        # Recover the exact history so curves/plots are continuous across the resume.
        if ckpt.get("history"):
            history = ckpt["history"]
        best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"Resumed from {resume_path}: starting at epoch {start_epoch}, "
                  f"best_val_acc so far = {best_val_acc:.4f}")
        if start_epoch > cfg.epochs:
            if verbose:
                print(f"Checkpoint epoch ({ckpt['epoch']}) >= requested --epochs ({cfg.epochs}); "
                      f"nothing to do. Pass a larger --epochs to keep training.")
        if stopper is not None and best_val_acc > -1.0:
            # Seed the stopper's "best score so far" so resuming doesn't reset the patience clock.
            stopper.best_score = best_val_acc

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss, train_acc, elapsed = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            cfg.grad_clip, cfg.log_every, epoch, max_batches=max_batches,
            amp=cfg.amp, scaler=scaler,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device,
                                      max_batches=max_batches, amp=cfg.amp)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if verbose:
            print(f"[{run_name}] epoch {epoch:03d}/{cfg.epochs} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"lr={current_lr:.6f} ({elapsed:.1f}s)")

        save_checkpoint(ckpt_last_path, model, optimizer, scheduler, epoch, best_val_acc, history, scaler=scaler)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(ckpt_best_path, model, optimizer, scheduler, epoch, best_val_acc, history, scaler=scaler)

        save_history_json(history, history_path)

        if stopper is not None and stopper.step(val_acc):
            if verbose:
                print(f"Early stopping triggered at epoch {epoch} "
                      f"(no improvement for {cfg.patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device, amp=cfg.amp)
    if verbose:
        print(f"[{run_name}] FINAL test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    history["best_val_acc"] = best_val_acc
    save_history_json(history, history_path)

    return {
        "history": history,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "best_val_acc": best_val_acc,
        "ckpt_best_path": ckpt_best_path,
        "ckpt_last_path": ckpt_last_path,
        "history_path": history_path,
        "run_name": run_name,
    }


if __name__ == "__main__":
    cfg = parse_args()
    run_training(cfg)
