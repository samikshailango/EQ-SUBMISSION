import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from config import CIFAR10_CLASSES, TrainConfig
from data import denormalize, get_dataloaders
from models import build_model
from utils import get_device, load_checkpoint, set_seed


def plot_curves(history: dict, out_path: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved curves -> {out_path}")


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_images, all_true, all_pred, all_conf = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(1)
        all_images.append(x.cpu())
        all_true.append(y)
        all_pred.append(pred.cpu())
        all_conf.append(conf.cpu())
    return (torch.cat(all_images), torch.cat(all_true),
            torch.cat(all_pred), torch.cat(all_conf))


def plot_confusion_matrix(y_true, y_pred, out_path: str) -> None:
    n = len(CIFAR10_CLASSES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {out_path}")


def plot_misclassified(images, y_true, y_pred, y_conf, out_path: str, n_show: int = 16) -> None:
    wrong_idx = (y_true != y_pred).nonzero(as_tuple=True)[0]
    if len(wrong_idx) == 0:
        print("No misclassified images found (perfect accuracy?) - skipping plot.")
        return

    order = torch.argsort(y_conf[wrong_idx], descending=True)
    chosen = wrong_idx[order[:n_show]]

    n_cols = 4
    n_rows = int(np.ceil(len(chosen) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, chosen):
        img = denormalize(images[idx]).permute(1, 2, 0).numpy()
        ax.imshow(img)
        true_name = CIFAR10_CLASSES[y_true[idx]]
        pred_name = CIFAR10_CLASSES[y_pred[idx]]
        conf = y_conf[idx].item()
        ax.set_title(f"true: {true_name}\npred: {pred_name} ({conf:.2f})", fontsize=9)
        ax.axis("off")

    for ax in axes[len(chosen):]:
        ax.axis("off")

    fig.suptitle("Sample misclassified test images (highest-confidence mistakes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved misclassified samples -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained CIFAR-10 model")
    p.add_argument("--model", type=str, default="baseline", choices=["baseline", "resnet"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--history", type=str, default=None,
                    help="Path to history JSON; defaults next to the checkpoint's run name.")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    cfg = TrainConfig(model_name=args.model, batch_size=args.batch_size,
                       dropout=args.dropout, device=args.device, augment=False)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    os.makedirs(cfg.output_dir, exist_ok=True)

    model = build_model(cfg.model_name, num_classes=cfg.num_classes, dropout=cfg.dropout).to(device)
    ckpt = load_checkpoint(args.checkpoint, model, map_location=str(device))
    print(f"Loaded checkpoint from {args.checkpoint} (epoch {ckpt.get('epoch')})")

    _, _, test_loader = get_dataloaders(cfg)
    criterion = nn.CrossEntropyLoss()

    images, y_true, y_pred, y_conf = collect_predictions(model, test_loader, device)
    test_acc = (y_true == y_pred).float().mean().item()
    test_loss = criterion(model(images[:512].to(device)), y_true[:512].to(device)).item()
    print(f"Test accuracy: {test_acc:.4f}")

    run_name = f"{cfg.model_name}"
    history = ckpt.get("history")
    if history is None and args.history:
        with open(args.history) as f:
            history = json.load(f)

    if history:
        plot_curves(history, os.path.join(cfg.output_dir, f"{run_name}_curves.png"))

    plot_confusion_matrix(y_true, y_pred, os.path.join(cfg.output_dir, f"{run_name}_confusion.png"))
    plot_misclassified(images, y_true, y_pred, y_conf,
                        os.path.join(cfg.output_dir, f"{run_name}_misclassified.png"))


if __name__ == "__main__":
    main()
