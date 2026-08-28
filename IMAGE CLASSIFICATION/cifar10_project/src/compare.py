import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn as nn

from config import TrainConfig
from data import get_dataloaders
from evaluate import collect_predictions
from models import build_model
from utils import get_device, load_checkpoint, set_seed


def default_ckpt_path(cfg: TrainConfig, which: str = "best") -> str:
    return os.path.join(cfg.checkpoint_dir, f"{cfg.run_name()}_{which}.pt")


def default_history_path(cfg: TrainConfig) -> str:
    return os.path.join(cfg.output_dir, f"{cfg.run_name()}_history.json")


def load_history(ckpt: dict, cfg: TrainConfig) -> dict:
    hist_path = default_history_path(cfg)
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            return json.load(f)
    return ckpt.get("history", {})


def evaluate_checkpoint(model_name: str, ckpt_path: str, cfg_template: TrainConfig):
    cfg = TrainConfig(model_name=model_name, batch_size=cfg_template.batch_size,
                       dropout=cfg_template.dropout, device=cfg_template.device,
                       seed=cfg_template.seed, augment=False)
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. Train it first with:\n"
            f"  python train.py --model {model_name} --epochs <N>\n"
            f"...or pass --train-if-missing to this script to train it now."
        )

    model = build_model(cfg.model_name, num_classes=cfg.num_classes, dropout=cfg.dropout).to(device)
    ckpt = load_checkpoint(ckpt_path, model, map_location=str(device))

    _, _, test_loader = get_dataloaders(cfg)
    criterion = nn.CrossEntropyLoss()
    images, y_true, y_pred, y_conf = collect_predictions(model, test_loader, device)
    test_acc = (y_true == y_pred).float().mean().item()
    test_loss = criterion(model(images[:512].to(device)), y_true[:512].to(device)).item()

    history = load_history(ckpt, cfg)
    best_val_acc = history.get("best_val_acc", max(history.get("val_acc", [0.0])))

    return {
        "test_acc": test_acc,
        "test_loss": test_loss,
        "best_val_acc": best_val_acc,
        "history": history,
        "epoch": ckpt.get("epoch"),
        "ckpt_path": ckpt_path,
    }


def main():
    p = argparse.ArgumentParser(description="Compare baseline vs residual CNN on CIFAR-10")
    p.add_argument("--baseline-ckpt", type=str, default=None,
                    help="Path to baseline checkpoint. Defaults to the standard "
                         "train.py naming convention if omitted.")
    p.add_argument("--resnet-ckpt", type=str, default=None,
                    help="Path to resnet checkpoint. Defaults to the standard "
                         "train.py naming convention if omitted.")
    p.add_argument("--ckpt-kind", type=str, default="best", choices=["best", "last"],
                    help="Which checkpoint variant to use when a path isn't given explicitly.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--train-if-missing", action="store_true",
                    help="Train any model whose checkpoint isn't found, instead of erroring.")
    p.add_argument("--epochs", type=int, default=30,
                    help="Only used if --train-if-missing triggers a training run.")
    args = p.parse_args()

    cfg_template = TrainConfig(batch_size=args.batch_size, dropout=args.dropout,
                                seed=args.seed, device=args.device)

    ckpt_paths = {
        "baseline": args.baseline_ckpt or default_ckpt_path(
            TrainConfig(model_name="baseline", batch_size=args.batch_size), args.ckpt_kind),
        "resnet": args.resnet_ckpt or default_ckpt_path(
            TrainConfig(model_name="resnet", batch_size=args.batch_size), args.ckpt_kind),
    }

    results = {}
    for model_name, ckpt_path in ckpt_paths.items():
        if not os.path.exists(ckpt_path) and args.train_if_missing:
            from train import run_training
            print(f"\nNo checkpoint at {ckpt_path} — training {model_name} for {args.epochs} epochs...")
            cfg = TrainConfig(model_name=model_name, epochs=args.epochs,
                               batch_size=args.batch_size, dropout=args.dropout,
                               seed=args.seed, device=args.device)
            train_result = run_training(cfg, verbose=True)
            ckpt_path = train_result["ckpt_best_path"]

        print(f"Loading {model_name} from {ckpt_path} ...")
        results[model_name] = evaluate_checkpoint(model_name, ckpt_path, cfg_template)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':10s} {'Epoch':>6s} {'Best Val Acc':>14s} {'Test Acc':>10s} {'Test Loss':>10s}")
    for name, r in results.items():
        print(f"{name:10s} {str(r['epoch']):>6s} {r['best_val_acc']:>14.4f} "
              f"{r['test_acc']:>10.4f} {r['test_loss']:>10.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted_any = False
    for name, r in results.items():
        val_acc = r["history"].get("val_acc")
        if val_acc:
            ax.plot(val_acc, label=f"{name} (val)")
            plotted_any = True
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Baseline CNN vs Residual CNN")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(cfg_template.output_dir, "baseline_vs_resnet.png")
    if plotted_any:
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved comparison plot -> {out_path}")
    else:
        print("\nNo per-epoch history found for either model, skipping comparison plot "
              "(test-accuracy numbers above are still valid).")


if __name__ == "__main__":
    main()
