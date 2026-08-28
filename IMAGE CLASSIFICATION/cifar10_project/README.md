# CIFAR-10 Image Classification

---

## 1. Project Structure

```
cifar10_project/
├── src/
│   ├── config.py       # All hyperparameters & paths (single source of truth)
│   ├── data.py         # CIFAR-10 loading, transforms/augmentation, DataLoaders (+ DDP variant)
│   ├── models.py        # BaselineCNN + ResNetCNN (residual blocks) built from scratch
│   ├── train.py         # Training loop: loss, accuracy, LR schedule, grad clip, AMP,
│   │                     #   checkpointing, early stopping, resume
│   ├── train_ddp.py       # Optional Extension: multi-GPU training via DistributedDataParallel
│   ├── evaluate.py       # Test accuracy, loss/accuracy curves, confusion matrix,
│   │                     #   misclassified-image grid
│   ├── compare.py        # Loads baseline + resnet checkpoints and plots a comparison
│   └── export_onnx.py     # Optional Extension: ONNX export + ONNX Runtime inference
├── deploy/
│   └── app.py             # Optional Extension: FastAPI serving app (/health, /predict)
├── tests/
│   ├── conftest.py
│   ├── test_models.py    # Unit tests: forward shapes, gradients, residual wiring
│   ├── test_utils.py     # Unit tests: seeding, early stopping, checkpoint I/O
│   └── test_extensions.py # Unit tests: ONNX export/inference, AMP config
├── notebooks/
│   └── cifar10_walkthrough.ipynb   # Same pipeline, notebook form, with visuals
├── checkpoints/          # Saved *_best.pt / *_last.pt model weights (created at runtime)
├── outputs/               # Plots + history JSON + exported .onnx (created at runtime)
├── data/                  # CIFAR-10 raw files (auto-downloaded here, created at runtime)
├── requirements.txt
└── README.md
```

## 2. Setup

Requires Python 3.9+.

```bash
cd cifar10_project
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`torch`/`torchvision` will pull CPU wheels by default; for GPU training, install
the CUDA build for your system from https://pytorch.org/get-started/locally/
**before** `pip install -r requirements.txt` (or just re-install torch after).

## 3. How to Run

All commands are run from the `src/` directory (or add `src/` to `PYTHONPATH`).

### Train the baseline CNN
```bash
cd src
python train.py --model baseline --epochs 30
```

### Train the residual CNN (Part 2 improvement)
```bash
python train.py --model resnet --epochs 30
```

CIFAR-10 (~170 MB) is downloaded automatically to `../data/` on first run.
Checkpoints go to `../checkpoints/<run_name>_best.pt` / `_last.pt`; per-epoch
history is saved to `../outputs/<run_name>_history.json`.

Useful flags (see `python train.py --help` for the full list):
```bash
python train.py --model resnet --epochs 40 --lr 5e-4 --optimizer sgd \
                 --batch-size 256 --dropout 0.4 --lr-schedule step \
                 --grad-clip 1.0 --patience 5
```

### Resume an interrupted run
A full checkpoint (model, optimizer, scheduler, and metric history) is saved to
`<run_name>_last.pt` after **every** epoch, so an interrupted run (crash,
closed laptop, etc.) never loses more than the current epoch's progress.

```bash
python train.py --model baseline --epochs 30 \
                 --resume ../checkpoints/baseline_bs128_lr0.001_optadamw_last.pt

### Evaluate a trained model
```bash
python evaluate.py --model baseline --checkpoint ../checkpoints/baseline_bs128_lr0.001_optadamw_best.pt
```
This prints test accuracy and writes to `../outputs/`:
- `<model>_curves.png` — train/val loss & accuracy curves
- `<model>_confusion.png` — confusion matrix
- `<model>_misclassified.png` — grid of the most confidently wrong test predictions

### Run the baseline-vs-residual comparison (Part 2 deliverable)
```bash
python compare.py --epochs 30
```
Trains both models under identical settings/seed and prints a summary table
plus `../outputs/baseline_vs_resnet.png` overlaying validation-accuracy curves.

### Run the notebook
```bash
jupyter notebook notebooks/cifar10_walkthrough.ipynb
```

### Run unit tests
```bash
cd cifar10_project    
pip install pytest
pytest tests/ -v
```
19 tests covering model forward/backward passes, the residual block's skip
connection, seeding determinism, early stopping, and checkpoint save/load
round-trips. All pass on CPU in a few seconds — no GPU or dataset download
required, since these test the model/utility code directly with random
tensors rather than real images.

---

## 4. Design Decisions

**Data pipeline** (`data.py`)
- Normalization uses CIFAR-10's real per-channel mean/std (not a generic
  0.5/0.5), which trains noticeably faster than naive [-1,1] scaling.
- Augmentation: random crop with reflect-padding, horizontal flip, mild
  color jitter, and `RandomErasing` — standard, cheap augmentations known to
  help CIFAR-10 generalization without needing heavier techniques (e.g.
  AutoAugment/CutMix) that overkill a small-CNN baseline.
- The 50k training images are split 90/10 into train/val (deterministic,
  seeded) so validation never touches the augmented-training transform —
  avoids a common leakage bug where the val split's `__getitem__` used the
  augmented dataset object.
- `DataLoader` uses `pin_memory` (when CUDA is available) and
  `persistent_workers` to avoid worker-respawn overhead per epoch.

**Models** (`models.py`)
- `BaselineCNN`: 3 stages of Conv-BN-ReLU ×2 + MaxPool (32→16→8→4 spatial,
  channels 32→64→128), global average pool, then a small FC head with
  dropout. Straightforward, easy to reason about, ~323K params.
- `ResNetCNN` (Part 2 / Option A): same channel/stage budget, but each stage
  is a `ResidualBlock` (Conv-BN-ReLU-Conv-BN + skip, with a 1×1 projection
  shortcut whenever channels or stride change) instead of plain conv+pool.
  Downsampling happens via stride-2 in the first block of each stage
  (matching modern ResNet convention) rather than MaxPool. This is the
  required from-scratch "simplified ResNet block" — not `torchvision.models.resnet18`.
- Both use `AdaptiveAvgPool2d(1)` before the classifier head instead of
  `Flatten` on a large feature map, which keeps the FC layer small and
  makes the model input-size-agnostic.

**Training** (`train.py`)
- Loss: `CrossEntropyLoss`. Metric: top-1 accuracy, computed per-batch and
  averaged over the epoch (not just the last batch).
- LR scheduling: cosine annealing by default (`--lr-schedule cosine`), step
  decay also supported.
- Gradient clipping (`clip_grad_norm_`, default max-norm 1.0) — cheap
  insurance against occasional loss spikes, especially with SGD+momentum.
- Early stopping on validation accuracy (default patience=7) to avoid
  wasting epochs once the model plateaus.
- Checkpointing saves both the best-val-accuracy model and the most recent
  epoch (so a crashed run can resume from `_last.pt`), plus optimizer/
  scheduler state and the full metric history in the same file.
- At the end of training, the best-val-accuracy weights (not the final
  epoch's) are reloaded before computing test accuracy — final-epoch
  weights can be worse than an earlier checkpoint due to overfitting or an
  unlucky LR-schedule tail.

**Reproducibility** (`utils.py`)
- `set_seed()` seeds `random`, `numpy`, and `torch` (CPU + all CUDA
  devices) and forces deterministic cuDNN kernels.
- The train/val split index permutation is generated from a seeded
  `torch.Generator`, so it's identical across runs with the same seed.

**Part 2 choice**: Option A (residual block) was chosen over Option B/C
because it's the most architecturally meaningful change to demonstrate
understanding of skip connections and gradient flow, and it composes
naturally with the other requirements (Option B's logging and Option C's
tuning are both still exercised implicitly — see gradient clipping, LR
scheduling, and the `--dropout`/`--lr`/`--optimizer`/`--batch-size` CLI
flags in `train.py`, which support ad-hoc hyperparameter sweeps).

## 5. Optional Extensions

All five optional extensions from the assignment are implemented.

### Mixed-precision training
```bash
python train.py --model resnet --epochs 30 --amp
```
Uses `torch.amp.autocast` + `GradScaler` — forward/backward passes run in
fp16 (or bf16 on supported hardware) where safe, with the loss scaled to
avoid gradient underflow. On a CUDA GPU this typically gives a meaningful
speedup and reduced memory use with no accuracy loss; on CPU it safely
no-ops (falls back to full precision with a printed note) since CPU AMP
isn't a Tensor Core operation. Works with `--resume` — the `GradScaler`'s
state is checkpointed alongside the model/optimizer/scheduler.

### Distributed training (multi-GPU via DDP)
```bash
# 2 GPUs on one machine:
torchrun --standalone --nproc_per_node=2 train_ddp.py --model resnet --epochs 30

# all available GPUs:
torchrun --standalone --nproc_per_node=gpu train_ddp.py --model baseline --epochs 30
```
`train_ddp.py` implements `DistributedDataParallel`: each rank gets its own
GPU and a disjoint shard of the training data via `DistributedSampler`
(re-shuffled every epoch via `set_epoch`), gradients are automatically
all-reduced by DDP on every `backward()`, and validation/test metrics are
manually all-reduced across ranks (`all_reduce_stats`) so the reported
loss/accuracy reflect the whole dataset, not just one rank's shard.
Checkpointing and logging happen only on rank 0 to avoid file clobbering.
Verified against PyTorch's `gloo` CPU backend with 2 simulated ranks (real
multi-GPU runs use `nccl` automatically when CUDA is available — see
`setup_distributed()`). TPU support was out of scope: it requires
`torch_xla` and a TPU runtime that can't be exercised or verified here.

### Deploy with FastAPI
```bash
cd deploy
export MODEL_NAME=baseline
export CHECKPOINT_PATH=../checkpoints/baseline_bs128_lr0.001_optadamw_best.pt
uvicorn app:app --host 0.0.0.0 --port 8000
```
Then:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@some_image.jpg"
```
Interactive docs at `http://localhost:8000/docs`. The app loads the
checkpoint once at startup, resizes/normalizes any uploaded image to the
CIFAR-10 input format, and returns the predicted class plus the full
probability distribution over all 10 classes. Returns HTTP 400 for
unreadable images and fails startup with a clear error if `CHECKPOINT_PATH`
is missing.

### Convert to ONNX and run inference
```bash
# Export
python export_onnx.py --model baseline \
    --checkpoint ../checkpoints/baseline_bs128_lr0.001_optadamw_best.pt \
    --output ../outputs/baseline.onnx --verify

# Run inference on any image file
python export_onnx.py --infer ../outputs/baseline.onnx --image path/to/image.png
```
`--verify` checks the exported graph is well-formed (`onnx.checker`) and
diffs its output against the original PyTorch model on random input
(max abs diff is ~1e-8 in practice — effectively identical). The exported
graph supports variable batch sizes via `dynamic_axes`. Inference uses
`onnxruntime`'s CPU execution provider, so it runs anywhere without a GPU
or even PyTorch installed.

### Unit tests for model components
Already covered in Section 3 (`pytest tests/ -v`) — 28 tests total, including
9 specifically for the extensions (`tests/test_extensions.py`): ONNX export
correctness/validity, variable-batch-size support, output-matches-PyTorch
verification, end-to-end image-to-prediction inference, and AMP config
plumbing (the `GradScaler` is provably inert on CPU, so `--amp` can never
silently change results where it can't help).

## 6. Notes / Known Limitations
- No pretrained weights are used anywhere, per the assignment's "from
  scratch" requirement.
- `--device cuda` falls back to CPU automatically if no GPU is available
  (see `utils.get_device`), so all commands above work either way; CPU
  training of 30 epochs takes on the order of 30–60+ minutes depending on
  hardware, GPU training a few minutes.
- The DDP script was validated for correctness on CPU with 2 simulated
  ranks (gradient sync, sampler sharding, cross-rank metric aggregation all
  confirmed working) but was not run on actual multi-GPU hardware, since
  none is available in this environment — the `nccl`/`device_ids` codepath
  is standard PyTorch DDP usage and should work as-is on a real multi-GPU
  machine, but hasn't been empirically verified there.
- `torch.onnx.export(..., dynamo=False)` is used deliberately (the newer
  dynamo-based exporter needs `onnxscript` and is less stable for this
  simple CNN graph); expect a harmless `DeprecationWarning` about the
  legacy exporter.
