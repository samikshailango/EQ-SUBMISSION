import os
from dataclasses import dataclass, field
from typing import Literal


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class TrainConfig:

    seed: int = 42

    data_dir: str = DATA_DIR
    batch_size: int = 128
    num_workers: int = 2
    val_split: float = 0.1          
    augment: bool = True


    model_name: Literal["baseline", "resnet"] = "baseline"
    dropout: float = 0.3
    num_classes: int = 10

    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 5e-4
    optimizer: Literal["adam", "sgd", "adamw"] = "adamw"
    momentum: float = 0.9           
    lr_schedule: Literal["cosine", "step", "none"] = "cosine"
    step_size: int = 10           
    gamma: float = 0.5             
    grad_clip: float = 1.0       
    amp: bool = False               

    # ---- early stopping ----
    early_stopping: bool = True
    patience: int = 7

    # ---- misc ----
    device: str = "cuda"           
    checkpoint_dir: str = CHECKPOINT_DIR
    output_dir: str = OUTPUT_DIR
    log_every: int = 100           

    def run_name(self) -> str:
        return f"{self.model_name}_bs{self.batch_size}_lr{self.lr}_opt{self.optimizer}"
