"""Shared helpers for the binary coffee-bean classification pipeline.

Used by `02_train_models.ipynb` and `03_evaluate_models.ipynb`. Importable
either directly (when the file lives next to the notebook) or by uploading
this file to the Colab session and `%run utils.py`.

References:
- Master Coding Prompt: ../00_MASTER_CODING_PROMPT.md
- Methodology: ../Latex-TA-IF-ITERA/.../chapters/chapter-3.tex
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no GUI pop-ups)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
    mobilenet_v3_large,
    resnet50,
    swin_t,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
NUM_CLASSES = 2  # binary: defect (0), non_defect (1) (folder-alphabetic order)
IMG_SIZE = 224  # ImageNet-pretrained backbone input
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ModelName = Literal["resnet50", "mobilenet_v3_large", "swin_t"]
MODEL_NAMES: tuple[ModelName, ...] = ("mobilenet_v3_large", "resnet50", "swin_t")


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for deterministic runs.

    Trades off a small amount of training speed for full reproducibility,
    which is required by the thesis-defense narrative.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Transforms (Master Prompt Section 3)
# -----------------------------------------------------------------------------
def get_transforms(train: bool) -> transforms.Compose:
    """ImageNet-style transforms.

    Training augmentation is applied ONLY to the training subset to prevent
    overfitting and is intentionally conservative -- beans are roughly
    rotation/flip invariant, but we avoid extreme color jitter that would
    obscure the visual cues of defect classes (color, holes, cracks).
    """
    base = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return transforms.Compose(base)

    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            ),
            *base,
        ]
    )


class TransformSubset(Dataset):
    """Wrap a `torch.utils.data.Subset` with its own transform.

    `random_split` produces `Subset` instances that share the parent dataset's
    transform; we need different transforms per split (augment training only).
    """

    def __init__(self, subset: Subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        img, label = self.subset[idx]
        # `ImageFolder` returns a PIL.Image when transform=None on the parent.
        return self.transform(img), label


# -----------------------------------------------------------------------------
# Deterministic Split Handling
# -----------------------------------------------------------------------------
_SPLIT_REQUIRED_KEYS = frozenset(
    {"seed", "class_to_idx", "samples", "train_indices", "val_indices", "test_indices"}
)


def load_or_create_split(
    data_root: Path,
    report_dir: Path,
    seed: int = 42,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict:
    """Load existing split_indices.json or generate and persist a new one.

    Returns dict with keys: seed, class_to_idx, samples,
    train_indices, val_indices, test_indices, full_dataset.

    Raises:
        ValueError: if existing file is malformed or dataset has changed.
    """
    split_path = Path(report_dir) / "split_indices.json"
    full_dataset = datasets.ImageFolder(root=str(data_root), transform=None)

    if split_path.exists():
        # --- Load and validate existing split ---
        try:
            raw = split_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"split_indices.json is malformed: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "split_indices.json is malformed: root element is not a JSON object."
            )

        missing_keys = _SPLIT_REQUIRED_KEYS - set(payload.keys())
        if missing_keys:
            raise ValueError(
                f"split_indices.json is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        # Compare basenames against current dataset
        stored_basenames = [
            os.path.basename(s) for s in payload["samples"]
        ]
        current_basenames = [
            os.path.basename(s) for s, _ in full_dataset.samples
        ]

        if stored_basenames != current_basenames:
            # Identify specific differences for a helpful error message
            stored_set = set(stored_basenames)
            current_set = set(current_basenames)
            added = current_set - stored_set
            removed = stored_set - current_set
            parts = []
            if added:
                parts.append(f"added={sorted(list(added)[:5])}")
            if removed:
                parts.append(f"removed={sorted(list(removed)[:5])}")
            if not added and not removed:
                parts.append("order changed")
            raise ValueError(
                f"Dataset has changed since split_indices.json was created. "
                f"Mismatch: {', '.join(parts)}. "
                f"Delete reports/split_indices.json and re-run to regenerate."
            )

        return {
            "seed": payload["seed"],
            "class_to_idx": payload["class_to_idx"],
            "samples": payload["samples"],
            "train_indices": payload["train_indices"],
            "val_indices": payload["val_indices"],
            "test_indices": payload["test_indices"],
            "full_dataset": full_dataset,
        }

    # --- Generate new split ---
    n_total = len(full_dataset)
    n_train = int(round(split_ratios[0] * n_total))
    n_val = int(round(split_ratios[1] * n_total))
    n_test = n_total - n_train - n_val

    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [n_train, n_val, n_test], generator=gen
    )

    payload = {
        "seed": seed,
        "class_to_idx": full_dataset.class_to_idx,
        "samples": [s for s, _ in full_dataset.samples],
        "train_indices": train_subset.indices,
        "val_indices": val_subset.indices,
        "test_indices": test_subset.indices,
    }

    # Persist to disk
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        **payload,
        "full_dataset": full_dataset,
    }


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------
def build_model(name: ModelName, num_classes: int = NUM_CLASSES,
                pretrained: bool = True) -> nn.Module:
    """Construct a backbone with its 1000-class head replaced for binary output.

    All three architectures are loaded with their best official ImageNet
    weights (Master Coding Prompt Section3 architecture list).
    """
    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        # classifier = Sequential(Linear, Hardswish, Dropout, Linear)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif name == "swin_t":
        # Swin-T was released with a single official weight set (V1).
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown model name: {name!r}. Expected one of {MODEL_NAMES}."
        )

    return model


def count_trainable_params(model: nn.Module) -> int:
    """Total number of parameters with `requires_grad=True`."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    """Total parameter count (frozen + trainable)."""
    return sum(p.numel() for p in model.parameters())


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
    train: bool,
) -> tuple[float, float]:
    """Run one training or validation epoch, return (avg_loss, accuracy)."""
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


# -----------------------------------------------------------------------------
# Training orchestration
# -----------------------------------------------------------------------------
def train_one_model(
    model_name: ModelName,
    train_loader: DataLoader,
    val_loader: DataLoader,
    full_dataset: datasets.ImageFolder,
    hparams: dict,
    device: torch.device,
    use_amp: bool,
    ckpt_dir: Path | str,
    report_dir: Path | str,
) -> dict:
    """Train a single model, save best checkpoint, return result dict.

    Parameters
    ----------
    model_name : str
        One of MODEL_NAMES (e.g. "mobilenet_v3_large").
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    full_dataset : ImageFolder
        The full dataset (used for class_to_idx in checkpoint).
    hparams : dict
        Hyperparameters dict with keys: seed, learning_rate, weight_decay,
        epochs, patience.
    device : torch.device
        Device to train on.
    use_amp : bool
        Whether to use automatic mixed precision.
    ckpt_dir : Path or str
        Directory to write the checkpoint file.
    report_dir : Path or str
        Directory to write history CSV, curves PNG, and timing JSON.

    Returns
    -------
    dict
        Keys: model_name, best_epoch, best_val_acc, train_secs, history_df.
    """
    import copy
    import time
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    ckpt_dir = Path(ckpt_dir)
    report_dir = Path(report_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING: {model_name}")
    print(f"{'=' * 60}")

    # Seed before model construction (Req 3.6)
    seed_everything(hparams["seed"])

    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=hparams["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(
        f"  Params: total={count_total_params(model):,} | "
        f"trainable={count_trainable_params(model):,}"
    )

    history: list[dict] = []
    best_val_acc, best_epoch = -1.0, -1
    best_state = None
    epochs_no_improve = 0

    t0 = time.time()
    for epoch in range(1, hparams["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, True
        )
        vl_loss, vl_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, use_amp, False
        )
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": vl_loss,
                "val_acc": vl_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        # Strict improvement: val_acc must STRICTLY exceed previous best
        improved = vl_acc > best_val_acc
        flag = "*" if improved else " "
        print(
            f"  E{epoch:02d}{flag} | "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
            f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if improved:
            best_val_acc, best_epoch = vl_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= hparams["patience"]:
                print(
                    f"  Early stop at epoch {epoch} "
                    f"(no val_acc gain for {hparams['patience']} epochs)."
                )
                break

    train_secs = time.time() - t0

    # --- Save checkpoint (only on successful completion) ---
    ckpt_path = ckpt_dir / f"best_model_{model_name}.pth"
    torch.save(
        {
            "model_name": model_name,
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "hparams": hparams,
            "class_to_idx": full_dataset.class_to_idx,
        },
        ckpt_path,
    )
    print(
        f"  Saved best checkpoint -> {ckpt_path}  "
        f"(epoch {best_epoch}, val_acc={best_val_acc:.4f})"
    )

    # --- Save per-epoch history CSV ---
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(report_dir / f"history_{model_name}.csv", index=False)

    # --- Save per-model curves PNG ---
    curves_path = report_dir / f"curves_{model_name}.png"
    save_per_model_curves(hist_df, curves_path)
    print(f"  Saved -> {curves_path}")

    # --- Save timing JSON ---
    timing_data = {
        "model_name": model_name,
        "train_secs": train_secs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
    }
    timing_path = report_dir / f"timing_{model_name}.json"
    timing_path.write_text(json.dumps(timing_data, indent=2), encoding="utf-8")
    print(f"  Saved -> {timing_path}")

    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "train_secs": train_secs,
        "history_df": hist_df,
    }


# -----------------------------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------------------------
def save_per_model_curves(history_df: pd.DataFrame, save_path: Path | str) -> None:
    """Save a two-panel training curves figure for a single model.

    Left panel: train and val loss vs epoch.
    Right panel: train and val accuracy vs epoch.

    Parameters
    ----------
    history_df : pd.DataFrame
        Must contain columns: epoch, train_loss, train_acc, val_loss, val_acc, lr.
    save_path : Path or str
        Destination file path for the PNG output.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left panel -- Loss
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right panel -- Accuracy
    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="train")
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="val")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Visualisation -- comparison curves
# -----------------------------------------------------------------------------
def save_comparison_curves(results: dict[str, dict], report_dir) -> None:
    """Overlay all models on a single 2-panel chart (val loss & val accuracy).

    Parameters
    ----------
    results : dict[str, dict]
        Mapping of model_name -> result dict. Each result dict must contain
        a ``"history_df"`` key whose value is a DataFrame with columns
        ``epoch``, ``val_loss``, and ``val_acc``.
    report_dir : Path-like
        Directory where ``curves_comparison.png`` will be saved.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from pathlib import Path

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name in MODEL_NAMES:
        if name not in results:
            continue
        df = results[name]["history_df"]
        axes[0].plot(df["epoch"], df["val_loss"], label=name)
        axes[1].plot(df["epoch"], df["val_acc"], label=name)

    axes[0].set(title="Validation Loss -- comparison", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Validation Accuracy -- comparison", xlabel="epoch", ylabel="accuracy")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()

    out_png = report_dir / "curves_comparison.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_png}")


__all__ = [
    "NUM_CLASSES",
    "IMG_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "MODEL_NAMES",
    "ModelName",
    "seed_everything",
    "get_transforms",
    "TransformSubset",
    "load_or_create_split",
    "build_model",
    "count_trainable_params",
    "count_total_params",
    "run_epoch",
    "train_one_model",
    "save_per_model_curves",
    "save_comparison_curves",
]
