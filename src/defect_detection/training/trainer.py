"""Two-stage fine-tuning with early stopping on validation PR-AUC.

Stage 1 ("head"): backbone frozen, only the new classifier learns (higher LR). A randomly
initialised head would otherwise send large, noisy gradients into the pretrained features.
Stage 2 ("finetune"): all layers unfrozen at a low LR with a cosine schedule.

Model selection uses validation PR-AUC (threshold-free, focused on the defective class), with
validation loss as the tie-breaker because PR-AUC can saturate at 1.0 on an easy dataset.
The test split is never touched here.
"""

import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader

from defect_detection.config import ModelConfig, PreprocessConfig, ProjectConfig
from defect_detection.core.constants import DEFECTIVE_LABEL
from defect_detection.core.preprocessing import PreprocessSpec
from defect_detection.data.augment import TrainTransform
from defect_detection.data.dataset import SplitDataset
from defect_detection.provenance import provenance
from defect_detection.training.evaluate import classification_metrics
from defect_detection.training.model_factory import (
    create_model,
    set_backbone_frozen,
    train_mode,
    trainable_parameters,
)

log = logging.getLogger(__name__)
CHECKPOINT_NAME = "best.pt"


@dataclass(frozen=True)
class EpochRecord:
    """One row of the training history."""

    stage: str
    epoch: int
    lr: float
    train_loss: float
    val_loss: float
    val_pr_auc: float
    val_roc_auc: float
    val_f1: float
    seconds: float


@dataclass(frozen=True)
class Stage:
    """A training stage: how long, how fast, and whether the backbone is frozen."""

    name: str
    epochs: int
    lr: float
    backbone_frozen: bool


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch, and request deterministic kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for CUDA determinism
    random.seed(seed)
    np.random.seed(seed)  # legacy global RNG, still used by some libraries
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init: derive python/numpy seeds from the worker's torch seed."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def resolve_device(name: str) -> torch.device:
    """``auto`` picks CUDA when available."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def class_weights(counts: npt.NDArray[np.int64], mode: str) -> torch.Tensor:
    """Cross-entropy weights from TRAIN counts: ``n_total / (n_classes * n_c)`` for balanced."""
    if mode == "none":
        return torch.ones(len(counts), dtype=torch.float32)
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)


def save_checkpoint(path: Path, model: nn.Module, meta: dict[str, Any]) -> None:
    """Weights plus plain-JSON metadata (loadable with ``weights_only=True``)."""
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint without unpickling arbitrary objects."""
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint


@dataclass(frozen=True)
class TrainedModel:
    """A trained run's network plus the preprocessing it was trained with."""

    model: nn.Module
    spec: PreprocessSpec
    backbone: str
    checkpoint: dict[str, Any]


def load_trained_model(run_dir: Path, device: torch.device) -> TrainedModel:
    """Rebuild the network from ``best.pt`` in EVAL mode (BatchNorm uses its running stats).

    Architecture and preprocessing come from the checkpoint. Returning eval mode by default
    matters: a train-mode forward on one image normalises with that image's own statistics.
    """
    checkpoint = load_checkpoint(run_dir / CHECKPOINT_NAME)
    trained = checkpoint["config"]
    model_cfg = ModelConfig.model_validate({**trained["model"], "pretrained": False})
    pre = PreprocessConfig.model_validate(trained["preprocess"])
    model = create_model(model_cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    spec = PreprocessSpec(pre.image_size, pre.mean, pre.std)
    return TrainedModel(model, spec, model_cfg.backbone, checkpoint)


def split_logits(
    config: ProjectConfig, trained: TrainedModel, split: str, device: torch.device
) -> tuple[list[str], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """(rel_paths, labels, logits) for one split, in deterministic order, no augmentation."""
    dataset = SplitDataset(
        config.data.processed_dir / "splits.csv", config.data.raw_dir, split, trained.spec
    )
    loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        dataset, batch_size=config.train.batch_size * 2, num_workers=config.train.num_workers
    )
    labels, logits = collect_logits(trained.model, loader, device)
    return dataset.rel_paths, labels, logits


@torch.no_grad()
def collect_logits(
    model: nn.Module, loader: DataLoader[tuple[torch.Tensor, int]], device: torch.device
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Labels and raw logits (n, 2) over a loader, in eval mode."""
    model.eval()
    labels, logits = [], []
    for images, targets in loader:
        logits.append(model(images.to(device)).float().cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(labels).astype(np.int64), np.concatenate(logits).astype(np.float64)


def predict(
    model: nn.Module, loader: DataLoader[tuple[torch.Tensor, int]], device: torch.device
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], float]:
    """Labels, P(defective) and mean (unweighted) cross-entropy over a loader."""
    labels, logits = collect_logits(model, loader, device)
    log_probs = torch.log_softmax(torch.from_numpy(logits), dim=1)
    loss = float(nn.functional.nll_loss(log_probs, torch.from_numpy(labels)).item())
    probs = log_probs.exp()[:, DEFECTIVE_LABEL].numpy().astype(np.float64)
    return labels, probs, loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    backbone_frozen: bool,
) -> float:
    """One pass over the training data; returns the mean (class-weighted) loss."""
    use_amp = scaler.is_enabled()
    train_mode(model, backbone_frozen)
    total, count = 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=use_amp):
            loss = criterion(model(images), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item() * len(targets)
        count += len(targets)
    return total / count


def plot_history(history: pd.DataFrame) -> Figure:
    """Loss and validation AUC curves over global epochs, with the stage boundary marked."""
    x = np.arange(1, len(history) + 1)
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(x, history["train_loss"], marker="o", label="train (weighted, augmented)")
    ax_loss.plot(x, history["val_loss"], marker="o", label="val")
    ax_loss.set_ylabel("cross-entropy")
    ax_auc.plot(x, history["val_pr_auc"], marker="o", label="val PR-AUC")
    ax_auc.plot(x, history["val_roc_auc"], marker="o", label="val ROC-AUC")
    ax_auc.set_ylabel("AUC")
    boundary = int((history["stage"] == "head").sum())
    for ax in (ax_loss, ax_auc):
        if 0 < boundary < len(history):
            ax.axvline(boundary + 0.5, color="grey", linestyle="--", label="unfreeze")
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    return fig


def train(config: ProjectConfig, run_dir: Path) -> dict[str, Any]:
    """Run both stages, keep the best checkpoint, and write history/curves/metrics to run_dir."""
    seed_everything(config.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.train.device)
    tc = config.train
    spec = PreprocessSpec(
        config.preprocess.image_size, config.preprocess.mean, config.preprocess.std
    )
    splits_csv, raw = config.data.processed_dir / "splits.csv", config.data.raw_dir

    train_ds = SplitDataset(splits_csv, raw, "train", spec, TrainTransform(spec, config.augment))
    train_eval_ds = SplitDataset(splits_csv, raw, "train", spec)  # no aug: overfitting check
    val_ds = SplitDataset(splits_csv, raw, "val", spec)
    loader_args: dict[str, Any] = {
        "num_workers": tc.num_workers,
        "worker_init_fn": seed_worker,
        "persistent_workers": tc.num_workers > 0,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=tc.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        **loader_args,
    )
    val_loader = DataLoader(val_ds, batch_size=tc.batch_size * 2, **loader_args)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=tc.batch_size * 2, **loader_args)

    model = create_model(config.model).to(device)
    weights = class_weights(train_ds.class_counts(), tc.class_weighting)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    log.info(
        "device=%s train=%d val=%d class_counts=%s weights=%s",
        device,
        len(train_ds),
        len(val_ds),
        train_ds.class_counts().tolist(),
        weights.tolist(),
    )

    stages = [
        Stage("head", tc.head_epochs, tc.head_lr, backbone_frozen=True),
        Stage("finetune", tc.finetune_epochs, tc.finetune_lr, backbone_frozen=False),
    ]
    meta_base = {
        **provenance(config),
        "backbone": config.model.backbone,
        "config": config.model_dump(mode="json"),
    }
    history: list[EpochRecord] = []
    best_score: tuple[float, float] | None = None
    best_info: dict[str, Any] = {}
    for stage in (s for s in stages if s.epochs > 0):
        set_backbone_frozen(model, stage.backbone_frozen)
        optimizer = torch.optim.AdamW(
            trainable_parameters(model), lr=stage.lr, weight_decay=tc.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage.epochs * len(train_loader)
        )
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        epochs_without_improvement = 0
        for epoch in range(1, stage.epochs + 1):
            start = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                scaler,
                device,
                stage.backbone_frozen,
            )
            labels, probs, val_loss = predict(model, val_loader, device)
            metrics = classification_metrics(labels, probs)
            record = EpochRecord(
                stage.name,
                epoch,
                lr,
                train_loss,
                val_loss,
                metrics["pr_auc"],
                metrics["roc_auc"],
                metrics["f1"],
                time.perf_counter() - start,
            )
            history.append(record)
            score = (metrics["pr_auc"], -val_loss)
            improved = best_score is None or score > best_score
            if improved:
                best_score, epochs_without_improvement = score, 0
                best_info = {"stage": stage.name, "epoch": epoch, "val_metrics": metrics}
                save_checkpoint(run_dir / CHECKPOINT_NAME, model, {**meta_base, **best_info})
            else:
                epochs_without_improvement += 1
            log.info(
                "%s %d/%d train_loss=%.4f val_loss=%.4f val_pr_auc=%.4f val_roc_auc=%.4f "
                "val_f1=%.4f %.0fs%s",
                stage.name,
                epoch,
                stage.epochs,
                train_loss,
                val_loss,
                metrics["pr_auc"],
                metrics["roc_auc"],
                metrics["f1"],
                record.seconds,
                " *best*" if improved else "",
            )
            if (
                not stage.backbone_frozen
                and epochs_without_improvement >= tc.early_stopping_patience
            ):
                log.info(
                    "early stopping: no val improvement for %d epochs", epochs_without_improvement
                )
                break

    # Re-evaluate the selected checkpoint on val and on un-augmented train (overfitting check).
    model.load_state_dict(load_checkpoint(run_dir / CHECKPOINT_NAME)["state_dict"])
    val_labels, val_probs, val_loss = predict(model, val_loader, device)
    tr_labels, tr_probs, tr_loss = predict(model, train_eval_loader, device)
    summary: dict[str, Any] = {
        **provenance(config),
        "run_dir": run_dir.as_posix(),
        "device": str(device),
        "best": {"stage": best_info["stage"], "epoch": best_info["epoch"]},
        "epochs_run": len(history),
        "class_weights": weights.tolist(),
        "val": {**classification_metrics(val_labels, val_probs), "loss": val_loss},
        "train_no_aug": {**classification_metrics(tr_labels, tr_probs), "loss": tr_loss},
    }
    summary["generalization_gap"] = {
        "loss": summary["val"]["loss"] - summary["train_no_aug"]["loss"],
        "pr_auc": summary["train_no_aug"]["pr_auc"] - summary["val"]["pr_auc"],
    }

    frame = pd.DataFrame([asdict(r) for r in history])
    frame.to_csv(run_dir / "history.csv", index=False)
    fig = plot_history(frame)
    fig.savefig(run_dir / "curves.png", dpi=120)
    plt.close(fig)
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
