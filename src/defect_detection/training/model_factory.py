"""timm backbone factory and freeze helpers for two-stage fine-tuning."""

import timm
from torch import nn

from defect_detection.config import ModelConfig
from defect_detection.core.constants import CLASS_NAMES


def create_model(cfg: ModelConfig) -> nn.Module:
    """ImageNet-pretrained backbone with a fresh 2-class head (logits for normal, defective)."""
    model: nn.Module = timm.create_model(
        cfg.backbone,
        pretrained=cfg.pretrained,
        num_classes=len(CLASS_NAMES),
        drop_rate=cfg.drop_rate,
    )
    return model


def classifier(model: nn.Module) -> nn.Module:
    """The final classification layer (timm exposes it uniformly via ``get_classifier``)."""
    head = model.get_classifier()  # type: ignore[operator]
    if not isinstance(head, nn.Module):
        raise TypeError(f"{type(model).__name__} has no classifier module")
    return head


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    """Freeze/unfreeze every parameter except the classifier's."""
    head_params = {id(p) for p in classifier(model).parameters()}
    for param in model.parameters():
        param.requires_grad = not frozen or id(param) in head_params


def train_mode(model: nn.Module, backbone_frozen: bool) -> None:
    """Put the model in training mode for the current stage.

    With a frozen backbone the whole network stays in eval mode except the head, so
    BatchNorm keeps its ImageNet running statistics instead of drifting on a few batches.
    """
    if backbone_frozen:
        model.eval()
        classifier(model).train()
    else:
        model.train()


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parameters the optimiser should update in the current stage."""
    return [p for p in model.parameters() if p.requires_grad]
