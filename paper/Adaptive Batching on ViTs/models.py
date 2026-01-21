import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from peft import LoraConfig, get_peft_model


def build_lora_vit(num_classes: int, config: Dict[str, Any]) -> torch.nn.Module:
    try:
        import fastervit
    except ImportError as exc:  # pragma: no cover - dependency is optional at import time
        raise ImportError("Install `fastervit` to build the ViT backbone.") from exc

    torch.serialization.add_safe_globals([argparse.Namespace])

    model_name = config.get("model_name", "faster_vit_0_224")
    pretrained = bool(config.get("pretrained", True))
    default_ckpt = Path(__file__).parent / "models" / "faster_vit_0.pth.tar"
    model_path = config.get("model_path") or default_ckpt
    ckpt_path = Path(model_path).expanduser()
    if pretrained and not ckpt_path.is_file():
        raise FileNotFoundError(
            f"FasterViT checkpoint not found at {ckpt_path}. "
            "Download the pretrained weight and place it there, or set `pretrained=False`."
        )
    lora_r = int(config.get("lora_r", 8))
    lora_alpha = int(config.get("lora_alpha", 16))
    lora_dropout = float(config.get("lora_dropout", 0.1))
    lora_targets = config.get("lora_target_modules", ["qkv"])
    lora_bias = config.get("lora_bias", "none")

    model = fastervit.create_model(
        model_name,
        pretrained=pretrained,
        model_path=str(ckpt_path) if pretrained else None,
    )

    if isinstance(lora_targets, str):
        lora_targets_list = [lora_targets]
    elif isinstance(lora_targets, Iterable):
        lora_targets_list = list(lora_targets)
    else:
        lora_targets_list = [str(lora_targets)]

    lora_conf = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_targets_list,
        lora_dropout=lora_dropout,
        bias=lora_bias,
    )
    peft_model = get_peft_model(model, lora_conf)
    head_in_features = peft_model.base_model.model.head.in_features
    peft_model.base_model.model.head = torch.nn.Linear(head_in_features, num_classes)
    peft_model.base_model.model.head.requires_grad_(True)
    return peft_model
