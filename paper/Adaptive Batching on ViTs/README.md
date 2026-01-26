# Adaptive Batching on ViTs

LoRA-tuned ViT baseline with adaptive batch sizing. The code mirrors the flow from `Achieving Linear Rate With Adaptive Batching`, but swaps the CNN for a FasterViT backbone wrapped with LoRA adapters.

## How to run
- Install deps (including `fastervit` and `peft`): `uv pip install -r pyproject.toml` or add them to your environment manually.
- FasterViT expects a local checkpoint when `pretrained=true`. Place it at `paper/Adaptive Batching on ViTs/models/faster_vit_0.pth.tar` (default in configs) or set `model_path` to your own location before running. The `models/` folder is git-ignored.
- From `paper/Adaptive Batching on ViTs`: `python main.py --config food101_adaptive.json`
- Example configs are provided for `food101` and `tiny_imagenet` (adaptive on by default). Toggle `adaptive_batch` to `false` for a fixed-batch baseline.
- Adaptive strategies:
  - `variance_ratio` (default): same variance/parameter-movement heuristic as the CNN version.
  - `seesaw`: cosine-LR crossings at `lr0/alpha^k`; on each crossing multiply batch size by `alpha` (clamped by `adaptive_batch_max`) and scale LR by `1/alpha` for SGD or `1/sqrt(alpha)` for Adam/AdamW. Config knobs: `adaptive_batch_strategy: "seesaw"`, `seesaw_alpha` (default 2.0). Uses shuffled loader and disables the scheduler (simulated cosine used only to locate cuts).

## Notes
- Food101 downloads via `torchvision`. Tiny ImageNet is fetched from the Stanford URL when `download=true` and keeps the original val split as the test set; train is further split into train/val.
- The LoRA setup targets the `qkv` projections by default and swaps the classifier head to match `num_classes`.
- Adaptive batching follows the same variance ratio heuristic as the CNN version, starting after `epoch_start_ab`.
