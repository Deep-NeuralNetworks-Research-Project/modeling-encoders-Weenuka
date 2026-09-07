# modeling-encoders-Weenuka

**P3 — Modeling A** (encoders, fusion, decoders) for *Pair-Order Consistent & Uncertainty-Aware Semantic Change Detection* — University of Moratuwa, CSE semester project. See the parent team repo for the full project (proposal, research dossier, other roles). This repo holds only the component set owned by P3 per `CODEOWNERS`:

```
src/cdlib/models/encoders/    src/cdlib/models/fusion/
src/cdlib/models/decoders/    src/cdlib/models/baselines/siamese_resnet18.py
```

It follows the frozen interfaces and registry pattern from the shared `CLAUDE.md`: encoders return a list of multi-scale features, fusion takes two feature lists and returns one, decoders take the fused list and return `[B,1,H,W]` logits. `models/build.py` in this repo is a **starter stub only** (see its docstring) — the real Hydra-driven `build_model(cfg)` is shared infra owned by P2 and will replace it when the team's main repo scaffold lands.

## What's implemented

- **`encoders/resnet.py`** — Siamese ResNet-18 (torchvision, ImageNet weights), taps C2–C5 (strides 4/8/16/32, channels 64/128/256/512). `use_c5` and `dilate_last` are config-exposed for the stride-32 ablation (`dilate_last` does a manual DeepLab-style atrous conversion of `layer4`, since torchvision's `replace_stride_with_dilation` doesn't support `BasicBlock` — see the docstring).
- **`encoders/efficientnet.py`** — EfficientNet-B0/B2 via `timm`, `features_only=True`, channels/strides read from `feature_info` at runtime (never hardcoded).
- Both encoders share a `forward_pair(img1, img2)` method implementing the Siamese-BN pattern: `[I1;I2]` concatenated along the batch dim, one forward pass, then split — not two sequential forwards.
- **`fusion/absdiff.py`** — `|a−b|`, order-invariant, FC-Siam-Diff-style.
- **`fusion/signed_fusion.py`** — the project's cheapest novelty claim. Configurable across all four modes P4's ablation B6 needs: `signed`, `signed_product`, `concat`, `signed_concat`. Docstring spells out the swap-symmetry argument (`h(b−a) = h(a−b)` only if `h` is even) that motivates the pair-order consistency loss.
- **`decoders/unet_decoder.py`** — top-down U-Net/FPN-style decoder over N skip levels, coarsest→finest, upsample+concat+conv at each step.
- **`baselines/siamese_resnet18.py`** — composes encoder + abs-diff fusion + decoder into the mandatory "stronger baseline", satisfying the frozen model `forward` contract (`logits`/`confidence`/`aux`).
- **`tests/test_shapes.py`** — parametrized over every `ENCODER_REGISTRY` / `FUSION_REGISTRY` key; also checks the stride-32 dilation path, abs-diff's order-invariance, signed-diff's swap-antisymmetry, and the baseline's forward contract + gradient flow.
- **`scripts/benchmark_encoders.py`** — standalone step-time benchmark (CUDA events, warm-up excluded, median + IQR, peak memory) for the Week 3 "which encoder do we default to" measurement. Meant to be superseded by/merged with P5's `cdlib.cli.benchmark` once that exists.

## Not yet implemented / explicitly out of scope here

- Datasets, trainer/engine, CLI, Hydra configs, CI, alignment/heads/proposed.py, losses, metrics — owned by P1/P2/P4/P5 respectively; this repo intentionally doesn't invent them.
- `models/build.py` here is a **throwaway stub** wiring only `siamese_resnet18`, just enough to make this component set runnable standalone. It is not meant to be merged as-is.

## Setup

```bash
pip install -e .        # installs the cdlib package (torch, torchvision, timm)
pip install -e .[dev]    # + pytest
pytest tests/test_shapes.py -v
python scripts/benchmark_encoders.py --batch-size 8 --img-size 256   # run on the actual Colab/Kaggle GPU
```

## Checklist status (mirrors `CLAUDE.local.md`)

- [x] Week 2 — `encoders/resnet.py`, `encoders/efficientnet.py`, registered, shape-tested
- [x] Week 3 — `fusion/absdiff.py`, `fusion/signed_fusion.py` (all 4 modes), registered, shape-tested
- [x] Week 3–4 — `decoders/unet_decoder.py`, `baselines/siamese_resnet18.py`
- [ ] Week 3 — encoder speed measurement run on an actual Colab/Kaggle GPU (script ready, needs a GPU session + team report-out)
- [ ] Week 3–4 — LR sweep + ≥3-seed tuning of the ResNet-18 baseline (needs P1's dataloaders + P2's trainer)
- [ ] Weeks 5+ — wire into `proposed.py` with P4; stride-32 ablation writeup; Experiments/ablations paper section
