# modeling-encoders-Weenuka

Weenuka Rajapaksha's work on *Pair-Order Consistent & Uncertainty-Aware Semantic Change Detection* — University of Moratuwa, CSE semester project. Covers two phases under two different role-numbering schemes the team used at different points:

- **Phase 1 — P3 "Modeling A"** (encoders, fusion, decoders — `team-claude/roles/P3-modeling-encoders.md`)
- **Phase 2 — Member 5 "Uncertainty, calibration, selective prediction, directional head"** (`docs/member-tracks/05_rajapaksha_uncertainty_and_directional.md`)

See the parent team repo (`change-detect`, merging this repo with `baseline-infa-bula` and `metrics-integration-kusal`) for the full project — proposal, research dossier, other members' tracks. This repo holds my own contribution plus the minimum vendored dependencies needed to run it standalone (see "Vendored vs. authored here" below).

It follows the frozen interfaces and registry pattern from the shared `CLAUDE.md`: encoders return a list of multi-scale features, fusion takes two feature lists and returns one, decoders take the fused list and return `[B,1,H,W]` logits; metrics implement `reset()/update()/compute()`. `models/build.py` in this repo is a **starter stub only** (see its docstring) — the real Hydra-driven `build_model(cfg)` is shared infra and will be replaced when merged into the team's main repo scaffold.

## What's implemented

- **`encoders/resnet.py`** — Siamese ResNet-18 (torchvision, ImageNet weights), taps C2–C5 (strides 4/8/16/32, channels 64/128/256/512). `use_c5` and `dilate_last` are config-exposed for the stride-32 ablation (`dilate_last` does a manual DeepLab-style atrous conversion of `layer4`, since torchvision's `replace_stride_with_dilation` doesn't support `BasicBlock` — see the docstring).
- **`encoders/efficientnet.py`** — EfficientNet-B0/B2 via `timm`, `features_only=True`, channels/strides read from `feature_info` at runtime (never hardcoded).
- Both encoders share a `forward_pair(img1, img2)` method implementing the Siamese-BN pattern: `[I1;I2]` concatenated along the batch dim, one forward pass, then split — not two sequential forwards.
- **`fusion/absdiff.py`** — `|a−b|`, order-invariant, FC-Siam-Diff-style.
- **`fusion/signed_fusion.py`** — the project's cheapest novelty claim. Configurable across all four modes P4's ablation B6 needs: `signed`, `signed_product`, `concat`, `signed_concat`. Docstring spells out the swap-symmetry argument (`h(b−a) = h(a−b)` only if `h` is even) that motivates the pair-order consistency loss.
- **`decoders/unet_decoder.py`** — top-down U-Net/FPN-style decoder over N skip levels, coarsest→finest, upsample+concat+conv at each step.
- **`baselines/siamese_resnet18.py`** — composes encoder + abs-diff fusion + decoder into the mandatory "stronger baseline", satisfying the frozen model `forward` contract (`logits`/`confidence`/`aux`).
- **`tests/test_shapes.py`** — parametrized over every `ENCODER_REGISTRY` / `FUSION_REGISTRY` key; also checks the stride-32 dilation path, abs-diff's order-invariance, signed-diff's swap-antisymmetry, and the baseline's forward contract + gradient flow.
- **`scripts/benchmark_encoders.py`** — standalone step-time benchmark (CUDA events, warm-up excluded, median + IQR, peak memory, plus pair-input GFLOPs via `torch.utils.flop_counter`) for the Week 3 "which encoder do we default to" measurement. Meant to be superseded by/merged with P5's `cdlib.cli.benchmark` once that exists.
- **`tests/test_overfit_one_batch.py`** — per research/06's test strategy: the baseline must drive loss down sharply on a tiny fixed batch in <100 steps. First attempt used a pixel-random target and plateaued near BCE's ln(2) — not a bug (gradients checked healthy early on), just no learnable signal in noise-vs-noise; fixed to use a deterministic spatial target, which is what this kind of test is actually meant to catch shape/gradient bugs against. See the file's docstring.
- **`tests/test_registries.py`** — registry integrity: expected keys present, unknown-key errors are helpful, every encoder/fusion entry builds and runs a dummy forward (caught a real edge case: EfficientNet-B2 at 32×32 collapses to 1×1 spatial and BatchNorm can't compute train-mode stats from a single value — fixed by using `.eval()` for this particular smoke check).
- **`configs/model/`** — draft Hydra-shaped (`_target_`) configs for the baseline, both encoders, and all five fusion options, since P2's Hydra root doesn't exist yet to compose them into. `tests/test_config_validate.py` builds every one against the real constructors so they can't silently drift.
- **`scripts/stride32_ablation.py`** — the cost side of the stride-32 ablation (params/pair-GFLOPs/output-stride across `use_c5`/`dilate_last`), runnable without any training data. The quality side (Boundary IoU) is blocked on P1/P2/a GPU.
- **`paper/sections/experiments.md`** — draft of my Experiments/ablations writing assignment (per research/06 §8's section ownership): fixed methodology + tables for all four planned experiments, `TBD` placeholders for the numbers that need real training runs.

## Phase 2 — Member 5: uncertainty, calibration, directional equivariance

Built against the team's merged `change-detect` monorepo, which by this phase already had most of "Part A" (calibration, temperature scaling) implemented from the `metrics-integration-kusal` slice, plus a working `ProposedModel` (P4 scaffold) emitting swap-pair outputs. My additions:

- **`metrics/swap.py`** — `swap_prob_disagreement()` / `SCE_prob`: the continuous per-pixel `|p_fwd − p_rev|` companion to the existing hard `swap_agreement`, needs no ground truth. `sce_prob_pair_score()`: a GT-free pair-level confidence score on the same axis as MSR/Soft-Dice-Confidence. `SwapCalibrationMetric`: per-ordering ECE and **`Δ-ECE_swap`** — the calibration analogue of swap-consistency — plus whether `SCE_prob` actually correlates with pixel error (the track doc's Part B headline question) and how it ranks pairs on AURC against max-softmax-response.
- **`metrics/directional.py`** (new) — `DirectionalEquivarianceMetric`: checks whether the directional head's appeared/disappeared channels actually **swap** under input reversal (equivariance — the correct symmetry, per `PairOrderConsistencyLoss`'s own permutation logic) rather than merely stay the same (invariance — the wrong property for this task).
- Both registered in `METRIC_REGISTRY`, with `configs/metrics/*.yaml` drafts.
- **13 new correctness tests**, including confirming end-to-end through the real `ProposedModel` that abs-diff fusion gives `Δ-ECE_swap = 0` and `SCE_prob = 0` exactly — not just in isolated math, but through the actual composed model (`test_uncertainty_integration.py`).

No trained checkpoints exist anywhere in this project yet (`paper/sections/results.tex` in `change-detect` is still all `TBD`), so every test here is built on synthetic logits/masks, the same approach Phase 1 used for the overfit-one-batch check.

**Still open in this track:** reliability diagrams split by ordering; Part C's real blocker is that no dataset ships directional ground-truth labels, so the equivariance metric works but has nothing labelled to score accuracy against yet (derived/pseudo-labels or a synthetic source is an open decision); everything downstream of actual training (real `Δ-ECE_swap` numbers, risk-coverage curves, the title go/no-go call to Member 1) is blocked on someone running training.

## Vendored vs. authored here

Some files in this repo are **not my own work** — they're copied as-is (or with a trivial import-path fix) from the team's merged `change-detect` monorepo, included only so this repo's own code actually runs standalone. Each vendored file says so in its docstring. Roughly:

| Mine | Vendored (dependency only) |
|---|---|
| `models/encoders/`, `models/fusion/`, `models/decoders/`, `models/baselines/siamese_resnet18.py` (Phase 1) | `models/alignment/`, `models/heads/`, `models/proposed.py`, `models/_model_registry.py` (P4 scaffold) |
| `metrics/swap.py`'s `SwapCalibrationMetric`/`SCE_prob`, `metrics/directional.py` (Phase 2) | `metrics/calibration.py`, `metrics/temperature.py`, `metrics/swap.py`'s pre-existing `SwapConsistencyMetric` (`metrics-integration-kusal`) |
| `metrics/base.py`, `metrics/_tensor.py`, `metrics/registry.py`, `utils/registry.py` | shared plumbing, not really "owned" by any one track |

## Setup

```bash
pip install -e .        # installs the cdlib package (torch, torchvision, timm, omegaconf)
pip install -e .[dev]    # + pytest, pyyaml
pytest -q                                                             # full suite (61 tests)
pytest tests/test_shapes.py -v                                        # Phase 1 only
pytest tests/test_swap_calibration.py tests/test_directional_equivariance.py tests/test_uncertainty_integration.py -v   # Phase 2 only
python scripts/benchmark_encoders.py --batch-size 8 --img-size 256    # run on the actual Colab/Kaggle GPU
```

## Checklist status (mirrors `CLAUDE.local.md`)

- [x] Week 2 — `encoders/resnet.py`, `encoders/efficientnet.py`, registered, shape-tested
- [x] Week 2 — `configs/model/` entries (draft, pending P2's Hydra root)
- [x] Week 3 — `fusion/absdiff.py`, `fusion/signed_fusion.py` (all 4 modes), registered, shape-tested
- [x] Week 3–4 — `decoders/unet_decoder.py`, `baselines/siamese_resnet18.py`
- [x] Week 3 — pair-input GFLOPs added to the benchmark script
- [x] Weeks 5+ — stride-32 ablation, cost side (params/GFLOPs/stride, no training needed)
- [x] Weeks 5+ — Experiments/ablations paper section, methodology + tables drafted (numbers pending)
- [ ] Week 3 — encoder speed measurement run on an actual Colab/Kaggle GPU (script ready, needs a GPU session + team report-out)
- [ ] Week 3–4 — LR sweep + ≥3-seed tuning of the ResNet-18 baseline (needs P1's dataloaders + P2's trainer)
- [ ] Weeks 5+ — wire into `proposed.py` with P4; stride-32 ablation quality side (Boundary IoU, needs real training)

### Phase 2 (Member 5) checklist status (mirrors `docs/member-tracks/05`)

- [x] Part A — ECE/Brier/NLL/AURC (vendored, pre-existing) + **Δ-ECE_swap** (new, `SwapCalibrationMetric`)
- [x] Part B — `SCE_prob` as a label-free confidence signal, tested against a constructed error-correlated case; AURC comparison vs. MSR
- [ ] Part B — real risk-coverage curves and an abstention operating point (needs real predictions)
- [x] Part C — the equivariance metric itself (`DirectionalEquivarianceMetric`)
- [ ] Part C — an actual directional-label source (derived or synthetic) so the metric has something to score accuracy against
- [ ] Title go/no-go recommendation to Member 1 — blocked until real calibration numbers exist
