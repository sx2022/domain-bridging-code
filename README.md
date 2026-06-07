# Domain Bridging — Bi-Level Domain Bridging (BL-DB)

Reference implementation accompanying the paper
*"Domain Bridging: Enabling Adaptation without Peeking at Target Data."*

This release reproduces the core **BL-DB** method on the **Office-31** benchmark.
The other datasets reported in the paper (Office-Home, PACS, VLCS, Amazon Review)
use the **identical algorithm**; only the data loading and the backbone differ.

---

## Repository structure

| File | Description |
|------|-------------|
| `main_new.py` | Entry point. Loads the source data, the target support/holdout split, and the pre-trained source model, then runs the BL-DB adaptation loop and logs support/holdout accuracy. |
| `bilevel_bb_updated.py` | Core BL-DB step: zeroth-order (SPSA) estimate of the parameter gradient, the one-step unrolled hypergradient w.r.t. the sample-wise perturbation `delta`, and the L2 projection. |

---

## Dependencies

```bash
pip install -r requirements.txt
```

Tested with Python 3.8+ and PyTorch on a single GPU. See `requirements.txt`
for the full list. Note: the code uses the classic
`torchvision.models.resnet50(pretrained=True)` API.

---

## Data preparation

The code consumes **pre-processed NumPy arrays**. Place the following files under
`--data_dir` (default `../data`):

```
# source domain (training)
{domain}_X_train.npy      # image tensors, shape (N, 3, 224, 224), ImageNet-normalized
{domain}_Y_train.npy      # integer class labels in [0, 30]

# target domain (evaluation): support = oracle used during adaptation, holdout = generalization test
{domain}_X_support.npy    {domain}_Y_support.npy
{domain}_X_holdout.npy    {domain}_Y_holdout.npy
```

where `domain` is one of `a` (Amazon), `w` (Webcam), `d` (DSLR) for Office-31.
The target support/holdout files are an even (50/50) split of the target domain.

**Original dataset.** Office-31 is publicly available, e.g.
<https://github.com/jindongwang/transferlearning/tree/master/data#office-31>.
Resize images to 224×224 and apply standard ImageNet normalization before saving
to `.npy`.

---

## Pre-trained source models

Place a source-domain ResNet-50 checkpoint at:

```
{model_dir}/resnet50_{source_domain}      # e.g. ../resnet50_a
```

This is a ResNet-50 fine-tuned on the source domain (the model to be adapted).
If the checkpoint is missing, the code falls back to an ImageNet-pretrained
ResNet-50 with a freshly initialized 31-way head.

---

## Running

```bash
# BL-DB (our method): Amazon -> Webcam, paper hyper-parameters
python main_new.py --source_domain a --target_domain w \
    --gradient_type estimate \
    --lr_theta 0.075 --lr_delta 0.01 --perturb_scale 0.01 \
    --num_epochs 10 --batch_size 64

# BL-DB* (exact-gradient upper bound)
python main_new.py --source_domain a --target_domain w \
    --gradient_type true \
    --lr_theta 0.075 --lr_delta 0.01 --perturb_scale 0.01
```

Per-batch support/holdout accuracy is printed to stdout, and the accuracy curves
plus the run configuration are saved under `--output_dir` (default `results/`).

### Key arguments

| Flag | Meaning | Paper value |
|------|---------|-------------|
| `--source_domain` / `--target_domain` | Office-31 domains (`a`/`w`/`d`) | — |
| `--gradient_type` | `estimate` = BL-DB, `true` = BL-DB* (exact-gradient oracle) | `estimate` |
| `--lr_theta` (ξ) | model-parameter learning rate | `0.075` |
| `--lr_delta` (η) | perturbation learning rate | `0.01` |
| `--perturb_scale` (γ) | ZO smoothing scale | `0.01` |
| `--support_ratio` | class-balanced fraction of the target support set | `1.0` |
| `--num_epochs`, `--batch_size` | optimization budget | `10`, `64` |

---

## Results

Average holdout accuracy on Office-31 (mean over the 6 transfer pairs
`a→w, a→d, w→a, w→d, d→a, d→w`), reproduced with the commands above:

| Method | `--gradient_type` | Office-31 avg. (%) |
|--------|-------------------|--------------------|
| BL-DB  | `estimate`        | 78.60 |
| BL-DB* | `true`            | 80.05 |

To reproduce a number in the table, run each of the 6 pairs and average the best
holdout accuracy.

---

## Computing infrastructure

All experiments were run on a single NVIDIA RTX A6000 GPU (48 GB). Adapting one Office-31
transfer pair takes on the order of minutes (the method converges within ~3–4
epochs over the source data).
