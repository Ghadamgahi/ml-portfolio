# Synthetic defect data: WGAN-GP, class-conditional diffusion, and a classifier trained on synthetic defects

Generative models produce extra defect images when real ones are scarce. A ResNet-18 trained with **only synthetic
defects** then reaches **87.8 % accuracy on real test images**.

## Problem

Supervised defect classifiers need many defect examples, but defects are rare on a production line. Can generated
images stand in for real ones?

## Approach

| Notebook | What it does |
|---|---|
| [`1_wgan_gp_synthetic_defects.ipynb`](1_wgan_gp_synthetic_defects.ipynb) | **WGAN-GP** trained to generate 128×128 grayscale line-scan crops. Details below. |
| [`2_conditional_ddpm_defect_generation.ipynb`](2_conditional_ddpm_defect_generation.ipynb) | **Class-conditional DDPM** built with 🤗 Diffusers that generates 500 images per defect class. Details below. |
| [`3_resnet18_trained_on_synthetic_defects.ipynb`](3_resnet18_trained_on_synthetic_defects.ipynb) | **Train on synthetic, test on real.** An ImageNet ResNet-18 learns from real *good* crops plus DDPM-generated *defect* crops, then is evaluated on real images only. Two variants: full fine-tuning and a frozen backbone. |

**WGAN-GP details**
* DCGAN-style generator and critic, gradient penalty with λ = 10.
* Generated samples are filtered by an independent classical detector: MSE + SSIM against a defect-erased version of
  the image, scored by a One-Class SVM. See [`../one-shot-defect-detection`](../one-shot-defect-detection).

**Conditional DDPM details**
* `UNet2DModel` with `num_class_embeds`, 1000-step DDPM scheduler, fp16 training with 🤗 Accelerate.
* Images are generated with a manual class-conditional sampling loop.

## Results

From the notebook outputs. The test set contains real images only:

| Classifier (ResNet-18, ImageNet weights) | Real-image test accuracy | Inference time per image |
|---|---|---|
| Full fine-tuning | **0.878** | 5.6 ms (GPU) |
| Frozen backbone, linear head only | 0.814 | 6.9 ms (GPU) |

* The validation split (accuracy 1.0) comes from the same synthetic + real-good pool as training, so it is optimistic.
  The **real** test set is the meaningful number, and the gap shows how much synthetic data still differs from real
  data.
* The generators themselves were not scored with a quantitative metric such as FID. Their samples were checked
  visually, and those images were removed here because they depict proprietary products.
* The WGAN-GP critic still uses BatchNorm, which the WGAN-GP paper advises against.
* The diffusion notebook is adapted from the official 🤗 Diffusers training tutorial. Class conditioning and the
  conditional sampling loop were added on top. It was saved without outputs.

## Tech

PyTorch · WGAN-GP · 🤗 Diffusers (DDPM, UNet2DModel) · 🤗 Accelerate · torchvision ResNet-18 transfer learning ·
scikit-learn One-Class SVM · scikit-image

## How to run

```bash
pip install -r requirements.txt
export DATA_ROOT=/path/to/data
# 1) train a generator (WGAN-GP or DDPM) on the defect crops
# 2) point notebook 3 at the generated folder (img_dir2) and at real good/bad test crops
```

The production images are **proprietary and not included**. Any two-class set of grayscale crops (good / defect) works
with the same folder layout.
