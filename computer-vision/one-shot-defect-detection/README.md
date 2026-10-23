# One-shot, learning-free defect detection (classical computer vision)

A classical image-processing pipeline that detects defects on line-scan images using only a handful of reference
images. There is no deep learning: each image is compared with a "defect-erased" version of itself.

## Problem

New products often arrive with no labelled data at all. The goal was a detector that works from day one: no
training set, fast on a CPU, and explainable.

## Approach

1. **Defect-erased reconstruction.**
   * Histogram equalisation.
   * Every row is replaced by its maximum (or mean) intensity. Small dark defects disappear while the stripe profile
     stays.
   * Morphological dilation and erosion clean up the result.
2. **Stripe localisation.**
   * The stripe borders and its centre line are found from the row-mean intensity profile, using a threshold below the
     peak plus a local-minimum search.
   * Outside the stripe, the original pixels are restored.
   * A template-correlation alternative is also explored.
3. **Distance features.** MSE, PSNR, SSIM, and a patch-wise MSE-exceedance count, each computed between an image and
   its reconstruction.
4. **Decision.** Either a threshold on one distance chosen from the ROC curve, or an RBF-SVM on (MSE or PSNR, SSIM)
   with an 80/20 split.

## Results

From the notebook outputs, on 292 labelled test images:

| Decision rule | Evaluation | Accuracy |
|---|---|---|
| Threshold on MSE (8281) | threshold chosen on the same 292 images (in-sample) | 0.979 |
| RBF-SVM on (MSE, SSIM) | 20 % held-out split | 1.000 |
| RBF-SVM on (PSNR, SSIM) | 20 % held-out split | 0.915 |

The threshold result is optimistic because the threshold was picked on the evaluation images. The SVM numbers come
from a small held-out split of about 59 images. The same distance features are reused to filter GAN samples in
[`../synthetic-defect-data`](../synthetic-defect-data).

## Tech

OpenCV · NumPy · scikit-image (SSIM, PSNR) · scikit-learn (ROC, SVM) · Matplotlib

## How to run

```bash
pip install -r requirements.txt
export DATA_ROOT=/path/to/data   # line_scan/train/*.png and line_scan/test/{good,bad}/*.png
jupyter notebook one_shot_defect_detection.ipynb
```

The images are proprietary and **not included**. Product images were removed from the outputs.
