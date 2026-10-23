# [YOUR NAME]: Machine Learning & Computer Vision Portfolio

[SHORT BIO: 2–3 sentences, e.g. "ML engineer focused on computer vision for industrial inspection: anomaly detection,
synthetic data generation and deployment to edge hardware."]

📫 [EMAIL] · [LINKEDIN] · [WEBSITE]

---

## Highlights

* **Industrial anomaly detection.** LGFDR (Swin-B) reaches **0.980 image AUROC** on MVTec AD *transistor*. PatchCore
  keeps **≥ 0.988 AUROC** even when 20 % of its "good" training images are defective or ambiguous.
  → [industrial-anomaly-detection](computer-vision/industrial-anomaly-detection)
* **Train on synthetic, test on real.** A ResNet-18 trained only on diffusion-generated defect images classifies
  **real** defects with **87.8 % accuracy**.
  → [synthetic-defect-data](computer-vision/synthetic-defect-data)
* **Edge deployment.** A convolutional auto-encoder with **97.2 % test accuracy**, plus a compact variant exported to a
  fully **INT8 TFLite** model. → [edge-autoencoder-defect-detection](computer-vision/edge-autoencoder-defect-detection)

## Projects

### Computer Vision

| Project | What it does | Key tech |
|---|---|---|
| [Industrial anomaly detection](computer-vision/industrial-anomaly-detection) | Unsupervised defect detection with LGFDR (dual local/global feature reconstruction) on MVTec AD and wood panels; PatchCore baseline and training-set contamination study | PyTorch, timm, Swin Transformer, PatchCore, FAISS |
| [Synthetic defect data](computer-vision/synthetic-defect-data) | WGAN-GP and class-conditional DDPM generate defect images; a ResNet-18 trained on them is tested on real defects | PyTorch, 🤗 Diffusers, Accelerate, transfer learning |
| [Edge auto-encoder](computer-vision/edge-autoencoder-defect-detection) | Reconstruction-based defect detection on line-scan images, exported to INT8 TFLite for an edge device | PyTorch, ai-edge-torch, TFLite quantisation, ONNX |
| [CycleGAN defect removal](computer-vision/cyclegan-defect-removal) | CycleGAN "repairs" defective images; SSIM between input and output scores anomalies (0.925 AUROC) | CycleGAN, OpenCV, scikit-image, SVM |
| [One-shot defect detection](computer-vision/one-shot-defect-detection) | Learning-free detector built from a defect-erased reference image and distance features | OpenCV, scikit-image, scikit-learn |
| [MNIST CNN + elastic augmentation](computer-vision/mnist-cnn-elastic-augmentation) | CNN with affine and elastic-distortion augmentation, 99.62 % test accuracy | TensorFlow/Keras, SciPy |
| [Robust fitting: A* vs RANSAC](computer-vision/robust-fitting-consensus-maximisation) | Study of globally optimal consensus maximisation (CVPR 2015) vs RANSAC for homography / image stitching | MATLAB, VLFeat SIFT, RANSAC |

### Machine Learning & Data Analysis

| Project | What it does | Key tech |
|---|---|---|
| [Spiral dataset classification](machine-learning/spiral-dataset-classification) | Benchmark of 6 tuned classifiers (99.3 % holdout accuracy) and reverse-engineering of a synthetic data generator with PCA and manifold learning | scikit-learn, XGBoost, PCA / t-SNE / Isomap |
| [Time-series forecasting](machine-learning/time-series-forecasting-lstm) | BiLSTM + Conv1D forecaster for 48 000 series; the autoregressive model scored 0.00536 test MSE (team challenge) | TensorFlow/Keras, LSTM |
| [MLP from scratch](machine-learning/mlp-from-scratch) | Back-propagation and gradient descent implemented in NumPy | NumPy |

### NLP / LLMs

*No public projects yet.* [Add LLM / RAG work here.]

## Skills demonstrated in this repository

* **Computer vision:**
  * unsupervised anomaly detection (feature reconstruction, PatchCore, auto-encoders, CycleGAN);
  * transfer learning with CNNs and Vision Transformers;
  * classical image processing (OpenCV, morphology, SSIM/PSNR, SIFT, RANSAC).
* **Generative models:** GANs (WGAN-GP, CycleGAN) and diffusion models (class-conditional DDPM with 🤗 Diffusers).
* **Deep learning:** PyTorch, TensorFlow/Keras, timm, LSTM/Conv1D sequence models, data augmentation, training
  schedules and early stopping.
* **Deployment:** INT8 post-training quantisation and TFLite export (ai-edge-torch), ONNX, inference benchmarking.
* **Classical ML:** scikit-learn and XGBoost, hyper-parameter search, stratified cross-validation, leakage-safe
  pre-processing, PCA and manifold learning, SVMs.
* **Evaluation:** ROC/AUROC, average precision, threshold selection, confusion matrices, held-out testing.

## Repository layout

```
computer-vision/<project>/      notebooks, README.md, requirements.txt, assets/
machine-learning/<project>/     notebooks, scripts, README.md, requirements.txt, assets/
```

Every project README explains the problem, the approach, the results (copied from the notebook outputs) and how to run
it.

* Paths are configured through a `DATA_ROOT` environment variable, so nothing depends on Google Drive.
* Datasets and model weights are not committed.
* Several projects come from industrial work on proprietary data. Their notebooks keep only code, plots and metrics;
  product names are anonymised.
