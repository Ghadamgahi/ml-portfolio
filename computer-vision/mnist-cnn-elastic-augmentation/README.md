# MNIST CNN with elastic-distortion augmentation: 99.62 % test accuracy

A small CNN reaches **99.62 % accuracy (38 errors in 10 000)** on the MNIST test set. The gain comes from heavy
affine and elastic-distortion augmentation, which expands 60 k training images to 1.2 M.

## Approach

* **Pre-processing.** Centre crop to 20×20 and per-image mean subtraction.
* **Augmentation.**
  * Affine: random rotation (±20°), shear (0.2) and shift (±20 %), multiplying the data ×4.
  * **Elastic distortions** ([Simard et al., 2003](https://cognitivemedium.com/assets/rmnist/Simard.pdf)): Gaussian-
    smoothed random displacement fields (α = 6, σ = 4), multiplying the data ×5.
* **Model.**
  * Conv(20) → MaxPool → Conv(40) → MaxPool → Dense(640) → Dropout(0.5) → Dense(1000) → Dropout(0.5) → Softmax.
  * SGD with Nesterov momentum 0.95, 30 epochs.

## Results

From the logged run: test loss 0.0214, **test accuracy 0.9962**, 38 misclassified test images.

## Tech

TensorFlow / Keras · SciPy (`map_coordinates`, `gaussian_filter`) · NumPy

## How to run

```bash
pip install -r requirements.txt        # TensorFlow 2.15 (Keras 2 API), Python 3.9–3.11
jupyter notebook mnist_cnn_elastic_augmentation.ipynb
```

MNIST is downloaded automatically by Keras.

* The original experiment ran on Keras 2.1.5 / TensorFlow 1.15, and the training log in the notebook is from that run.
* The code has since been ported to `tf.keras` and smoke-tested end-to-end on a subset with TensorFlow 2.15.
* A full run trains on 1.2 M images, about 6 h on the original GPU.
