# Edge auto-encoder for surface-defect detection (INT8 TFLite)

Convolutional auto-encoders that flag defects on line-scan images by their reconstruction error. A compact variant is
designed for, and converted to, a fully INT8-quantised TFLite model for a microcontroller-class edge device.

## Problem

The goal is to detect defects on grayscale line-scan images of laminated panels on the production line. The model has
to be small enough to run on an embedded device next to the camera.

## Approach

* **Data preparation.**
  * Full-width scans are cut into overlapping sub-images (150 px or 128 px wide, shift 50 px).
  * The data is split 70/10/20 into train, validation and test with a fixed seed.
* **Training target ("couple" mode).** Normal crops are reconstructed as themselves, while defective crops are mapped
  to a fixed defect-free reference crop. The network learns to "repair" defects, so the reconstruction error spikes
  where a defect is present.
* **Scoring.** The anomaly score is the MSE between input and reconstruction, with image borders cropped. The decision
  threshold maximises TPR − FPR (Youden's J) on the ROC curve.
* **Training.** AdamW optimiser, learning-rate reduction on plateau, early stopping on the validation loss.
* **Edge variant.** The decoder uses `Upsample + Conv2d` instead of `ConvTranspose2d`, because transposed convolutions
  with output padding are not supported by the converter. The model is then exported with **ai-edge-torch** to a fully
  INT8 TFLite model: a representative dataset from the training loader, `TFLITE_BUILTINS_INT8` ops, and int8
  input/output. An optional ONNX export is included.

## Results

The confusion matrices are taken from the notebook outputs. Accuracy, precision and recall were computed from them.

| Notebook | Input | Test set | Confusion matrix `[[TN FP] [FN TP]]` | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|
| [`conv_autoencoder_defect_detection.ipynb`](conv_autoencoder_defect_detection.ipynb) | 100×150 | 832 crops | `[[521 6] [17 288]]` | **97.2 %** | 98.0 % | 94.4 % |
| [`edge_autoencoder_int8_tflite.ipynb`](edge_autoencoder_int8_tflite.ipynb) | 32×128 | 504 crops | `[[139 4] [27 334]]` | **93.8 %** | 98.8 % | 92.5 % |

The edge model also reaches 98.0 % accuracy on its 251-image threshold-selection set (`[[71 0] [5 175]]`). Its INT8
TFLite export completed successfully (output path shown in the notebook).

The first notebook defines four candidate encoder/decoder architectures. The saved run does not record which of them
produced the logged confusion matrix, so treat that table row as the best result of the architecture search.

## Tech

PyTorch · torchinfo · scikit-learn (ROC, confusion matrix) · ai-edge-torch · TensorFlow Lite (INT8 post-training
quantisation) · ONNX

## How to run

```bash
pip install -r requirements.txt
export DATA_ROOT=/path/to/data   # <product>/Zipped/normal.zip and defects.zip
jupyter notebook conv_autoencoder_defect_detection.ipynb
```

The production images are **proprietary and not included**. Product names were anonymised, and sample and
reconstruction images were removed from the outputs. Any folder of grayscale crops with `normal/` and `defects/` zips
can be used. The notebooks were developed on Colab with a T4 GPU.
