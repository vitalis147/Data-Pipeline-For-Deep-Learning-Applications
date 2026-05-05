# Data Pipeline with Keras and tf.data

A deep learning project demonstrating end-to-end data ingestion, preprocessing, and augmentation pipelines using `tf.keras` and `tf.data`, applied to image classification tasks on the **LSUN** and **CIFAR-100** datasets.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Part 1: tf.keras — LSUN Scene Classification](#part-1-tfkeras--lsun-scene-classification)
- [Part 2: tf.data — CIFAR-100 Subset Classification](#part-2-tfdata--cifar-100-subset-classification)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Key Concepts](#key-concepts)

---

## Overview

This project builds two complete deep learning pipelines:

| Pipeline | Dataset | Tool | Classes |
|---|---|---|---|
| Part 1 | LSUN (subset) | `tf.keras.ImageDataGenerator` | church, classroom, conference_room |
| Part 2 | CIFAR-100 (subset) | `tf.data.Dataset` | apple (0), worm (29), dinosaur (99) |

Both pipelines feed a shared CNN architecture and demonstrate the impact of **data augmentation** on reducing overfitting.

---

## Project Structure

```
├── data/
│   ├── lsun/
│   │   ├── train/          # 300 training images (3 classes)
│   │   ├── valid/          # 120 validation images
│   │   └── test/           # 300 test images
│   └── cifar100/
│       └── cifar100_labels.json
├── coding tutorial for data pipeline for deep learning tasks.ipynb
├── deep pipeline for deep learning.ipynb          # Main project notebook
└── README.md
```

---

## Part 1: tf.keras — LSUN Scene Classification

### Dataset
A subset of the [LSUN dataset](https://arxiv.org/abs/1506.03365) with three scene categories:

| Class | Label |
|---|---|
| classroom | `[1., 0., 0.]` |
| conference_room | `[0., 1., 0.]` |
| church_outdoor | `[0., 0., 1.]` |

### Pipeline

**Base pipeline (no augmentation):**
```python
def get_ImageDataGenerator():
    return ImageDataGenerator(rescale=1/255.)
```

**Augmented pipeline:**
```python
def get_ImageDataGenerator_augmented():
    return ImageDataGenerator(
        rescale=1/255.,
        brightness_range=(0.5, 1.5),
        rotation_range=30,
        horizontal_flip=True
    )
```

**Generator configuration:**
- Batch size: `20`
- Target image size: `64 × 64 × 3`
- Label mode: `categorical` (one-hot)

### Model Architecture

Built using the Keras **Functional API**:

```
Input(64, 64, 3)
    → Conv2D(8 filters, 8×8, ReLU, SAME)
    → MaxPooling2D(2×2)
    → Conv2D(4 filters, 4×4, ReLU, SAME)
    → MaxPooling2D(2×2)
    → Flatten
    → Dense(16, ReLU)
    → Dense(3, Softmax)
```

| Parameter | Value |
|---|---|
| Total params | 18,511 |
| Optimizer | Adam (lr=0.0005) |
| Loss | Categorical Cross-Entropy |
| Metric | Accuracy |

### Training Callbacks

```python
EarlyStopping(monitor='val_accuracy', patience=10)
ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_lr=0.0001)
```

---

## Part 2: tf.data — CIFAR-100 Subset Classification

### Dataset
A filtered subset of [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) — 100 classes, 500 training / 100 test images per class — narrowed to three classes:

| Class Index | Label | One-Hot |
|---|---|---|
| 0 | apple | `[1., 0., 0.]` |
| 29 | worm | `[0., 1., 0.]` |
| 99 | dinosaur | `[0., 0., 1.]` |

### Pipeline Functions

**1. Create Dataset**
```python
def create_dataset(data, labels):
    return tf.data.Dataset.from_tensor_slices((data, labels))
```

**2. Filter by Class**
```python
def filter_classes(dataset, classes):
    def is_allowed(image, label):
        allowed = tf.cast(classes, label.dtype)
        matches = tf.equal(label, allowed)
        return tf.math.reduce_any(matches)
    return dataset.filter(is_allowed)
```

**3. One-Hot Encode Labels**
```python
def map_labels(dataset):
    def one_hot_encoding(data, labels):
        one_hot = tf.cast(
            tf.equal(labels, tf.constant([0, 29, 99], dtype=tf.int64)),
            tf.float32
        )
        return data, one_hot
    return dataset.map(one_hot_encoding)
```

**4. Preprocess Images**
```python
def map_images(dataset):
    def rescale_and_transform(data, labels):
        rescaled = tf.cast(data, tf.float32) / 255.
        grayscale = tf.reduce_mean(rescaled, axis=-1, keepdims=True)
        return grayscale, labels
    return dataset.map(rescale_and_transform)
```

Image transformation summary:

```
Input:          (32, 32, 3)  uint8   — values 0–255
After /255:     (32, 32, 3)  float32 — values 0.0–1.0
After mean:     (32, 32, 1)  float32 — grayscale, values 0.0–1.0
```

**5. Batch and Shuffle**
```python
train_dataset_bw = train_dataset_bw.batch(10).shuffle(100)
test_dataset_bw  = test_dataset_bw.batch(10).shuffle(100)
```

---

## Results

### Part 1 — LSUN (without augmentation)
| Metric | Training | Validation |
|---|---|---|
| Final Accuracy | ~91% | ~65% |
| Observation | Significant overfitting after epoch 20 |

### Part 1 — LSUN (with augmentation)
| Metric | Training | Validation |
|---|---|---|
| Final Accuracy | ~74% | ~70% |
| Observation | Reduced train/val gap — less overfitting |

> **Key insight:** Data augmentation (rotation, brightness, horizontal flip) narrowed the generalisation gap significantly, at the cost of slightly lower training accuracy — a desirable trade-off.

### Part 2 — CIFAR-100 Subset
| Metric | Training | Validation |
|---|---|---|
| Final Accuracy (Epoch 15) | ~82.6% | ~81.0% |
| Final Loss | 0.4435 | 0.5073 |

> Training and validation curves remain closely aligned, indicating healthy generalisation on the 3-class grayscale task.

---

## Requirements

```txt
tensorflow >= 2.x
numpy
matplotlib
```

Install dependencies:
```bash
pip install tensorflow numpy matplotlib
```

---

## Usage

```python
# Clone the repo and launch the notebook
git clone https://github.com/your-username/data-pipeline-keras-tfdata.git
cd data-pipeline-keras-tfdata
jupyter notebook notebook.ipynb
```

Run all cells in order. Sections are clearly separated into **Part 1** (tf.keras) and **Part 2** (tf.data).

---

## Key Concepts

| Concept | Where Used |
|---|---|
| `ImageDataGenerator` | Part 1 — loading and augmenting images from disk |
| `flow_from_directory` | Part 1 — streaming batches with one-hot labels |
| `tf.data.Dataset.from_tensor_slices` | Part 2 — creating datasets from NumPy arrays |
| `Dataset.filter` + `tf.math.reduce_any` | Part 2 — filtering by class index |
| `Dataset.map` + `tf.reduce_mean` | Part 2 — grayscale conversion with keepdims |
| `tf.equal` + `tf.cast` | Part 2 — custom one-hot encoding |
| `EarlyStopping` + `ReduceLROnPlateau` | Part 1 — training stability callbacks |
| Functional API | Both parts — CNN model definition |
| Data augmentation | Part 1 — mitigating overfitting |

---

## References

- Yu, F. et al. (2015). *LSUN: Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop*. [arXiv:1506.03365](https://arxiv.org/abs/1506.03365)
- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. University of Toronto.
- [TensorFlow tf.data Guide](https://www.tensorflow.org/guide/data)
- [Keras ImageDataGenerator Docs](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
