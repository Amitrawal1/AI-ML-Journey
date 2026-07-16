# CIFAR-10 Image Classification using CNN 🖼️🧠

A deep learning project focused on Computer Vision. This repository contains a Convolutional Neural Network (CNN) built from scratch using PyTorch to classify images from the famous CIFAR-10 dataset into 10 distinct categories. 

## 🚀 Project Overview
Unlike traditional tabular data, image processing requires spatial feature extraction. This project implements a multi-layered CNN architecture to learn visual patterns (like edges, textures, and shapes) and accurately classify 32x32 color images.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch (`torch.nn`, `torch.optim`)
* **Computer Vision Tools:** Torchvision (`transforms`, `datasets`)
* **Data Processing:** DataLoader

## 📊 Dataset: CIFAR-10
The model is trained on the standard CIFAR-10 dataset, which consists of 60,000 color images (32x32 pixels) divided into 10 classes:
`Airplane`, `Automobile`, `Bird`, `Cat`, `Deer`, `Dog`, `Frog`, `Horse`, `Ship`, `Truck`.

**Preprocessing Applied:**
* Converted raw images to PyTorch Tensors.
* Normalized pixel values to a range of `[-1, 1]` using mean `(0.5, 0.5, 0.5)` and standard deviation `(0.5, 0.5, 0.5)` for stable and faster convergence.

## 🧠 CNN Model Architecture
The custom CNN architecture is designed with 3 Convolutional blocks for feature extraction, followed by a Fully Connected (Dense) network for classification.

**1. Feature Extraction (Convolutional Layers):**
* **Block 1:** `Conv2d` (3 -> 32 channels) ➔ ReLU ➔ `MaxPool2d` (2x2)
* **Block 2:** `Conv2d` (32 -> 64 channels) ➔ ReLU ➔ `MaxPool2d` (2x2)
* **Block 3:** `Conv2d` (64 -> 128 channels) ➔ ReLU ➔ `MaxPool2d` (2x2)

**2. Classification (Fully Connected Layers):**
* **Flattening Layer:** Converts the 3D tensor (`128 x 4 x 4`) into a 1D vector of `2048` features.
* **Hidden Layer:** Linear (2048 -> 256) ➔ ReLU
* **Output Layer:** Linear (256 -> 10) (Predicts the 10 target classes)

**Training Configuration:**
* **Loss Function:** `CrossEntropyLoss`
* **Optimizer:** Adam Optimizer
* **Batch Size:** 64
* **Epochs:** 10

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Amitrawal1/AI-ML-Journey.git](https://github.com/Amitrawal1/AI-ML-Journey.git)
   cd AI-ML-Journey/CNN-CIFAR10