![AgriSmart Banner](image_name.png)

# AgriSmart: AI-Driven Crop Recommendation System 🌱

AgriSmart is a Deep Learning-based multiclass classification project designed to recommend the most suitable crop for cultivation based on soil and weather parameters. Built from scratch using **PyTorch**, this Artificial Neural Network (ANN) analyzes agricultural data to empower farmers with data-driven decision-making.

## 🚀 Features
* **Multiclass Classification:** Predicts the best crop out of **22 different classes** (e.g., Rice, Maize, Apple, Mango, etc.).
* **Deep Learning Architecture:** Custom Multi-Layer Perceptron (MLP) trained using PyTorch.
* **Robust Preprocessing:** Utilizes `LabelEncoder` for categorical target conversion and strictly prevents data leakage by properly applying `StandardScaler` `fit` and `transform` methods.
* **High Accuracy:** Achieved a peak testing accuracy of **97.73%** through systematic hyperparameter tuning.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch (`torch.nn`, `torch.optim`)
* **Data Processing:** Pandas, NumPy
* **Machine Learning Tools:** Scikit-Learn (`StandardScaler`, `LabelEncoder`, `train_test_split`)

## 📊 Dataset Parameters
The model is trained on the [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) from Kaggle. It takes exactly **7 input features**:
1. **N:** Ratio of Nitrogen content in soil
2. **P:** Ratio of Phosphorous content in soil
3. **K:** Ratio of Potassium content in soil
4. **Temperature:** Temperature in degree Celsius
5. **Humidity:** Relative humidity in %
6. **pH:** pH value of the soil
7. **Rainfall:** Rainfall in mm

## 🧠 Model Architecture & Hyperparameter Tuning
The final chosen architecture for the ANN features two hidden layers to capture the complex, non-linear relationships in the agricultural data:

* **Input Layer:** 7 Nodes (Features)
* **Hidden Layer 1:** 64 Nodes (ReLU Activation)
* **Hidden Layer 2:** 64 Nodes (ReLU Activation)
* **Output Layer:** 22 Nodes (Crop Classes)

**Training Configuration:**
* **Loss Function:** `CrossEntropyLoss` (ideal for multiclass output)
* **Optimizer:** Adam (Learning Rate: 0.001)
* **Batch Size:** 32
* **Epochs:** 100

### 📈 Experimental Results
During the development phase, various architectures were tested to find the optimal balance between performance and computational efficiency (preventing overfitting):

| Architecture | Nodes per Layer | Training Loss | Test Accuracy |
| :--- | :--- | :--- | :--- |
| 1 Hidden Layer | 32 | 0.0301 | 97.05% |
| 1 Hidden Layer | 64 | 0.0210 | 96.82% |
| 2 Hidden Layers | 32, 32 | 0.0194 | 97.27% |
| **2 Hidden Layers (Final)** | **64, 64** | **0.0079** | **97.73%** |

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Amitrawal1/AI-ML-Journey.git](https://github.com/Amitrawal1/AI-ML-Journey.git)
   cd AI-ML-Journey/AgriSmart