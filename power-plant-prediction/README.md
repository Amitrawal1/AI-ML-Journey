# ThermoNet: Power Plant Energy Output Predictor

## 📌 About The Project
This project is an Artificial Neural Network (ANN) model built using **PyTorch**. The goal of this model is to predict the **Net Hourly Electrical Energy Output (PE)** of a Combined Cycle Power Plant based on environmental factors like Temperature, Ambient Pressure, Vacuum, and Humidity.

## 🛠️ Tech Stack Used
* **Python** (Core Logic)
* **PyTorch** (Building and Training the Deep Learning Model)
* **Pandas & NumPy** (Data Handling)
* **Scikit-Learn** (Data Splitting, Scaling, and Evaluation)
* **Matplotlib** (Graph Visualization)

---

## 🚀 Step-by-Step Workflow
Here is exactly how the code works from start to finish:

### 1. Data Loading & Cleaning
* We load the `data_1a.csv` dataset using Pandas.
* We separate the input features (Temperature, Pressure, etc.) into the `X` variable and our target prediction (Energy Output / PE) into the `y` variable.

### 2. Train-Test Split
* We use `train_test_split` to divide our data into two parts: **80% for training** the model and **20% for testing** it on unseen data.

### 3. Data Scaling (Standardization)
* Neural networks work best when all numbers are on a similar scale. We use `StandardScaler` so that features with large values (like Pressure) don't overpower features with small values (like Temperature).

### 4. Converting to PyTorch Tensors & DataLoaders
* Since PyTorch doesn't directly understand Pandas DataFrames, we convert our scaled data into PyTorch **Tensors**. 
* We then group the data into batches of 32 using `DataLoader`. This helps the model train faster and more efficiently.

### 5. Building the ANN Model
We created a 4-layer neural network architecture:
* **Input Layer:** Takes the 4 environmental features.
* **Hidden Layers:** 3 separate hidden layers (each with 6 nodes) using the `ReLU` activation function to learn complex patterns.
* **Output Layer:** 1 single node that outputs the final predicted Megawatt value.

### 6. Training the Model
* **Optimizer & Loss:** We use the `Adam` optimizer and Mean Squared Error (`MSELoss`) to measure mistakes.
* The model trains for **100 Epochs**. After every epoch, it calculates both Training Loss and Validation Loss.
* **Auto-Save:** We added a smart feature that automatically saves the weights of the best-performing model (`best_model.pt`) whenever the validation loss drops to a new low.

### 7. Visualizing the Learning Curve
* The code uses Matplotlib to plot a graph comparing the Training Loss and Validation Loss over 100 epochs. This creates a perfect "L-shaped" curve, proving the model learned successfully without overfitting.

### 8. Final Evaluation
* We load our saved `best_model.pt` and test it one last time.
* We calculate the final **MSE Loss** and the **R2 Score** to measure exactly how accurate our model's predictions are compared to real-world power plant data.

---

## 💻 How to Run This Project
1. Clone this repository to your local machine.
2. Ensure you have PyTorch, Pandas, and Scikit-Learn installed (`pip install torch pandas scikit-learn matplotlib`).
3. Run the `project1.py` file or open the `.ipynb` notebook to see the step-by-step execution.
