# SimpleNet CNN with HOG Feature Extraction

## 📌 Project Overview
The SimpleNet project is a basic implementation of a feedforward neural network with HOG (Histogram of Oriented Gradients) feature extraction for image classification.
https://github.com/kavinkumarManielayaperumal/HOGNet-CNN
## 📂 File Descriptions

### **model.py**
This file defines the `HOG_INN_Deep` class, which inherits from PyTorch's `nn.Module`.

### **train.py**
This file contains the training loop for the neural network.

### **save_load_model.py**
This file provides functionality to save and load the trained model.

### **data_loader.py**
In this file, a custom dataset class is defined using PyTorch DataLoader.

### **main.py**
This is the main entry point of the project. It ties together dataset loading, training, and evaluation.

## ⚙️ Installation & Setup

### 🔹 1. Clone the Repository
```bash
git clone https://github.com/kavinkumarManielayaperumal/HOGNet-CNN
cd HOGNet-CNN

```

### 🔹 2. Set Up Virtual Environment
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
```

### 🔹 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📊 Dataset

### 📌 Dataset
We use the **MNIST dataset**, which consists of **28x28 grayscale images** of handwritten digits (0-9). The dataset is loaded from a local directory instead of using pre-built PyTorch datasets.

## 🚀 Workflow

### 🔹 1. Preprocessing: Load Dataset & Extract HOG Features
- Convert images to grayscale
- Apply **HOG feature extraction**
- Convert extracted features into a numerical format for model input

### 🔹 2. Model Architecture
- **Fully Connected Neural Network (FCNN)**
- Input size = Extracted **HOG features**
- Uses **Batch Normalization & Dropout** for better generalization

### 🔹 3. Training
```bash
python train.py
```
- Trains the model using **CrossEntropy Loss & Adam Optimizer**
- Saves the trained model as **hog_model.pth**

### 🔹 4. Evaluation
```bash
python evaluate.py
```
- Loads the trained model and evaluates on the test dataset
- Computes **accuracy and classification metrics**

## 🎯 Results
✅ **Test Accuracy:** ~98%
✅ **Balanced Performance:** High precision & recall for all classes
✅ **Predictions Saved:** `predictions.txt`

## 📂 Project Structure
```
├── data/                      # MNIST dataset (local bin files)
├── models/                    # Saved model weights
├── Feature_Extraction.py      # HOG feature extraction module
├── for_dataset_view.py        # Dataset loader (MNIST bin files)
├── data_loader.py             # PyTorch DataLoader for HOG
├── train.py                   # Training script
├── evaluate.py                # Model evaluation script
├── README.md                  # Project documentation
├── requirements.txt           # Dependencies
```

## 🤝 Contribution
Feel free to fork this repository, open issues, and submit pull requests!

## 📜 License
This project is open-source under the **MIT License**.

---

🚀 **Let's build more awesome models!**
