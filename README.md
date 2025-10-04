# 📊 Customer Churn Prediction  

This project predicts **customer churn** (whether a customer is likely to leave or stay) using **Machine Learning and Deep Learning models**. It compares traditional algorithms like **KNN, Decision Tree, and Random Forest** with a **Neural Network (ANN) built using Keras**.  

---

## 🚀 Project Workflow  

### 1️⃣ Data Preprocessing  
- Removed duplicates & handled missing values  
- Handled **outliers using Z-score**  
- Encoded categorical variables using **OneHotEncoder & LabelEncoder**  
- Scaled numerical features with **MinMaxScaler**  

### 2️⃣ Exploratory Data Analysis (EDA)  
- **Boxplots** to detect outliers  
- **Correlation Heatmap** to analyze feature relationships  
- Feature reduction by removing highly correlated variables  

### 3️⃣ Machine Learning Models  
- **K-Nearest Neighbors (KNN)**  
- **Decision Tree**  
- **Random Forest**  

### 4️⃣ Deep Learning Model (Keras ANN)  
- Input layer with **64 neurons (ReLU activation)**  
- **Dropout layer (0.3)** to reduce overfitting  
- Hidden layer with **32 neurons (ReLU activation)**  
- Output layer with **1 neuron (Sigmoid activation)** for binary classification  
- Optimized using **Adam** with **binary crossentropy loss**  

### 5️⃣ Model Evaluation  
- Metrics: **Accuracy, Precision, Recall, F1-score, Confusion Matrix**  
- Feature importance analysis (Random Forest)  

---

## 📈 Results  

| Model             | Accuracy | Key Notes |
|-------------------|----------|-----------|
| **KNN**           | ~49.55%  | Sensitive to noise & high-dimensional data |
| **Decision Tree** | ~52.03%  | Best among ML models, but risk of overfitting |
| **Random Forest** | ~51.13%  | More stable, limited improvement |
| **Keras ANN**     | ~XX.XX%  | Deep learning model with competitive performance |

---

## 🛠️ Technologies & Skills Used  

- **Programming:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow, Keras  
- **Concepts:**  
  - Data Cleaning & Preprocessing  
  - Outlier Handling (Z-score)  
  - Feature Engineering & Scaling  
  - Classification Algorithms (KNN, Decision Tree, Random Forest, ANN)  
  - Model Training & Evaluation (Accuracy, F1, Precision, Recall)  
  - Feature Importance Analysis  

---

## 📌 How to Run  

1. Clone this repository:  
   ```bash
   git clone https://github.com/Hritro/churn-prediction.git
   cd churn-prediction

## Install dependencies

   pip install -r requirements.txt
