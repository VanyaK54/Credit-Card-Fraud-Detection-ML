# Credit Card Fraud Detection - Machine Learning Project

## 🔍 Overview
This project uses machine learning to detect fraudulent credit card transactions. The dataset is highly imbalanced, making the problem realistic and challenging.

## 📁 Project Structure

```bash
credit-card-fraud-detection/
├── data/ # Raw dataset (creditcard.csv)
├── notebooks/ # Jupyter notebook for EDA
├── src/ # Source code for preprocessing, training, and evaluation
├── outputs/ # Generated figures and plots
├── main.py # Main script to run the pipeline
├── requirements.txt # Python dependencies
└── README.md # Project overview and instructions
```


## 📦 Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions
- Features: `V1` to `V28`, `Amount`, `Time`
- Target: `Class` (1 for fraud, 0 for normal)

## ⚙️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/VanyaK54/Credit-Card-Fraud-Detection-ML
cd credit-card-fraud-detection
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Add dataset
Download creditcard.csv and place it inside the data/ folder.

### 4. Run the pipeline
```bash
python main.py
```

### 📊 Output
Prints evaluation metrics: Accuracy, Precision, Recall, F1-Score, MCC

Saves:

Confusion matrix: outputs/figures/confusion_matrix.png

Correlation heatmap: outputs/figures/correlation_matrix.png

### ✅ Techniques Used
SMOTE (Synthetic Minority Over-sampling)

Random Forest Classifier

Evaluation using precision, recall, MCC

Correlation matrix and class analysis

### 📚 Libraries
pandas, numpy

matplotlib, seaborn

scikit-learn

imbalanced-learn

### 🧠 Future Improvements
Try XGBoost, LightGBM, or AutoML

Build a Streamlit app for real-time predictions

Add dashboards via PowerBI/Tableau
