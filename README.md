

# XGBoost-Based Insurance Prediction Model with SHAP Explainability

## Project Overview
This project focuses on predicting insurance responses using an XGBoost model, with an emphasis on feature importance and interpretability using SHAP (SHapley Additive Explanations). The goal is to build a highly accurate predictive model while ensuring transparency in decision-making through SHAP analysis.

The project includes:
- **Training an XGBoost model** on structured tabular data.
- **Using SHAP** to explain how features influence predictions.
- **Evaluating model performance** with various ML metrics.

---

## Features
### **XGBoost Model Training**
- Data preprocessing and **feature engineering**.
- **Hyperparameter tuning** for optimal performance.
- Model evaluation using **accuracy, precision, recall, and ROC-AUC**.

### **SHAP Model Interpretability**
- **Feature importance analysis** to explain model behavior.
- **SHAP summary plots, force plots, and dependency plots** for visualization.
- Helps in making **transparent, data-driven decisions**.

### **Performance Evaluation**
- **Confusion matrix, ROC curves, and classification metrics**.
- Helps ensure the model is **accurate and interpretable**.

---

## Dataset Information
- **Dataset**: `train.csv` 
- **Features**:  Includes insurance applicant details, medical history, BMI, employment information, and other personal data.
- **Target Variable**: Predicting insurance response classes.

---

## Installation & Requirements
### **Prerequisites**
- **Python 3.8 or higher**
- Install dependencies: pip install -r requirements.txt

### **Required Libraries** 
xgboost, shap, scikit-learn, pandas, numpy, matplotlib, seaborn

---

## How to Run the Project
### **1. Clone the Repository**
git clone https://github.com/shalini363/xgboost_shap_feature_analysis.git
cd xgboost-shap-feature-analysis


### **2. Run the Jupyter Notebook** 
jupyter xgboost-shap-feature-analysis.ipynb

### **3. Follow Steps in the Notebook**
- Train the **XGBoost model**.
- Analyze **SHAP feature importance**.
- Evaluate the model's accuracy.

---


## **Goal**
The goal of this project is to **optimize insurance risk assessment** by **reducing dependency on excessive customer data**, improving privacy, and **maintaining prediction accuracy**. By applying **feature reduction techniques (PCA, t-SNE, SHAP, Autoencoders)**, the project enhances **decision-making efficiency** in insurance underwriting.

---

## **Experimentation and Results**
- Implemented various **dimensionality reduction techniques**: **PCA, t-SNE, Autoencoders**.
- Evaluated different models: **Logistic Regression, Random Forest, and XGBoost**.
- XGBoost **achieved a Quadratic Kappa score of 0.65**, balancing accuracy and interpretability.
- SHAP analysis highlighted **Medical_History_15 and BMI** as the most impactful features.
- Reduced feature set while maintaining model robustness.

---

## Key Learnings
- **SHAP enhances ML model transparency and interpretability**.
- **Feature analysis improves model insights and decision-making**.
- **XGBoost performs well with hyperparameter tuning**.

---

## Future Improvements
- Test the model on **additional insurance datasets**.
- Improve **hyperparameter tuning** for better performance.
- Explore **alternative interpretability methods (LIME, Permutation Importance)**.

