**Zero-Leakage Financial Fraud Detection Pipeline**
An end-to-end supervised machine learning pipeline built to tackle highly imbalanced financial datasets (99.83% legitimate vs. 0.17% fraud) while strictly adhering to data science best practices like **zero data leakage** and **safe cross-validation resampling**.
**Project Overview**
In financial fraud detection, standard accuracy is a dangerous illusion. A lazy model predicting all transactions as "legitimate" achieves 99.8% accuracy while catching zero fraud. 
This project solves the class imbalance problem using **SMOTE** (Synthetic Minority Over-sampling Technique) bundled inside **Imblearn Pipelines**, ensuring that synthetic data generation happens *only* on training folds during cross-validation—leaving the test set completely untouched and secure.
**Architecture & Tech Stack**
* **Language:** Python 
* **Libraries:** Pandas, Scikit-Learn, Imbalanced-Learn (Imblearn)
* **Models Tested:** Logistic Regression (with `StandardScaler`), Random Forest Classifier
* **Evaluation Metrics:** Precision, Recall, Confusion Matrix, and ROC-AUC Score (avoiding the accuracy trap)
