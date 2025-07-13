# 🧠 Fraud transactions analysis in Banking using Python

<img width="1440" height="754" alt="Fraudulent-transactions" src="https://github.com/user-attachments/assets/b592d503-12cf-4935-8b61-79ebf7382471" />

**Author**: Nguyen Anh Huy  
**Date**: 13/07/2025  
**Tools Used**: Python (Jupyter Notebook)

---

## I. 🎯 Background & Overview

📖 **What is this project about?**  
As financial fraud becomes increasingly sophisticated, early detection through machine learning has become essential. Traditional rule-based systems are no longer sufficient for real-time fraud detection.

This project aims to:
- Build predictive models to identify fraudulent transactions.
- Compare the performance of Logistic Regression and Random Forest.
- Engineer useful features and optimize model performance using evaluation metrics.

👥 **Who is this project for?**
- Fraud Analysts & Risk Teams  
- Data Scientists in FinTech  
- Stakeholders aiming to reduce financial fraud losses

---

## II. 📂 Data Description

### 📌 Overview
- Source: Credit card transaction data (provided dataset)
- Format: CSV
- Rows: ~100,000+
- Target variable: `is_fraud` (binary classification)

<details>
<summary>📋 Feature Overview (Click to expand)</summary>

| Column | Description |
|--------|-------------|
| trans_date_trans_time | Date and time of transaction |
| cc_num | Credit card number (masked) |
| merchant | Name of merchant |
| category | Type of transaction |
| amount | Transaction amount |
| gender | Customer gender |
| job | Customer occupation |
| dob | Customer date of birth |
| city, state, zip | Customer location |
| long, lat | Customer geolocation |
| city_pop | Population of customer's city |
| is_fraud | Target variable (1 = Fraud, 0 = Not fraud) |

</details>

---

## III. ⚙️ Main Process

### 🧹 Data Preprocessing

- Dropped irrelevant columns (`Unnamed: 0`, `cc_num`, `street`, etc.)
- Converted date fields (`trans_date_trans_time`, `dob`) to datetime
- Checked missing values and duplicates (none significant)
- Transformed time to hour (`trans_hour`)
- Calculated user `age` from `dob`

### 🛠 Feature Engineering

- Encoded categorical variables: `category`, `gender`, `job`
- Removed highly granular or sensitive features (e.g., coordinates)
- Final dataset ready for model training

```python
df_encoded = pd.get_dummies(data, columns = ['category','gender','job'], drop_first=True)
```

---

### 🤖 Model Building

#### 📊 Data Split
```python
X_train, X_val, X_test = 70% / 15% / 15%
```

#### 🔄 Normalization
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMinMaxScaler()
```

#### 📈 Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)
```

#### 🌲 Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(max_depth=15, n_estimators=100)
rf_model.fit(X_train_scaled, y_train)
```

---

### 📏 Model Evaluation

#### Logistic Regression:
- Balanced Accuracy (Train): ~0.93  
- Balanced Accuracy (Validation): ~0.88  
- Performed reasonably but slightly underfit

#### Random Forest:
- Balanced Accuracy (Train): 1.0 (overfit risk)  
- Balanced Accuracy (Validation): ~0.91  
- Better at capturing fraud patterns

#### 🛠 Hyperparameter Tuning:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 100],
    'max_depth': [None, 15]
}
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='balanced_accuracy')
```
- Best Parameters: `max_depth=15`, `n_estimators=100`
- Final Accuracy (Test): ~0.93

---

## IV. 🧾 Final Conclusion & Recommendation

### ✅ Key Takeaways

- **Random Forest** showed superior performance over Logistic Regression, especially after tuning.
- **Feature Engineering** (e.g., `age`, `trans_hour`, encoded categories) played a critical role in model performance.
- **Balanced Accuracy** was a suitable metric due to class imbalance in fraud detection.

### 🧠 Recommendations

- Deploy the Random Forest model in batch or real-time fraud detection pipelines.
- Implement threshold tuning or probability calibration for production.
- Further improve with:
  - **SMOTE or oversampling** techniques
  - **SHAP values** for explainability
  - **Anomaly detection** models for unsupervised fraud spotting
