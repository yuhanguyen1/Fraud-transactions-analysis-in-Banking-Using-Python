# 🧠 Fraud transactions analysis in Banking using Python

<img width="1440" height="754" alt="Fraudulent-transactions" src="https://github.com/user-attachments/assets/b592d503-12cf-4935-8b61-79ebf7382471" />

**Author**: Nguyen Anh Huy  
**Date**: 13/07/2025  
**Tools Used**: Python (Jupyter Notebook)

---

## 📚 Table of Contents  
[I. 🎯 Background & Overview]()

[II. 📂 Data Description]() 

[III. ⚒️ Main Process]()  

[IV. 🧾 Final Conclusion & Recommendations]()

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

## III. ⚒️ Main Process

### 🧹 1. Data Preprocessing

##### 📌 Objective:
Prepare and transform the raw transaction data into a clean, structured format suitable for Machine Learning modeling.

---

#### 🔻 Drop Irrelevant Columns

```python
exclude_cols = ['trans_date_trans_time', 'cc_num','first','last','dob','trans_num','unix_time',
               'long','lat','merch_lat','merch_long','street','city','state','zip','city_pop']
data.drop(columns = exclude_cols, inplace=True)
```

> ✅ These columns either contain personal identifiable information (PII), or are irrelevant for fraud detection modeling (e.g., `street`, `cc_num`, `first name`). Dropping them helps reduce noise and dimensionality.

---

#### 📆 Convert Date Columns to Datetime Format

```python
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['dob'] = pd.to_datetime(data['dob'])
```

> ✅ Converting to `datetime` format allows feature engineering like extracting transaction hour and computing user age.

---

#### 🔍 Missing Value Check

```python
missing_dict = {'volume': data.isnull().sum(), 
                'missing_percentage': data.isnull().sum()/data.shape[0]*100}
missing_df = pd.DataFrame(missing_dict)
```

> ✅ Verifies data completeness. The dataset had no critical missing values, so no imputation or row dropping was required.

---

#### 🔁 Duplicate Check

```python
duplicate_count = data.duplicated().sum()
print(f'There are {duplicate_count} duplicate values')
```

> ✅ Ensures data integrity by checking for repeated records. No significant duplicates were found.

---

#### ⏰ Create Transaction Hour Feature

```python
data['trans_hour'] = data['trans_date_trans_time'].dt.hour
```

> ✅ Hour of transaction can be a strong fraud indicator (e.g., fraudulent transactions often occur at unusual hours).

---

#### 🎂 Calculate Customer Age

```python
data['age'] = (data['trans_date_trans_time'] - data['dob']).dt.days / 365
data['age'] = round(data['age'])
```

> ✅ Age can influence transaction behavior and fraud patterns. Older vs. younger users may exhibit different risk profiles.

---

#### 🧪 Feature Selection

```python
# Already handled above with drop()
```

> ✅ Final dataset now only includes relevant engineered and encoded features for modeling.

---

#### 🧬 One-Hot Encoding Categorical Variables

```python
list_column = ['category','gender','job']
df_encoded = pd.get_dummies(data, columns = list_column, drop_first=True)
df_encoded.head(2)
```

> ✅ Converts categorical features into numerical format suitable for machine learning models. `drop_first=True` prevents multicollinearity.

---

> 🎯 **Result**: The cleaned and transformed dataset is now ready for normalization and model training, with well-engineered features like `trans_hour`, `age`, and encoded demographic/transaction variables.

---

### 🤖 2. Model Building

#### 📊 Data Split
```python
# Split train/test/validate set
from sklearn.model_selection import train_test_split
X = df_encoded.drop(columns='is_fraud', axis = 1)
y = df_encoded['is_fraud']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Number data of train set: {len(X_train)}")
print(f"Number data of validate set: {len(X_val)}")
print(f"Number data of test set: {len(X_test)}")
```

#### 🔄 Normalization
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

#### 📈 Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state=0)
lr_model.fit(X_train_scaled, y_train)

y_pred_val = lr_model.predict(X_val_scaled)
y_pred_train = lr_model.predict(X_train_scaled)
```

#### 🌲 Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(max_depth=15, random_state=0, n_estimators=100)
rf_model.fit(X_train_scaled, y_train)

y_rf_pred_val = rf_model.predict(X_val_scaled)
y_rf_pred_train = rf_model.predict(X_train_scaled)
```

---

### 📏 Model Evaluation

#### 🔹 Logistic Regression

**📌 Goal:**  
Use Logistic Regression as a simple baseline model to evaluate its ability in detecting fraudulent transactions.

**📊 Results:**
- **Balanced Accuracy (Train):** ~0.69  
- **Balanced Accuracy (Validation):** ~0.69  

**📝 Notes:**
- The model shows consistent performance between training and validation sets, indicating **no overfitting**.
- However, the overall accuracy is relatively low, suggesting the model lacks the capacity to capture complex, non-linear fraud patterns.
- Suitable as a baseline, but insufficient for real-world fraud detection where accuracy is critical.

---

#### 🔹 Random Forest

**📌 Goal:**  
Apply Random Forest to capture more complex relationships in the data and improve fraud classification accuracy.

**📊 Results:**
- **Balanced Accuracy (Train):** ~0.76  
- **Balanced Accuracy (Validation):** ~0.74  

**📝 Notes:**
- Performs noticeably better than Logistic Regression on both train and validation sets.
- Slight drop in validation accuracy compared to training suggests **mild overfitting**, but still acceptable.
- Provides a more powerful, generalizable model for fraud detection.

---

### 🔧 Hyperparameter Tuning (Grid Search)

**📌 Objective:**  
To fine-tune Random Forest performance using a **Grid Search** with 5-fold cross-validation.

**📦 Parameter Grid:**
```python
param_grid = {
    'n_estimators': [10, 100],
    'max_depth': [None, 15]
}
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='balanced_accuracy')
grid_search.fit(X_train, y_train)
```
**✅ Best Parameters Found:** {'max_depth': none, 'n_estimators': 100}

**🎯 Performance of Tuned Model on Test Set:**

+ Accuracy: ~0.99

+ R² Score: ~0.85

---

## IV. 🧾 Final Conclusion & Recommendations

### 🧠 Summary:
- Two models were evaluated on the task of fraud transaction classification: **Logistic Regression** and **Random Forest**.
- **Random Forest** demonstrated higher accuracy and better generalization compared to Logistic Regression.

### 💡 Final Conclusions:
- **Logistic Regression** is useful as a baseline but not recommended for deployment due to its lower performance.
- **Random Forest** is the better choice for this dataset, offering higher balanced accuracy and stronger predictive capability.
- The accuracy scores indicate that while the model is functional, further improvement is needed before deployment in a high-risk environment.

### ✅ Recommendations:
- **Use Random Forest** as the primary model for fraud classification in the current pipeline.
- **Tune hyperparameters** further (e.g. number of estimators, max depth) and test additional models like **XGBoost** or **LightGBM** for improved performance.
- **Investigate class imbalance** and consider techniques like **SMOTE** to further boost detection of rare fraud cases.
- Continuously **retrain the model with fresh transaction data** to adapt to evolving fraud tactics.
- Explore **feature importance** from the Random Forest model to understand key fraud signals and improve fraud prevention strategies.
