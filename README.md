# Credit Risk Modeling and Loan Approval Classification using ML

## 🔍 Project Overview
This is an end-to-end machine learning project that predicts bank loan approval using a real-world Kaggle credit-risk dataset. The project demonstrates the complete data science workflow—from data exploration and cleaning to feature engineering, model training, and evaluation. The final model achieves approximately **92% test accuracy**  on a dataset of **45,000 loan applications**.

**Use Case:**  
This project can help banks automate and accelerate loan approval decisions while maintaining risk management through data-driven credit scoring.

---

## 📊 Dataset

### Source
Loan Approval Classification Dataset – Kaggle

### Dataset Statistics
- **Records:** 45,000 loan applications  
- **Features:** 14 variables (6 float, 3 integer, 5 categorical)  
- **Target Variable:** `loan_status` (1 = approved, 0 = rejected)  
- **Class Distribution:** ~22% approved, ~78% rejected (imbalanced)  
- **Missing Values:** None  

### Feature Descriptions

| Feature | Type | Description |
|------|------|------------|
| person_age | Float | Age of applicant (20–144 years) |
| person_gender | Categorical | Gender (Male/Female) |
| person_education | Categorical | Highest education level |
| person_income | Float | Annual income ($8K–$7.2M) |
| person_emp_exp | Integer | Years of employment experience |
| person_home_ownership | Categorical | Home status (Rent/Own/Mortgage) |
| loan_amnt | Float | Loan amount requested ($500–$35K) |
| loan_intent | Categorical | Purpose of loan |
| loan_int_rate | Float | Interest rate assigned (5.42%–20%) |
| loan_percent_income | Float | Loan as % of annual income |
| cb_person_cred_hist_length | Float | Credit history length (years) |
| credit_score | Integer | Credit score (390–850) |
| previous_loan_defaults_on_file | Categorical | Prior defaults (Yes/No) |
| loan_status | Integer | Target: Loan approval (1/0) |

---

## 🔄 Project Pipeline

### 1. Data Loading & Exploration
- Loaded 45,000 records with no missing values  
- Analyzed data types and distributions  
- Identified class imbalance (22% approved vs. 78% rejected)  
- Visualized feature distributions using histograms and boxplots  

### 2. Data Cleaning & Outlier Treatment
- Used **IQR (Interquartile Range)** method for outlier detection  
- Applied **1.5 × IQR** bounds for capping extreme values  
- Preserved `loan_int_rate` due to normal distribution  
- Capped outliers in:
  - person_age  
  - person_income  
  - person_emp_exp  
  - loan_amnt  
  - loan_percent_income  
  - cb_person_cred_hist_length  
  - credit_score  

### 3. Feature Engineering
Created 6 derived features:
- **age_group**: Young, Mid, Older  
- **income_bracket**: Very_Low → Very_High (quantiles)  
- **emp_exp_category**: No_Experience, Little_Experience, Experienced  
- **credit_score_category**: Poor, Fair, Good, Excellent  
- **high_risk_loan**: High interest + high debt-to-income  
- **debt_income_risk**: Risk tiers from loan_percent_income  

### 4. Data Preprocessing
- **Encoding:** Label Encoding for categorical features  
- **Scaling:** StandardScaler for numerical features  
- **Split:** 80–20 stratified train-test split (random_state=42)  
- **Training Shape:** 36,000 × 19 features  

---

## 🤖 Model Building & Comparison

| Model | CV Accuracy | Test Accuracy | Notes |
|----|----|----|----|
| Logistic Regression | 89.41% ± 0.88% | 89.31% | Baseline |
| Random Forest | 92.36% ± 0.29% | 92.25% | ✅ Best |
| Gradient Boosting | 92.13% ± 0.36% | 92.09% | Strong performer |

**Winner:** Random Forest Classifier

---

## ⚙️ Hyperparameter Tuning
Bayesian Optimization using **Optuna**

**Search Space**
- n_estimators: 10–200  
- max_depth: 2–30  
- min_samples_split: 2–10  
- min_samples_leaf: 1–10  
- max_features: sqrt, log2, None  

**Best Parameters**
- n_estimators: 135  
- max_depth: 19  
- min_samples_split: 5  
- min_samples_leaf: 8  
- max_features: None  

**Optimized CV Accuracy:** ~92.52%

---

## 📈 Key Results & Insights

### Final Model Performance
- **Test Accuracy:** ~92.3%  
- **Cross-Validation Accuracy:** ~92.5% ± 0.29%  
- **Stable generalization across folds**

### Top Feature Drivers
- credit_score  
- loan_percent_income  
- loan_int_rate  
- person_income  
- high_risk_loan  

### Business Insights
- Low credit score + high debt ratio → high rejection probability  
- Short credit history reduces approval certainty  
- Previous defaults are strong negative indicators  
- Loan purpose has lower predictive impact  

---

## 📁 Project Structure
