# 💳 Credit Card Fraud Detection - Data Engineering Project

## 📌 Overview

This project focuses on **credit card transaction fraud detection** within the domain of financial analytics. The current goal is to build a reliable analytical dataset that supports data exploration, preprocessing, and future machine learning modeling.

---

## 🎯 Objectives - Week 3

* Build a clean and structured dataset for fraud analysis
* Design an analytical schema (fact + dimension tables)
* Perform data preprocessing and feature engineering
* Prepare the foundation for a binary classification model

---

## 📂 Dataset Information

* **Source:** Kaggle - Credit Card Transactions Fraud Detection Dataset
* **Files:**

  * `fraudTrain.csv` (~352 MB)
  * `fraudTest.csv` (~150 MB)
* **Total Size:** ~502 MB
* **Records:** 1,852,394 rows
* **Time Range:** Jan 2019 – Dec 2020

🔗 [https://www.kaggle.com/datasets/kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

---

## 🧱 Data Architecture

### ⭐ Fact Table

* **fact_transactions**

  * Primary Key: `trans_num`
  * Includes: amount, timestamp, merchant, customer, category, `is_fraud`

### 📊 Dimension Tables

* **dim_cards** (`cc_num`)
* **dim_customers** (`customer_id`)
* **dim_merchants** (`merchant_id`)
* **dim_time** (`date_key`)

### 🔗 Relationships

* `fact_transactions.cc_num → dim_cards.cc_num`
* `fact_transactions.customer_id → dim_customers.customer_id`
* `fact_transactions.merchant_id → dim_merchants.merchant_id`
* `fact_transactions.date_key → dim_time.date_key`

---

## ⚙️ Data Preprocessing (V1)

### ✔️ Steps Performed

* Merged training and test datasets
* Added column `type` (train/test)
* Removed technical column `Unnamed: 0`
* Converted datetime fields
* Standardized data types

### 🧠 Feature Engineering

* Transaction hour
* Day of the week
* Month
* Customer age (from DOB)
* Distance (customer ↔ merchant)

### 🔐 Data Privacy

Removed sensitive attributes:

* Name (`first`, `last`)
* Address (`street`)
* Credit card number (`cc_num`)

---

## 📘 Data Dictionary (Key Fields)

| Column                | Description          | Role              |
| --------------------- | -------------------- | ----------------- |
| trans_num             | Transaction ID       | Primary Key       |
| trans_date_trans_time | Timestamp            | Temporal feature  |
| amt                   | Transaction amount   | Numerical feature |
| merchant              | Merchant name        | Categorical       |
| category              | Purchase category    | Categorical       |
| city, state           | Location             | Geographic        |
| lat, long             | Coordinates          | Geospatial        |
| merch_lat, merch_long | Merchant coordinates | Geospatial        |
| dob                   | Date of birth        | Used for age      |
| is_fraud              | Fraud label          | Target variable   |

---

## 📈 Data Analysis

* **No missing values** detected
* **No duplicate rows** found
* **Class imbalance:**

  * Fraud cases ≈ **0.58%**

⚠️ This imbalance must be handled in future modeling (e.g., SMOTE, resampling)

---

## ⚖️ Ethical Considerations

Although the dataset is simulated, it includes sensitive-like data:

* Personal identifiers
* Geographic data
* Financial information

### ✅ Best Practices

* Data anonymization
* Avoid re-identification
* Use only for academic purposes
* Follow Kaggle license terms

---

## 🚀 Future Work

* Handle class imbalance
* Train classification models (e.g., Random Forest, XGBoost)
* Evaluate performance (Precision, Recall, F1-score)
* Deploy pipeline for fraud detection

---

## 👨‍💻 Authors

* Karim Wagner Samanamud Mosquera
* Ian Joaquin Sanchez Alva
* Ronaldo David Cornejo Valencia

---

## 🏫 Institution

**Universidad Peruana de Ciencias Aplicadas (UPC)**
Major: Computer Science
Course: Big Data

---

## 📌 License

The dataset used in this project is licensed under **CC0: Public Domain**.

This means:

* The data is free to use, modify, and distribute
* No attribution is required
* It can be used for academic and commercial purposes

However, as a best practice, this project still promotes:

* Responsible data usage
* Ethical handling of sensitive-like information
* Proper documentation of data sources
