# 🚀 End-to-End ML Data Engineering Project with Snowflake

## E-Commerce Customer Analytics & Churn Prediction Pipeline

---

## 📋 Project Overview

Build a production-ready data engineering pipeline using **Snowflake** to process e-commerce data, create ML features, and deploy predictive models. This project demonstrates advanced Snowflake capabilities including **Snowpark**, **ML UDFs**, and **real-time analytics**.

**Business Case:** An e-commerce company needs to predict customer churn and optimize marketing campaigns using real-time transaction data.

---


## 🛠️ Required Downloads & Setup

### 1. Snowflake Trial Account

* Sign up at: [https://signup.snowflake.com/](https://signup.snowflake.com/)
* Choose: AWS | US East (N. Virginia) | Standard Edition (Free Tier)

### 2. Python Environment Setup

```bash
python3.10 -m venv snowflake_ml_env
snowflake_ml_env\Scripts\activate

pip install snowflake-connector-python
pip install snowflake-snowpark-python
pip install pandas numpy scikit-learn xgboost
pip install streamlit plotly faker
```

### 3. VS Code Extensions

* Snowflake Extension
* Python
* Jupyter

---

## 📊 Project Architecture

```
Raw Data (CSV) ➔ Snowflake Stage ➔ Raw Tables ➔
Transformed Tables ➔ Feature Store (Snowpark) ➔
ML Models (UDFs) ➔ Streamlit Dashboard
```

---

## 🌟 Phase 1: Data Generation & Ingestion

1. Generate synthetic users, products, and transactions using `data_generator.py`
2. Save to CSV and upload to Snowflake stage
3. Create raw tables in Snowflake and copy data from stage

---

## 🔄 Phase 2: Data Loading & Transformation

1. Use `data_loader.py` to PUT CSVs into Snowflake and populate raw tables
2. Use `data_transformation.py` (Snowpark) to create `FEATURES.USER_FEATURES`
3. Generate churn labels based on recent transaction activity

---

## 🤖 Phase 3: ML Model Development

1. Use `model_training.py` to train churn model on features using Random Forest or XGBoost
2. Evaluate with classification report and save model using `joblib`

---

## 💾 Phase 4: Model Deployment

1. Deploy model as **Snowflake UDF** using `deploy_model_udf.py`
2. Register both `predict_churn` and `predict_churn_probability` functions
3. Test using SQL queries and create prediction view `ML_MODELS.CUSTOMER_CHURN_PREDICTIONS`

---

## 📊 Phase 5: Analytics Dashboard

1. Use `dashboard.py` to build an interactive Streamlit dashboard
2. Visualize:

   * Customer segmentation
   * Revenue by category
   * Churn analysis by segment
   * List of high-risk customers

---

## 📆 Phase 6: Automation

* Schedule daily runs with `automated_pipeline.py`
* Integrate feature generation, model retraining, and UDF updates

---

## 📅 Testing & Validation

* Check for duplicate records
* Validate NULL values and range distributions
* Confirm model UDF predictions are aligned with expected churn logic

---

## 💼 Key Skills Demonstrated

* Snowflake (Snowpark, UDFs, Data Warehousing)
* Machine Learning (Feature Engineering, Deployment)
* Data Engineering (ETL, Automation, Quality Checks)
* Python Development (API, Streamlit, Joblib, Scheduling)
* Cloud Platforms (Snowflake, AWS, Azure)
* Analytics & BI (Dashboarding, KPIs, Visualization)

---

## 📈 Expected Outcomes

* 100K+ transactions processed daily
* 89% model accuracy for churn prediction
* Real-time scoring with Snowflake UDFs
* Fully automated and tested pipeline

> This project demonstrates **enterprise-level data engineering and ML deployment** on Snowflake and is aligned with real-world business use cases
