# Payroll-Anomaly-detection
Built an end-to-end Payroll Anomaly Detection system using Isolation Forest to identify suspicious salary and overtime patterns. The solution supports batch analysis and real-time detection via FastAPI, with model persistence and concept drift monitoring for long-term reliability.
# Payroll Anomaly Detection System

## Overview

This project implements an end-to-end **Payroll Anomaly Detection System** using an unsupervised machine learning approach. The goal is to identify suspicious salary and overtime patterns—such as salary manipulation or fake overtime—without relying on labeled data.

The solution is designed to be **practical, scalable, and production-oriented**, supporting both **batch analysis** and **real-time anomaly detection**.

---

## Key Features

* Unsupervised anomaly detection using **Isolation Forest**
* Batch processing on historical payroll data (CSV-based)
* Real-time anomaly detection via **FastAPI**
* Model persistence using **joblib (.pkl files)**
* Concept drift monitoring based on anomaly rate deviation
* Clear separation between training and deployment layers

---

## Dataset

The project uses a San Francisco public salary dataset enhanced with **synthetically injected anomalies** (≈2%) to simulate abnormal payroll behavior.

**Key Features Used:**

* BasePay
* OvertimePay
* TotalPay

Injected anomaly labels are used **only for evaluation and validation**. The Isolation Forest model itself remains fully unsupervised.

---

## Model & Methodology

* **Algorithm**: Isolation Forest
* **Learning Type**: Unsupervised
* **Contamination Factor**: 0.02 (expected anomaly rate)

Isolation Forest isolates anomalies by randomly partitioning feature space. Records that are isolated quickly are considered anomalous.

---

## Batch Anomaly Detection

The notebook (`payroll_anomaly_detection.ipynb`) handles:

* Data loading and preprocessing
* Feature scaling
* Model training and evaluation
* Anomaly visualization
* Concept drift detection

Batch predictions produce:

* `anomaly_label` → 1 (Normal), -1 (Anomaly)
* `anomaly_score` → Degree of abnormality

---

## Real-Time API (FastAPI)

A real-time inference API is implemented in `api.py` using **FastAPI**.

### Endpoint

* `POST /predict`

### Input Parameters

* `base_pay` (float)
* `overtime_pay` (float)
* `total_pay` (float)

### Output

* Anomaly flag
* Anomaly score
* Risk level (NORMAL / HIGH RISK)

Interactive testing is available via **Swagger UI**:

```
http://127.0.0.1:8000/docs
```

---

## Model Artifacts

* `isolation_forest_model.pkl`: Serialized trained model
* `scaler.pkl`: Serialized feature scaler

These artifacts ensure consistent preprocessing and inference between batch training and real-time deployment.

---

## Concept Drift Handling

Concept drift is monitored by tracking the **model’s anomaly rate over time**.

* Expected anomaly rate is derived from the contamination factor
* Significant deviation triggers a drift alert
* Drift mitigation includes retraining, sliding-window learning, and threshold recalibration

This approach ensures long-term reliability in dynamic payroll environments.

---

## How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Batch Analysis

Open and run:

```
payroll_anomaly_detection.ipynb
```

### 3. Start Real-Time API

```bash
uvicorn api:app --reload
```

---

## Conclusion

This project delivers a complete payroll anomaly detection framework that combines **analytical robustness** with real-world deployment practices. It demonstrates how unsupervised learning, real-time APIs, and drift monitoring can be integrated into a scalable and production-ready system.

---
