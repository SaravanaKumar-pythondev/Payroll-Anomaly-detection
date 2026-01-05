from fastapi import FastAPI
import numpy as np
import pandas as pd
import joblib

# Load model & scaler
iso_forest = joblib.load("isolation_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(
    title="Payroll Anomaly Detection API",
    description="Real-time payroll anomaly detection using Isolation Forest",
    version="1.0"
)

@app.post("/predict")
def predict_anomaly(base_pay: float, overtime_pay: float, total_pay: float):
    
    # Convert input to DataFrame
    input_data = pd.DataFrame([{
        "BasePay": base_pay,
        "OvertimePay": overtime_pay,
        "TotalPay": total_pay
    }])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict anomaly
    prediction = iso_forest.predict(input_scaled)
    score = iso_forest.decision_function(input_scaled)
    
    # Interpret result
    anomaly_flag = int(prediction[0] == -1)
    
    if anomaly_flag:
        risk = "HIGH RISK"
    else:
        risk = "NORMAL"
    
    return {
        "anomaly_flag": anomaly_flag,
        "anomaly_score": float(score[0]),
        "risk_level": risk
    }

@app.get("/")
def root():
    return {"message": "Payroll Anomaly Detection API is running"}
