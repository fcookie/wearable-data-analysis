# Wearable Data Analysis for Rehabilitation Monitoring

## Overview
This project demonstrates how to use **wearable sensor data** (e.g., accelerometers or gyroscopes) to analyze and classify patient activity.  
The goal is to create a **machine learning pipeline** that predicts activity type or rehabilitation progress from wearable signals.  

## Workflow
1. Load time-series sensor data (accelerometer).  
2. Clean and preprocess the dataset.  
3. Extract features (mean, variance, peaks, signal energy).  
4. Train ML models (Logistic Regression, Random Forest).  
5. Evaluate accuracy and visualize results.  

## Clinical Relevance
Wearables are increasingly used in **digital health** and **rehabilitation monitoring**.  
This prototype shows how ML can:  
- Track patient activity remotely.  
- Detect improvement in movement quality.  
- Support clinicians with **objective metrics** instead of only self-reports.  

## Tech Stack
- **Python 3.9+**  
- **Pandas, NumPy** (data handling)  
- **scikit-learn** (ML models)  
- **Matplotlib, Seaborn** (visualizations)  

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook wearable_data_analysis.ipynb
