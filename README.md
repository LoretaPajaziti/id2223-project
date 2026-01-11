# Aurora Forecasting System for Sweden

This repository contains an end-to-end machine learning system for forecasting aurora-favorable geomagnetic conditions over Sweden. The project integrates data ingestion, feature engineering, model training, batch inference, and probabilistic visualization into a coherent ML pipeline.

The project was developed as part of ID2223 – Scalable Machine Learning and Deep Learning.

## Environment Setup

   # Create a conda or virtual environment for the project
    conda create -n book 
    conda activate book

    # Install 'uv' and 'invoke'
    pip install invoke dotenv

    # 'invoke install' installs python dependencies using uv and requirements.txt
    invoke install

## System Overview

The system is organized as a set of pipelines forming an end-to-end ML workflow.

### 1. Backfill Pipeline

The backfill pipeline constructs the historical dataset used for model development.

- Fetches historical geomagnetic data (Kp and Ap indices)
- Fetches historical local weather data for Sweden
- An additional data source based on solar activity (e.g. solar flux and sunspot indices) was tested during development to assess its predictive value.
  
### 2. Daily Feature Pipeline
The daily feature pipeline maintains an up-to-date feature set for inference.

- Fetches and updates the latest complete geomagnetic observations
- Fetches and updates the most recent local weather observations

### 3. Training Pipeline
The training pipeline builds time-aware binary classification models for multiple forecast horizons.

- Retrieves geomagnetic and weather features from versioned Hopsworks feature groups
- Joins features through a versioned Feature View
- Aggregates sub-daily Kp values into daily mean and maximum features
- Adds lagged Kp features for t−1, t−2, and t-3
- Ensures strict temporal ordering to prevent data leakage

For each forecast horizon h∈{1,…,5}, a separate model is trained to predict whether Apt+h≥15, representing aurora-favorable geomagnetic conditions.

Models are trained using XGBoost classifiers with explicit handling of class imbalance. Evaluation is performed using ROC AUC, accuracy, precision, recall, and F1-score. Diagnostic artifacts such as confusion matrices and feature importance plots are generated.

Each model is serialized and registered in the Hopsworks Model Registry.

### 4. Inference Pipeline
The inference pipeline generates forward-looking predictions using the trained models.

- Loads the appropriate model for each forecast horizon
- Consumes the most recent engineered features
- Produces probabilistic predictions for +1 to +5 days
- Exports predictions to a structured file (predictions.json)

### 5. Dashboard (Presentation Layer)
The dashboard serves as the final interpretation layer of the system.

- Displays recent geomagnetic and weather context using a navigable 5-day window
- Visualizes forecast probabilities across horizons
- Summarizes predictions using categorical risk levels
- Updates automatically whenever new inference outputs are generated

The dashboard can be accessed through: https://loretapajaziti.github.io/id2223-project/aurora/ 

<img width="533" height="474" alt="image" src="https://github.com/user-attachments/assets/02bf6095-1c77-404f-9e8d-7f0c47d3b507" />



