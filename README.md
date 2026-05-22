# Customer Churn Prediction Pipeline

## Project Overview

This project implements an end-to-end machine learning pipeline for customer churn prediction.  
The pipeline covers data ingestion, preprocessing, model training, prediction generation, drift monitoring, and dashboard visualization.

## Tech Stack

- Python 3.12
- SQLite
- Scikit-learn
- Supabase
- Data Studio (Previously known as Looker Studio)
- Evidently AI

## Environment Setup

This project was developed using Python 3.12.

### Create virtual environment

```bash
python3.12 -m venv .venv
```

### Activate virtual environment

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
churn_pipeline/
├── Data/
│   └── Customer-Churn-Records.csv      # Raw dataset
├── Database/
│   └── churn_database.db               # Local SQLite database
├── Models/
│   └── final_model.pkl                 # Trained production model
├── Notebooks/
│   ├── exploration.ipynb               # Exploratory Data Analysis (EDA)
│   └── modeling_experiments.ipynb      # Model experimentation and tuning
├── src/
│   ├── ingest_data.py                  # Creates SQLite database from CSV
│   ├── preproccess.py                  # Data cleaning and preprocessing
│   ├── train_model.py                  # Model training and serialization
│   ├── generate_predictions.py         # Prediction generation and Supabase upload
│   └── evaluate_drift.py               # Data drift evaluation using Evidently AI
├── requirements.txt                    # Python dependencies
├── .env                                # Environment variables (excluded from Git)
├── .gitignore                          # Git ignored files and folders
└── README.md                           # Project documentation
```

## Pipeline Flow

1. ingest_data.py - raw CSV ingestion into SQLite
2. exploration.ipynb - EDA
3. preproccess.py - data preprocessing and feature engineering
4. modeling_experiments.ipynb - model experimentation and evaluation
5. train_model.py - final model training
6. generate_predictions.py - prediction generation
7. evaluate_drift.py - drift monitoring
8. Dashboard visualization - in google data studio

## Important Notes

- The dataset consists of cross-sectional data rather than time-series data
- No temporal customer behavior is available
- Drift analysis is simulated due to the static nature of the dataset