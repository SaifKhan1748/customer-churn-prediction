# Customer Churn Prediction

## Setup
1. python -m venv .venv
2. source .venv/bin/activate
3. pip install -r requirements.txt

## Train
python src/train.py --data data/raw.csv --out models/churn_pipeline.joblib

## Predict (batch)
python src/predict.py --input data/new_customers.csv --model models/churn_pipeline.joblib --out predictions.csv

## Run demo
streamlit run src/app.py
