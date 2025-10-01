# src/predict.py
import argparse
import pandas as pd
import joblib
from src.features import create_features

def batch_predict(input_csv, model_path, output_csv):
    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    df_feat = create_features(df)
    X = df_feat.drop(columns=['churn', 'customerID'], errors='ignore')
    probs = model.predict_proba(X)[:,1]
    df['churn_probability'] = probs
    df.to_csv(output_csv, index=False)
    print(f"Predictions written to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="models/churn_pipeline.joblib")
    parser.add_argument("--out", default="predictions.csv")
    args = parser.parse_args()
    batch_predict(args.input, args.model, args.out)
