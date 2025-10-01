# src/train.py
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

from src.features import create_features

def build_pipeline(num_cols, cat_cols, scale_pos_weight=1.0):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preproc = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    clf = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
    model = Pipeline([
        ("preproc", preproc),
        ("clf", clf)
    ])
    return model

def main(data_path, model_out):
    df = pd.read_csv(data_path)
    df = create_features(df)

    TARGET = 'churn'   # change if your label has a different name
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in data.")

    X = df.drop(columns=[TARGET, 'customerID'], errors='ignore')
    y = df[TARGET].astype(int)

    # train/val split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # detect types
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # handle class imbalance via scale_pos_weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / max(1, pos)

    model = build_pipeline(num_cols, cat_cols, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)

    # Evaluate
    y_prob = model.predict_proba(X_val)[:,1]
    y_pred = model.predict(X_val)
    print("ROC AUC:", roc_auc_score(y_val, y_prob))
    print(classification_report(y_val, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_val, y_pred))

    # Save model
    joblib.dump(model, model_out)
    print(f"Model saved to {model_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw.csv", help="Path to raw CSV")
    parser.add_argument("--out", default="models/churn_pipeline.joblib", help="Path to save model")
    args = parser.parse_args()
    main(args.data, args.out)
