# src/app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.features import create_features

MODEL_PATH = "models/churn_pipeline.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("Customer Churn Prediction Demo")

model = load_model()

uploaded = st.file_uploader("Upload CSV (customer rows)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    df_feat = create_features(df)
    X = df_feat.drop(columns=['churn','customerID'], errors='ignore')

    probs = model.predict_proba(X)[:,1]
    df['churn_prob'] = probs
    st.write("Top 20 by predicted churn probability")
    st.dataframe(df.sort_values('churn_prob', ascending=False).head(20))

    if st.button("Show Model ROC / Importance"):
        # Feature importance bar (from fitted XGB)
        try:
            clf = model.named_steps['clf']
            # feature names after preprocessing
            feat_names = model.named_steps['preproc'].get_feature_names_out()
            importances = clf.feature_importances_
            imp_df = pd.Series(importances, index=feat_names).sort_values(ascending=False)[:30]
            plt.figure(figsize=(6,8))
            imp_df.plot(kind='barh')
            plt.title("Top 30 feature importances")
            st.pyplot(plt.gcf())
        except Exception as e:
            st.error(f"Could not compute importance: {e}")

    if st.button("Compute SHAP summary (may take time)"):
        st.info("Computing SHAP values...")
        # transform X
        X_trans = model.named_steps['preproc'].transform(X)
        explainer = shap.TreeExplainer(model.named_steps['clf'])
        # shap_values shape: (n_samples, n_features)
        shap_values = explainer.shap_values(X_trans)
        feat_names = model.named_steps['preproc'].get_feature_names_out()
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_values, X_trans, feature_names=feat_names, show=False)
        st.pyplot(plt.gcf())
