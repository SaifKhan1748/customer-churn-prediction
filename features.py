# src/features.py
import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with additional engineered features.
       This function is intentionally conservative (checks for column existence)."""
    df = df.copy()
    # Example: numeric conversions
    if 'tenure' in df.columns:
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')

    if 'monthly_charges' in df.columns:
        df['monthly_charges'] = pd.to_numeric(df['monthly_charges'], errors='coerce')

    # Tenure group (example)
    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(df['tenure'],
                                    bins=[-1, 12, 24, 48, 9999],
                                    labels=['0-12','13-24','25-48','49+'])

    # Usage change example (if you have two windows)
    if 'usage_last_30' in df.columns and 'usage_prev_30' in df.columns:
        df['usage_change'] = (df['usage_last_30'] + 1) / (df['usage_prev_30'] + 1)

    # Missing-value indicator for a couple important fields
    for col in ['monthly_charges', 'tenure']:
        if col in df.columns:
            df[f'is_null_{col}'] = df[col].isnull().astype(int)

    # You can add more domain-specific features here
    return df
