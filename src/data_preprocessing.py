import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Replace ? with NaN
    df.replace('?', np.nan, inplace=True)

    # Drop columns with too many missing values
    drop_cols = ['weight', 'payer_code', 'medical_specialty']
    df.drop(columns=drop_cols, inplace=True)

    # Create target variable
    df['readmitted'] = df['readmitted'].apply(
        lambda x: 1 if x == '<30' else 0
    )

    # Drop IDs
    df.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)

    return df