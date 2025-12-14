from sklearn.preprocessing import LabelEncoder

def encode_features(df):
    le = LabelEncoder()

    categorical_cols = df.select_dtypes(include='object').columns

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df