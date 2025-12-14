from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import encode_features
from src.model_utils import train_model

data_path = 'data/raw/diabetic_data.csv'

df = load_data(data_path)
df = preprocess_data(df)
df = encode_features(df)

df.to_csv('data/processed/cleaned_data.csv', index=False)

train_model(df)