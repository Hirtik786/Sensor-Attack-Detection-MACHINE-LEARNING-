import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(path):
    df = pd.read_csv(path)
    le = LabelEncoder()
    df['source_type'] = le.fit_transform(df['source_type'])
    df['source_location'] = le.fit_transform(df['source_location'])
    df['operation'] = le.fit_transform(df['operation'])
    df['wind_direction'] = le.fit_transform(df['wind_direction'])
    df['attack'] = df['attack'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    return df
