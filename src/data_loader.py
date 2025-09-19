import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        return pd.read_csv(self.filepath)

    def preprocess(self, df):
        # Example: fill missing values
        return df.fillna(0)
