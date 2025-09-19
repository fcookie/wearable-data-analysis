import numpy as np
import pandas as pd

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def add_magnitude(self):
        self.df["magnitude"] = np.sqrt(
            self.df["accel_x"]**2 + self.df["accel_y"]**2 + self.df["accel_z"]**2
        )
        return self.df
