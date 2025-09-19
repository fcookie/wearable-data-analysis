import pandas as pd
from sklearn.model_selection import train_test_split
from src import DataLoader, FeatureEngineer, ActivityModel, Evaluator, Visualizer


# 1. Load Data
loader = DataLoader("data/processed/wearable_data.csv")
df = loader.load_data()
df = loader.preprocess(df)

# 2. Features
engineer = FeatureEngineer(df)
df = engineer.add_magnitude()

X = df[["accel_x", "accel_y", "accel_z", "magnitude"]]
y = df["label"]

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = ActivityModel()
model.train(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Evaluate
Evaluator.evaluate(y_test, y_pred)

# 6. Visualize
Visualizer.plot_confusion(y_test, y_pred, classes=model.model.classes_)
