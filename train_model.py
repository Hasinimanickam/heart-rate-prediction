import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# Load dataset
df = pd.read_csv("heart_rate_data.csv", encoding="utf-8")

# Features and target
X = df.drop("Max Heart Rate During Exercise", axis=1)
y = df["Max Heart Rate During Exercise"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "heart_rate_model.pkl")

print("Model trained successfully!")
print("heart_rate_model.pkl created successfully!")