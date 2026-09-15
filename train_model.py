
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "ecommerce_customer_value.csv"

df = pd.read_csv(DATA)
X = df.drop(columns=["customer_id", "customer_lifetime_value"])
y = df["customer_lifetime_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200, random_state=42, max_depth=10
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("E-commerce Customer Value Prediction")
print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
print(f"RMSE: {mean_squared_error(y_test, pred) ** 0.5:.2f}")
print(f"R2 Score: {r2_score(y_test, pred):.4f}")

joblib.dump(model, BASE / "ecommerce_customer_value_model.pkl")

importance = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(9, 5))
importance.plot(kind="bar")
plt.title("Feature Importance - Customer Value Prediction")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(BASE / "feature_importance.png")
print("Model and feature importance chart saved.")
