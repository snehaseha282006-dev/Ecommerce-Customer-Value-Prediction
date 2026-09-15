
from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
model = joblib.load(BASE / "ecommerce_customer_value_model.pkl")

print("Enter customer details:")
age = int(input("Age: "))
income = float(input("Annual income: "))
orders = int(input("Total orders: "))
avg_order = float(input("Average order value: "))
recency = int(input("Days since last purchase: "))
discount = float(input("Discount usage percentage: "))
tickets = int(input("Support tickets: "))
clicks = int(input("Email clicks: "))

sample = pd.DataFrame([{
    "age": age,
    "annual_income": income,
    "total_orders": orders,
    "average_order_value": avg_order,
    "days_since_last_purchase": recency,
    "discount_usage_percent": discount,
    "support_tickets": tickets,
    "email_clicks": clicks
}])

prediction = model.predict(sample)[0]
print(f"\nPredicted Customer Lifetime Value: {prediction:.2f}")
