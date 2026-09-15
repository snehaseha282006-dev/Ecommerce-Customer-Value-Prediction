# E-commerce Customer Value Prediction

## Description
This machine learning project predicts the estimated customer lifetime value of an e-commerce customer.

## Technology
- Python
- Pandas
- Scikit-learn
- Random Forest Regressor
- Joblib
- Matplotlib

## Features
- Age
- Annual income
- Total orders
- Average order value
- Days since last purchase
- Discount usage percentage
- Support tickets
- Email clicks

## Files
- `train_model.py` - trains and evaluates the model
- `predict.py` - predicts customer lifetime value for a new customer
- `data/ecommerce_customer_value.csv` - synthetic dataset
- `ecommerce_customer_value_model.pkl` - trained model
- `feature_importance.png` - feature importance chart
- `requirements.txt` - required libraries

## Run
```bash
pip install -r requirements.txt
python train_model.py
python predict.py
```

## Note
The dataset is synthetic and intended for educational/project demonstration purposes only.
