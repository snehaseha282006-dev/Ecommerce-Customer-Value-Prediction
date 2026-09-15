# Ecommerce-Customer-Value-Prediction
Machine learning project that predicts E-commerce Customer Lifetime Value (CLV) using a Random Forest Regression model with Scikit-learn.
 E-commerce Customer Lifetime Value Prediction

A machine learning project that predicts the Customer Lifetime Value (CLV) of e-commerce customers using a Random Forest Regression model built with Scikit-learn.

The project uses customer demographic, purchasing, engagement, and support-related information to estimate the expected lifetime value of each customer.

 Project Overview

Customer Lifetime Value (CLV) is an important business metric that estimates the total value a customer may generate over their relationship with an e-commerce business.

This project builds a supervised machine learning regression model to predict customer lifetime value based on historical customer characteristics and behavior.

The trained model can also be used to make predictions for new customers through an interactive command-line interface.

 Objectives
Predict customer lifetime value using machine learning.
Analyze factors that contribute to customer value.
Train and evaluate a Random Forest Regression model.
Save the trained model for future predictions.
Provide a simple script for predicting CLV for new customers.
Visualize feature importance.
 Machine Learning Approach
Algorithm

Random Forest Regressor

The model is configured with:

n_estimators = 200
max_depth = 10
random_state = 42

The dataset is divided into:

80% Training Data
20% Testing Data
 Dataset

The project contains a synthetic e-commerce customer dataset with 400 records.

Features
Feature	Description
age	Customer age
annual_income	Customer's annual income
total_orders	Total number of orders placed
average_order_value	Average value of customer orders
days_since_last_purchase	Number of days since the customer's last purchase
discount_usage_percent	Percentage of purchases using discounts
support_tickets	Number of customer support tickets
email_clicks	Number of email interactions/clicks
Target Variable

customer_lifetime_value

The target represents the estimated lifetime value of the customer.

 Model Performance

Using the current dataset and model configuration, the Random Forest model achieves approximately:

Metric	Score
MAE	225.27
RMSE	316.65
R² Score	0.9602

An R² score of approximately 0.96 indicates that the model explains a large proportion of the variation in customer lifetime value on the test dataset.

Note: Results can vary if the dataset, train/test split, or model parameters are changed.

 Feature Importance

The model also calculates the relative importance of each input feature.

The most influential features in the current model are:

Average Order Value — ~44.60%
Total Orders — ~43.71%
Annual Income — ~8.49%
Days Since Last Purchase — ~0.83%
Discount Usage Percentage — ~0.78%
Email Clicks — ~0.66%
Age — ~0.59%
Support Tickets — ~0.35%

This suggests that average order value and total orders are the strongest predictors of customer lifetime value in this dataset.

The repository includes feature_importance.png for visualizing these results.

 Project Structure
Ecommerce_Customer_Value_Prediction_Sklearn/
│
├── data/
│   └── ecommerce_customer_value.csv
│
├── ecommerce_customer_value_model.pkl
├── feature_importance.png
├── predict.py
├── train_model.py
├── requirements.txt
└── README.md
 Technologies Used
Python
Pandas — data loading and manipulation
Scikit-learn — machine learning and evaluation
Random Forest Regressor — regression algorithm
Joblib — model serialization
Matplotlib — feature importance visualization
 Getting Started
1. Clone the Repository
git clone https://github.com/your-username/Ecommerce_Customer_Value_Prediction_Sklearn.git

Navigate to the project directory:

cd Ecommerce_Customer_Value_Prediction_Sklearn
2. Install Dependencies
pip install -r requirements.txt
3. Train the Model

Run:

python train_model.py

This will:

Load the dataset.
Separate features and target.
Split the data into training and testing sets.
Train the Random Forest Regressor.
Evaluate the model using MAE, RMSE, and R².
Save the trained model as ecommerce_customer_value_model.pkl.
Generate the feature importance chart.
 Make a Prediction

Run:

python predict.py

The program will ask for customer information such as:

Age:
Annual income:
Total orders:
Average order value:
Days since last purchase:
Discount usage percentage:
Support tickets:
Email clicks:

It then returns the predicted customer lifetime value:

Predicted Customer Lifetime Value: 3500.25
 Example Input Features

A customer prediction is generated using the following inputs:

Age
Annual Income
Total Orders
Average Order Value
Days Since Last Purchase
Discount Usage Percentage
Support Tickets
Email Clicks

These values are passed to the trained Random Forest model to estimate the customer's lifetime value.

 Business Applications

Customer lifetime value prediction can help e-commerce businesses:

Identify high-value customers.
Improve customer segmentation.
Develop personalized marketing strategies.
Allocate marketing budgets more effectively.
Identify customers with strong purchasing potential.
Support customer retention strategies.
Understand the factors associated with customer value.
 Limitations
The dataset used in this project is synthetic.
The model is intended primarily for educational and demonstration purposes.
Model performance on real-world e-commerce data may differ significantly.
The current implementation does not include categorical features or advanced feature engineering.
Further validation would be required before using the model for real business decisions.
 Possible Future Improvements
Use a larger real-world customer dataset.
Add customer purchase history and transaction-level features.
Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
Compare Random Forest with XGBoost, Gradient Boosting, and other regression algorithms.
Add cross-validation.
Build an interactive Streamlit web application.
Add customer segmentation using clustering.
Create a REST API for model predictions.
Add automated model evaluation and retraining.
Use explainable AI techniques such as SHAP for deeper model interpretation.
 License

This project is intended for educational and portfolio purposes.

