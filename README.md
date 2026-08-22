# Used Car Price Prediction Using Machine Learning

## Project Overview

This project develops a machine learning-based system for predicting the asking price of used cars.

The system accepts vehicle characteristics such as brand, model, age, kilometers driven, transmission, owner type, and fuel type and estimates the expected asking price.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Flask
- Joblib
- HTML
- CSS
- JavaScript

## Machine Learning Algorithms

The following regression algorithms were evaluated:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. CatBoost Regressor

Hyperparameter tuning was performed on the Random Forest model.

## Dataset

The dataset contains used-car listings with information including:

- Brand
- Model
- Manufacturing Year
- Vehicle Age
- Kilometers Driven
- Transmission
- Owner
- Fuel Type
- Asking Price

## Data Preprocessing

The following preprocessing operations were performed:

- Duplicate removal
- Missing-value handling
- Currency conversion
- Kilometer value conversion
- Removal of extreme mileage records
- Categorical feature encoding
- Train-test splitting

## Final Model

The tuned Random Forest Regressor was selected as the final model.

### Performance

- MAE: ₹215,107
- RMSE: ₹893,123
- R²: 0.6369

## Web Application

The trained model was integrated into a Flask web application.

Users can enter:

- Car brand
- Car model
- Vehicle age
- Kilometers driven
- Transmission
- Owner type
- Fuel type

The application then displays the estimated asking price.

## Project Structure

```text
UsedCarPricePrediction/
├── app.py
├── train_model.py
├── test_prediction.py
├── requirements.txt
├── dataset/
├── models/
├── templates/
├── static/
└── analysis/