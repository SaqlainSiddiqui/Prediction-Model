import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD FEATURE-ENGINEERED DATASET
# ============================================================

DATA_PATH = (
    "dataset/"
    "feature_engineered_used_car_dataset.csv"
)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. DEFINE FEATURES
# ============================================================

features = [
    "Brand",
    "model",
    "Age",
    "kmDriven",
    "Transmission",
    "Owner",
    "FuelType",
    "Brand_Model",
    "AgeGroup",
    "km_per_year",
    "log_kmDriven"
]

target = "AskPrice"

X = df[features]
y = df[target]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 4. CATEGORICAL + NUMERICAL FEATURES
# ============================================================

categorical_features = [
    "Brand",
    "model",
    "Transmission",
    "Owner",
    "FuelType",
    "Brand_Model",
    "AgeGroup"
]

numerical_features = [
    "Age",
    "kmDriven",
    "km_per_year",
    "log_kmDriven"
]


# ============================================================
# 5. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 6. RANDOM FOREST - NORMAL TARGET
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT A: FEATURE ENGINEERING + NORMAL TARGET")
print("=" * 70)

normal_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=25,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

normal_model.fit(
    X_train,
    y_train
)

normal_predictions = normal_model.predict(
    X_test
)

normal_mae = mean_absolute_error(
    y_test,
    normal_predictions
)

normal_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        normal_predictions
    )
)

normal_r2 = r2_score(
    y_test,
    normal_predictions
)

print("MAE :", round(normal_mae, 2))
print("RMSE:", round(normal_rmse, 2))
print("R²  :", round(normal_r2, 4))


# ============================================================
# 7. RANDOM FOREST - LOG TARGET
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT B: FEATURE ENGINEERING + LOG TARGET")
print("=" * 70)


# Transform target
y_train_log = np.log1p(y_train)


log_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=25,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


log_model.fit(
    X_train,
    y_train_log
)


# Predict log prices
log_predictions = log_model.predict(
    X_test
)


# Convert predictions back to rupees
log_predictions_original = np.expm1(
    log_predictions
)


log_mae = mean_absolute_error(
    y_test,
    log_predictions_original
)

log_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        log_predictions_original
    )
)

log_r2 = r2_score(
    y_test,
    log_predictions_original
)


print("MAE :", round(log_mae, 2))
print("RMSE:", round(log_rmse, 2))
print("R²  :", round(log_r2, 4))


# ============================================================
# 8. COMPARE RESULTS
# ============================================================

comparison = pd.DataFrame({

    "Experiment": [
        "Baseline Random Forest",
        "Feature Engineering + Normal Target",
        "Feature Engineering + Log Target"
    ],

    "MAE": [
        246979.72,
        normal_mae,
        log_mae
    ],

    "RMSE": [
        893645.38,
        normal_rmse,
        log_rmse
    ],

    "R2": [
        0.6365,
        normal_r2,
        log_r2
    ]

})


print("\n\n")
print("=" * 80)
print("FINAL EXPERIMENT COMPARISON")
print("=" * 80)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

if log_r2 > normal_r2:

    best_model = log_model
    best_name = "Feature Engineering + Log Target"

else:

    best_model = normal_model
    best_name = "Feature Engineering + Normal Target"


print("\nBest Version:", best_name)


# ============================================================
# 10. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/best_used_car_price_model_v2.pkl"
)

print(
    "\nBest Version 2 model saved successfully!"
)

print(
    "models/best_used_car_price_model_v2.pkl"
)