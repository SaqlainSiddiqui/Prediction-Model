import pandas as pd
import numpy as np
import joblib

from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "dataset/cleaned_used_car_dataset.csv"
)

print("Original dataset:", df.shape)


# ============================================================
# 2. REMOVE EXTREME KM VALUES
# ============================================================

df = df[
    df["kmDriven"] <= 300000
].copy()

print(
    "After km filtering:",
    df.shape
)


# ============================================================
# 3. FEATURES
# ============================================================

features = [
    "Brand",
    "model",
    "Age",
    "kmDriven",
    "Transmission",
    "Owner",
    "FuelType"
]

target = "AskPrice"

X = df[features].copy()
y = df[target]


# ============================================================
# 4. CATEGORICAL COLUMNS
# ============================================================

categorical_features = [
    "Brand",
    "model",
    "Transmission",
    "Owner",
    "FuelType"
]

categorical_indices = [
    X.columns.get_loc(column)
    for column in categorical_features
]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# ============================================================
# 6. CATBOOST MODEL
# ============================================================

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100,
    l2_leaf_reg=5
)


# ============================================================
# 7. TRAIN
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING CATBOOST")
print("=" * 70)

model.fit(
    X_train,
    y_train,
    cat_features=categorical_indices,
    eval_set=(X_test, y_test),
    early_stopping_rounds=100
)


# ============================================================
# 8. PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# 9. EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n")
print("=" * 70)
print("CATBOOST TEST RESULTS")
print("=" * 70)

print(
    "MAE :",
    round(mae, 2)
)

print(
    "RMSE:",
    round(rmse, 2)
)

print(
    "R²  :",
    round(r2, 4)
)


# ============================================================
# 10. COMPARE WITH RANDOM FOREST
# ============================================================

rf_mae = 215107.33
rf_rmse = 893122.78
rf_r2 = 0.6369


comparison = pd.DataFrame({

    "Metric": [
        "MAE",
        "RMSE",
        "R²"
    ],

    "Tuned Random Forest": [
        rf_mae,
        rf_rmse,
        rf_r2
    ],

    "CatBoost": [
        mae,
        rmse,
        r2
    ]
})


print("\n")
print("=" * 70)
print("RANDOM FOREST VS CATBOOST")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 11. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/catboost_used_car_price_model.pkl"
)

print("\nCatBoost model saved successfully!")