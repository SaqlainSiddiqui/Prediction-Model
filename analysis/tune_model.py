import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from scipy.stats import randint


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "dataset/cleaned_used_car_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Original dataset:", df.shape)


# ============================================================
# 2. REMOVE EXTREME KM VALUES
# ============================================================

df = df[df["kmDriven"] <= 300000].copy()

print(
    "After km filtering:",
    df.shape
)


# ============================================================
# 3. FEATURES AND TARGET
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

X = df[features]
y = df[target]


# ============================================================
# 4. TRAIN / TEST SPLIT
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
# 5. FEATURE TYPES
# ============================================================

categorical_features = [
    "Brand",
    "model",
    "Transmission",
    "Owner",
    "FuelType"
]


# ============================================================
# 6. PREPROCESSING
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
# 7. RANDOM FOREST PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# 8. HYPERPARAMETER SEARCH SPACE
# ============================================================

param_distributions = {

    "model__n_estimators":
        randint(200, 601),

    "model__max_depth":
        [None, 10, 15, 20, 25, 30, 35, 40],

    "model__min_samples_split":
        randint(2, 11),

    "model__min_samples_leaf":
        randint(1, 6),

    "model__max_features":
        ["sqrt", "log2", 0.5, 0.7, 1.0]
}


# ============================================================
# 9. RANDOMIZED SEARCH
# ============================================================

print("\n")
print("=" * 70)
print("STARTING HYPERPARAMETER TUNING")
print("=" * 70)

search = RandomizedSearchCV(

    estimator=pipeline,

    param_distributions=param_distributions,

    n_iter=12,

    scoring="neg_mean_absolute_error",

    cv=3,

    verbose=2,

    random_state=42,

    n_jobs=-1
)


search.fit(
    X_train,
    y_train
)


# ============================================================
# 10. BEST PARAMETERS
# ============================================================

print("\n")
print("=" * 70)
print("BEST HYPERPARAMETERS")
print("=" * 70)

print(
    search.best_params_
)


print("\nBest CV MAE:")

print(
    -search.best_score_
)


# ============================================================
# 11. EVALUATE ON TEST SET
# ============================================================

best_model = search.best_estimator_

predictions = best_model.predict(
    X_test
)


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
print("TUNED MODEL TEST RESULTS")
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
# 12. BASELINE COMPARISON
# ============================================================

baseline_mae = 246979.72
baseline_rmse = 893645.38
baseline_r2 = 0.6365


print("\n")
print("=" * 70)
print("BASELINE VS TUNED MODEL")
print("=" * 70)

comparison = pd.DataFrame({

    "Metric": [
        "MAE",
        "RMSE",
        "R²"
    ],

    "Baseline Random Forest": [
        baseline_mae,
        baseline_rmse,
        baseline_r2
    ],

    "Tuned Random Forest": [
        mae,
        rmse,
        r2
    ]
})


print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 13. SAVE TUNED MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/tuned_used_car_price_model.pkl"
)

print("\n")
print(
    "Tuned model saved successfully!"
)

print(
    "models/tuned_used_car_price_model.pkl"
)