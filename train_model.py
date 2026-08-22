import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "dataset/cleaned_used_car_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# ============================================================
# 2. REMOVE EXTREME KM VALUES
# ============================================================

df = df[df["kmDriven"] <= 300000].copy()

print("Dataset shape after km filtering:", df.shape)


# ============================================================
# 3. SELECT FEATURES
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

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 5. DEFINE FEATURE TYPES
# ============================================================

categorical_features = [
    "Brand",
    "model",
    "Transmission",
    "Owner",
    "FuelType"
]

numerical_features = [
    "Age",
    "kmDriven"
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
# 7. DEFINE MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42,
            max_depth=20,
            min_samples_leaf=2
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            max_depth=25,
            min_samples_leaf=2
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
}


# ============================================================
# 8. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}


for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Metrics
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

    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R²  :", round(r2, 4))

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[model_name] = pipeline


# ============================================================
# 9. DISPLAY MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

print("\n\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# ============================================================
# 10. SAVE BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\nBest Model:", best_model_name)

joblib.dump(
    best_model,
    "models/best_used_car_price_model.pkl"
)

print(
    "\nBest model saved successfully!"
)

print(
    "models/best_used_car_price_model.pkl"
)