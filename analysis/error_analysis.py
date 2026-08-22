import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "dataset/cleaned_used_car_dataset.csv"
)

# Same filtering used during training
df = df[
    df["kmDriven"] <= 300000
].copy()


# ============================================================
# 2. FEATURES / TARGET
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
# 3. SAME TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 4. LOAD TUNED MODEL
# ============================================================

model = joblib.load(
    "models/tuned_used_car_price_model.pkl"
)


# ============================================================
# 5. PREDICT
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# 6. CREATE RESULTS TABLE
# ============================================================

results = X_test.copy()

results["ActualPrice"] = y_test.values

results["PredictedPrice"] = predictions

results["AbsoluteError"] = abs(
    results["ActualPrice"]
    -
    results["PredictedPrice"]
)

results["PercentageError"] = (
    results["AbsoluteError"]
    /
    results["ActualPrice"]
) * 100


# ============================================================
# 7. WORST PREDICTIONS
# ============================================================

worst_predictions = (
    results
    .sort_values(
        "AbsoluteError",
        ascending=False
    )
    .head(20)
)


print("\n")
print("=" * 90)
print("20 LARGEST PREDICTION ERRORS")
print("=" * 90)

print(
    worst_predictions[
        [
            "Brand",
            "model",
            "Age",
            "kmDriven",
            "Transmission",
            "FuelType",
            "ActualPrice",
            "PredictedPrice",
            "AbsoluteError",
            "PercentageError"
        ]
    ].to_string(index=False)
)


# ============================================================
# 8. BEST PREDICTIONS
# ============================================================

best_predictions = (
    results
    .sort_values(
        "AbsoluteError"
    )
    .head(20)
)


print("\n")
print("=" * 90)
print("20 BEST PREDICTIONS")
print("=" * 90)

print(
    best_predictions[
        [
            "Brand",
            "model",
            "Age",
            "kmDriven",
            "ActualPrice",
            "PredictedPrice",
            "AbsoluteError",
            "PercentageError"
        ]
    ].to_string(index=False)
)


# ============================================================
# 9. ERROR BY PRICE RANGE
# ============================================================

def price_range(price):

    if price < 300000:
        return "Below ₹3 Lakh"

    elif price < 600000:
        return "₹3–6 Lakh"

    elif price < 1000000:
        return "₹6–10 Lakh"

    elif price < 2000000:
        return "₹10–20 Lakh"

    else:
        return "Above ₹20 Lakh"


results["PriceRange"] = (
    results["ActualPrice"]
    .apply(price_range)
)


range_error = (
    results
    .groupby("PriceRange")
    .agg(
        Cars=("ActualPrice", "count"),
        MAE=("AbsoluteError", "mean")
    )
    .sort_values("MAE")
)


print("\n")
print("=" * 90)
print("ERROR BY PRICE RANGE")
print("=" * 90)

print(
    range_error.to_string()
)


# ============================================================
# 10. OVERALL ERROR
# ============================================================

print("\n")
print("=" * 90)
print("OVERALL ERROR ANALYSIS")
print("=" * 90)

print(
    "Mean Absolute Error:",
    round(
        results["AbsoluteError"].mean(),
        2
    )
)

print(
    "Median Absolute Error:",
    round(
        results["AbsoluteError"].median(),
        2
    )
)

print(
    "Mean Percentage Error:",
    round(
        results["PercentageError"].mean(),
        2
    ),
    "%"
)

print(
    "Median Percentage Error:",
    round(
        results["PercentageError"].median(),
        2
    ),
    "%"
)