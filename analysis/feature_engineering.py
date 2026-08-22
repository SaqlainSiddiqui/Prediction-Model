import pandas as pd
import numpy as np

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "dataset/cleaned_used_car_dataset.csv"
)

print("Original shape:", df.shape)


# ============================================================
# REMOVE EXTREME KM VALUES
# ============================================================

df = df[df["kmDriven"] <= 300000].copy()

print(
    "After km filtering:",
    df.shape
)


# ============================================================
# REMOVE YEAR
# ============================================================

df = df.drop(
    columns=["Year"]
)


# ============================================================
# BRAND + MODEL FEATURE
# ============================================================

df["Brand_Model"] = (
    df["Brand"].str.strip()
    + "_"
    + df["model"].str.strip()
)


# ============================================================
# AGE GROUP
# ============================================================

def age_category(age):

    if age <= 2:
        return "New"

    elif age <= 5:
        return "Recent"

    elif age <= 10:
        return "Mid"

    else:
        return "Old"


df["AgeGroup"] = df["Age"].apply(
    age_category
)


# ============================================================
# KM PER YEAR
# ============================================================

df["km_per_year"] = (
    df["kmDriven"] /
    df["Age"].replace(0, 1)
)


# ============================================================
# LOG TRANSFORMED FEATURES
# ============================================================

df["log_kmDriven"] = np.log1p(
    df["kmDriven"]
)


# ============================================================
# DISPLAY
# ============================================================

print("\n========== NEW FEATURES ==========")

print(
    df[
        [
            "Brand",
            "model",
            "Age",
            "kmDriven",
            "Brand_Model",
            "AgeGroup",
            "km_per_year",
            "log_kmDriven",
            "AskPrice"
        ]
    ].head()
)


print("\n========== DATA TYPES ==========")

print(df.dtypes)


# ============================================================
# SAVE FEATURE-ENGINEERED DATASET
# ============================================================

output_path = (
    "dataset/"
    "feature_engineered_used_car_dataset.csv"
)

df.to_csv(
    output_path,
    index=False
)

print(
    "\nFeature engineered dataset saved:"
)

print(output_path)