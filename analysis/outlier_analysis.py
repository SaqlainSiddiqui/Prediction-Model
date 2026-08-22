import pandas as pd

# Load cleaned dataset
df = pd.read_csv(
    "dataset/cleaned_used_car_dataset.csv"
)

# ==========================================
# PRICE OUTLIERS
# ==========================================

Q1_price = df["AskPrice"].quantile(0.25)
Q3_price = df["AskPrice"].quantile(0.75)

IQR_price = Q3_price - Q1_price

lower_price = Q1_price - 1.5 * IQR_price
upper_price = Q3_price + 1.5 * IQR_price

price_outliers = df[
    (df["AskPrice"] < lower_price) |
    (df["AskPrice"] > upper_price)
]

print("========== PRICE OUTLIER ANALYSIS ==========")

print("Q1:", Q1_price)
print("Q3:", Q3_price)
print("IQR:", IQR_price)
print("Lower Bound:", lower_price)
print("Upper Bound:", upper_price)

print("\nNumber of price outliers:")
print(len(price_outliers))

print("\nMost expensive cars:")
print(
    df[
        [
            "Brand",
            "model",
            "Age",
            "kmDriven",
            "AskPrice"
        ]
    ]
    .sort_values(
        "AskPrice",
        ascending=False
    )
    .head(20)
)


# ==========================================
# KM DRIVEN OUTLIERS
# ==========================================

Q1_km = df["kmDriven"].quantile(0.25)
Q3_km = df["kmDriven"].quantile(0.75)

IQR_km = Q3_km - Q1_km

lower_km = Q1_km - 1.5 * IQR_km
upper_km = Q3_km + 1.5 * IQR_km

km_outliers = df[
    (df["kmDriven"] < lower_km) |
    (df["kmDriven"] > upper_km)
]

print("\n========== KM DRIVEN OUTLIER ANALYSIS ==========")

print("Q1:", Q1_km)
print("Q3:", Q3_km)
print("IQR:", IQR_km)
print("Lower Bound:", lower_km)
print("Upper Bound:", upper_km)

print("\nNumber of kmDriven outliers:")
print(len(km_outliers))

print("\nHighest kmDriven cars:")
print(
    df[
        [
            "Brand",
            "model",
            "Age",
            "kmDriven",
            "AskPrice"
        ]
    ]
    .sort_values(
        "kmDriven",
        ascending=False
    )
    .head(20)
)