import pandas as pd

df = pd.read_csv(
    "dataset/cleaned_used_car_dataset.csv"
)

print("========== KM DRIVEN DISTRIBUTION ==========")

limits = [
    100000,
    150000,
    200000,
    300000,
    400000,
    500000,
    700000,
    900000
]

for limit in limits:
    count = (df["kmDriven"] > limit).sum()
    
    print(
        f"Cars above {limit:,} km: {count}"
    )


print("\n========== EXTREME KM RECORDS ==========")

extreme = (
    df[df["kmDriven"] > 300000]
    [
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
)

print(extreme.to_string(index=False))


print("\n========== AGE vs KM CHECK ==========")

df["km_per_year"] = (
    df["kmDriven"] /
    df["Age"].replace(0, 1)
)

print(
    df[
        [
            "Brand",
            "model",
            "Age",
            "kmDriven",
            "km_per_year",
            "AskPrice"
        ]
    ]
    .sort_values(
        "km_per_year",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)