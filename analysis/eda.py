import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# LOAD CLEANED DATASET
# ==========================================

file_path = "dataset/cleaned_used_car_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==========================================
# BASIC STATISTICS
# ==========================================

print("\n========== NUMERICAL SUMMARY ==========")

print(
    df[
        ["Year", "Age", "kmDriven", "AskPrice"]
    ].describe()
)


# ==========================================
# 1. PRICE DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["AskPrice"],
    bins=50
)

plt.title("Distribution of Used Car Prices")
plt.xlabel("Ask Price (₹)")
plt.ylabel("Number of Cars")

plt.tight_layout()

plt.savefig(
    "price_distribution.png"
)

plt.show()


# ==========================================
# 2. PRICE VS CAR AGE
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Age"],
    df["AskPrice"],
    alpha=0.4
)

plt.title("Car Age vs Asking Price")
plt.xlabel("Car Age (Years)")
plt.ylabel("Ask Price (₹)")

plt.tight_layout()

plt.savefig(
    "price_vs_age.png"
)

plt.show()


# ==========================================
# 3. PRICE VS KILOMETERS DRIVEN
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["kmDriven"],
    df["AskPrice"],
    alpha=0.4
)

plt.title("Kilometers Driven vs Asking Price")
plt.xlabel("Kilometers Driven")
plt.ylabel("Ask Price (₹)")

plt.tight_layout()

plt.savefig(
    "price_vs_km.png"
)

plt.show()


# ==========================================
# 4. TOP CAR BRANDS
# ==========================================

brand_counts = (
    df["Brand"]
    .value_counts()
    .head(15)
)

plt.figure(figsize=(12, 6))

brand_counts.plot(
    kind="bar"
)

plt.title("Top 15 Car Brands by Number of Listings")
plt.xlabel("Brand")
plt.ylabel("Number of Cars")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "top_brands.png"
)

plt.show()


# ==========================================
# 5. AVERAGE PRICE BY FUEL TYPE
# ==========================================

fuel_prices = (
    df.groupby("FuelType")["AskPrice"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 6))

fuel_prices.plot(
    kind="bar"
)

plt.title("Average Car Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Average Ask Price (₹)")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "price_by_fuel.png"
)

plt.show()


# ==========================================
# 6. AVERAGE PRICE BY TRANSMISSION
# ==========================================

transmission_prices = (
    df.groupby("Transmission")["AskPrice"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 6))

transmission_prices.plot(
    kind="bar"
)

plt.title("Average Car Price by Transmission")
plt.xlabel("Transmission")
plt.ylabel("Average Ask Price (₹)")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "price_by_transmission.png"
)

plt.show()


# ==========================================
# 7. AVERAGE PRICE BY OWNER
# ==========================================

owner_prices = (
    df.groupby("Owner")["AskPrice"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 6))

owner_prices.plot(
    kind="bar"
)

plt.title("Average Car Price by Owner Type")
plt.xlabel("Owner")
plt.ylabel("Average Ask Price (₹)")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "price_by_owner.png"
)

plt.show()


# ==========================================
# 8. CORRELATION MATRIX
# ==========================================

numeric_df = df[
    ["Year", "Age", "kmDriven", "AskPrice"]
]

correlation = numeric_df.corr()

print("\n========== CORRELATION MATRIX ==========")

print(correlation)


plt.figure(figsize=(8, 6))

plt.imshow(
    correlation,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "correlation_matrix.png"
)

plt.show()


print("\nEDA completed successfully!")