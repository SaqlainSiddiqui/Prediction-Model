import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

file_path = "dataset/used_car_dataset.csv"

df = pd.read_csv(file_path)

print("Original dataset shape:")
print(df.shape)


# ==========================================
# 2. REMOVE DUPLICATES
# ==========================================

print("\nDuplicate rows before removal:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(df.shape)


# ==========================================
# 3. CLEAN kmDriven
# ==========================================

df["kmDriven"] = (
    df["kmDriven"]
    .str.replace(" km", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["kmDriven"] = pd.to_numeric(
    df["kmDriven"],
    errors="coerce"
)


# ==========================================
# 4. HANDLE MISSING kmDriven
# ==========================================

print("\nMissing kmDriven values:")
print(df["kmDriven"].isnull().sum())

df["kmDriven"] = df["kmDriven"].fillna(
    df["kmDriven"].median()
)


# ==========================================
# 5. CLEAN AskPrice
# ==========================================

df["AskPrice"] = (
    df["AskPrice"]
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["AskPrice"] = pd.to_numeric(
    df["AskPrice"],
    errors="coerce"
)


# ==========================================
# 6. REMOVE INVALID PRICE VALUES
# ==========================================

df = df.dropna(subset=["AskPrice"])

df = df[df["AskPrice"] > 0]


# ==========================================
# 7. DROP UNUSED COLUMNS
# ==========================================

df = df.drop(
    columns=["PostedDate", "AdditionInfo"]
)


# ==========================================
# 8. DISPLAY RESULT
# ==========================================

print("\n========== CLEANED DATASET ==========")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 9. SAVE CLEAN DATASET
# ==========================================

output_file = "dataset/cleaned_used_car_dataset.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nCleaned dataset saved successfully!")
print(output_file)