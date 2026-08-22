import pandas as pd
import joblib

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    "models/tuned_used_car_price_model.pkl"
)

print("Model loaded successfully!")


# ============================================================
# CREATE SAMPLE CAR
# ============================================================

sample_car = pd.DataFrame({

    "Brand": ["Maruti Suzuki"],

    "model": ["Swift"],

    "Age": [5],

    "kmDriven": [45000],

    "Transmission": ["Manual"],

    "Owner": ["first"],

    "FuelType": ["Petrol"]
})


# ============================================================
# PREDICT
# ============================================================

prediction = model.predict(
    sample_car
)


# ============================================================
# DISPLAY RESULT
# ============================================================

predicted_price = prediction[0]

print("\n==============================")
print("USED CAR PRICE PREDICTION")
print("==============================")

print(
    "Predicted Price: ₹",
    f"{predicted_price:,.0f}"
)