from flask import Flask, render_template, request
import pandas as pd
import joblib


# ============================================================
# CREATE FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "models/tuned_used_car_price_model.pkl"

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# LOAD DATASET FOR DROPDOWN OPTIONS
# ============================================================

DATA_PATH = "dataset/cleaned_used_car_dataset.csv"

df = pd.read_csv(DATA_PATH)


# Get unique values for dropdowns
brands = sorted(
    df["Brand"].dropna().unique().tolist()
)

brand_models = {}

for brand in brands:

    brand_models[brand] = sorted(
        df.loc[
            df["Brand"] == brand,
            "model"
        ]
        .dropna()
        .unique()
        .tolist()
    )

transmissions = sorted(
    df["Transmission"].dropna().unique().tolist()
)

owners = sorted(
    df["Owner"].dropna().unique().tolist()
)

fuel_types = sorted(
    df["FuelType"].dropna().unique().tolist()
)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        brands=brands,
        brand_models=brand_models,
        transmissions=transmissions,
        owners=owners,
        fuel_types=fuel_types
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # GET FORM DATA
        # ----------------------------------------------------

        brand = request.form["brand"]

        model_name = request.form["model"]

        age = int(
            request.form["age"]
        )

        km_driven = float(
            request.form["kmDriven"]
        )

        transmission = request.form[
            "transmission"
        ]

        owner = request.form[
            "owner"
        ]

        fuel_type = request.form[
            "fuelType"
        ]


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if age < 0:

            return render_template(
                "result.html",
                error="Car age cannot be negative."
            )


        if km_driven < 0:

            return render_template(
                "result.html",
                error="Kilometers driven cannot be negative."
            )


        if km_driven > 300000:

            return render_template(
                "result.html",
                error=(
                    "Kilometers driven must be "
                    "300,000 km or less."
                )
            )


        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "Brand": [brand],

            "model": [model_name],

            "Age": [age],

            "kmDriven": [km_driven],

            "Transmission": [transmission],

            "Owner": [owner],

            "FuelType": [fuel_type]

        })


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        # ----------------------------------------------------
        # FORMAT PRICE
        # ----------------------------------------------------

        predicted_price = max(
            0,
            prediction
        )


        formatted_price = (
            f"₹{predicted_price:,.0f}"
        )


        # ----------------------------------------------------
        # RESULT PAGE
        # ----------------------------------------------------

        return render_template(

            "result.html",

            predicted_price=formatted_price,

            brand=brand,

            model_name=model_name,

            age=age,

            km_driven=km_driven,

            transmission=transmission,

            owner=owner,

            fuel_type=fuel_type

        )


    except Exception as e:

        return render_template(

            "result.html",

            error=(
                "An error occurred while "
                "making the prediction: "
                + str(e)
            )
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )