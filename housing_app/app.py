
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin


# Page configuration

st.set_page_config(page_title="California Housing Predictor")

st.title("California Housing Price Predictor")
st.write("Enter housing information to estimate the median house value.")


# Custom transformer
# Must be defined before loading the pipeline

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_ix = 3
        bedrooms_ix = 4
        population_ix = 5
        households_ix = 6

        rooms_per_household = (
            X[:, rooms_ix] / X[:, households_ix]
        )
        population_per_household = (
            X[:, population_ix] / X[:, households_ix]
        )

        if self.add_bedrooms_per_room:
            bedrooms_per_room = (
                X[:, bedrooms_ix] / X[:, rooms_ix]
            )

            return np.c_[
                X,
                rooms_per_household,
                population_per_household,
                bedrooms_per_room
            ]

        return np.c_[
            X,
            rooms_per_household,
            population_per_household
        ]


# Load saved model and pipeline

APP_DIR = Path(__file__).resolve().parent

pipeline_path = APP_DIR / "housing_pipeline.pkl"
model_path = APP_DIR / "final_housing_model.pkl"

pipeline = joblib.load(pipeline_path)
model = joblib.load(model_path)


# User input form

with st.form("housing_form"):
    longitude = st.number_input(
        "Longitude", value=-122.0
    )

    latitude = st.number_input(
        "Latitude", value=37.0
    )

    housing_median_age = st.number_input(
        "Housing median age",
        min_value=1.0,
        value=30.0
    )

    total_rooms = st.number_input(
        "Total rooms",
        min_value=1.0,
        value=2000.0
    )

    total_bedrooms = st.number_input(
        "Total bedrooms",
        min_value=1.0,
        value=400.0
    )

    population = st.number_input(
        "Population",
        min_value=1.0,
        value=1000.0
    )

    households = st.number_input(
        "Households",
        min_value=1.0,
        value=350.0
    )

    median_income = st.number_input(
        "Median income (dataset units)",
        min_value=0.0,
        value=4.0
    )

    ocean_proximity = st.selectbox(
        "Ocean proximity",
        [
            "<1H OCEAN",
            "INLAND",
            "ISLAND",
            "NEAR BAY",
            "NEAR OCEAN"
        ]
    )

    submitted = st.form_submit_button(
        "Predict house value"
    )


# Make prediction

if submitted:
    input_data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    try:
        prepared_data = pipeline.transform(input_data)
        prediction = model.predict(prepared_data)[0]

        st.success(
            f"Predicted median house value: ${prediction:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")



# -----------------------------
# Batch prediction from CSV
# -----------------------------
st.divider()
st.header("Batch Housing Price Prediction")

st.write(
    "Upload a CSV containing raw housing features to predict "
    "median house values for multiple records."
)

uploaded_file = st.file_uploader(
    "Choose a housing CSV file",
    type=["csv"],
    key="batch_csv"
)

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)

        required_columns = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "ocean_proximity"
        ]

        missing_columns = [
            column for column in required_columns
            if column not in batch_data.columns
        ]

        if missing_columns:
            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        elif batch_data.empty:
            st.error("The uploaded CSV contains no records.")

        else:
            st.write(f"Records uploaded: {len(batch_data):,}")
            st.dataframe(batch_data.head(10))

            if st.button("Predict All House Values"):
                actual_values = (
                    batch_data["median_house_value"]
                    if "median_house_value" in batch_data.columns
                    else None
                )

                X_batch = batch_data[required_columns].copy()

                with st.spinner("Predicting housing prices..."):
                    prepared_batch = pipeline.transform(X_batch)
                    predictions = model.predict(prepared_batch)

                results = batch_data.copy()
                results["predicted_median_house_value"] = predictions

                st.success(
                    f"Predictions completed for {len(results):,} records."
                )

                st.dataframe(results.head(100))

                if actual_values is not None:
                    from sklearn.metrics import (
                        mean_absolute_error,
                        mean_squared_error
                    )

                    if actual_values.isna().any():
                        st.warning(
                            "Some actual prices are missing; "
                            "metrics are calculated only on rows "
                            "with actual prices."
                        )

                    valid = actual_values.notna()

                    if valid.any():
                        mae = mean_absolute_error(
                            actual_values[valid],
                            predictions[valid]
                        )

                        rmse = np.sqrt(
                            mean_squared_error(
                                actual_values[valid],
                                predictions[valid]
                            )
                        )

                        col1, col2 = st.columns(2)
                        col1.metric("MAE", f"${mae:,.2f}")
                        col2.metric("RMSE", f"${rmse:,.2f}")

                csv_output = results.to_csv(index=False)

                st.download_button(
                    "Download all predictions",
                    data=csv_output,
                    file_name="housing_predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
