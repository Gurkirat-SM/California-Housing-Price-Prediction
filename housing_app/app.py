
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import joblib
import streamlit as st
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(page_title="California Housing Predictor")

st.title("California Housing Price Predictor")
st.write("Enter housing information to estimate the median house value.")


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

        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
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


pipeline = joblib.load("housing_pipeline.pkl")
model = joblib.load("final_housing_model.pkl")

with st.form("housing_form"):
    longitude = st.number_input("Longitude", value=-122.0)
    latitude = st.number_input("Latitude", value=37.0)
    housing_median_age = st.number_input(
        "Housing median age", min_value=1.0, value=30.0
    )
    total_rooms = st.number_input(
        "Total rooms", min_value=1.0, value=2000.0
    )
    total_bedrooms = st.number_input(
        "Total bedrooms", min_value=1.0, value=400.0
    )
    population = st.number_input(
        "Population", min_value=1.0, value=1000.0
    )
    households = st.number_input(
        "Households", min_value=1.0, value=350.0
    )
    median_income = st.number_input(
        "Median income (dataset units)",
        min_value=0.0,
        value=4.0
    )
    ocean_proximity = st.selectbox(
        "Ocean proximity",
        ["<1H OCEAN", "INLAND", "ISLAND",
         "NEAR BAY", "NEAR OCEAN"]
    )

    submitted = st.form_submit_button("Predict house value")

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

    prepared_data = pipeline.transform(input_data)
    prediction = model.predict(prepared_data)[0]

    st.success(f"Predicted median house value: ${prediction:,.2f}")
