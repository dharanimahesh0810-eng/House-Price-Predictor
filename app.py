import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# Load the trained model and the list of locations
# (these files must be in the same folder as this script)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("house_price_model.pkl")
    locations = joblib.load("locations.pkl")
    return model, locations

model, locations = load_model()

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 Bengaluru House Price Predictor")
st.write("Enter the property details below to estimate its price (in Lakhs ₹).")

# ------------------------------------------------------------
# Input widgets
# ------------------------------------------------------------
location = st.selectbox("Location", options=locations)

total_sqft = st.number_input(
    "Total Area (sqft)",
    min_value=200.0,
    max_value=10000.0,
    value=1200.0,
    step=50.0
)

bath = st.slider("Number of Bathrooms", min_value=1, max_value=10, value=2)

bedrooms = st.slider("Number of Bedrooms (BHK)", min_value=1, max_value=10, value=2)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if st.button("Predict Price"):
    input_data = {
        "location": location,
        "total_sqft": total_sqft,
        "bath": bath,
        "bedrooms": bedrooms
    }
    input_df = pd.DataFrame([input_data])

    predicted_price = model.predict(input_df)[0]

    st.success(f"Estimated Price: ₹ {predicted_price:.2f} Lakhs")
    st.caption("Note: This is an estimate based on a linear regression model and historical data.")
