import pandas as pd
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("🛡️ Insurance Cost Prediction")
st.write("Enter the details below to estimate the medical insurance cost.")

st.divider()

# ---------------- LOAD DATA ----------------
data = pd.read_csv("insurance.csv")

# Encode categorical columns
sex_encoder = LabelEncoder()
smoker_encoder = LabelEncoder()
region_encoder = LabelEncoder()

data["sex"] = sex_encoder.fit_transform(data["sex"])
data["smoker"] = smoker_encoder.fit_transform(data["smoker"])
data["region"] = region_encoder.fit_transform(data["region"])

# Input and output
X = data.drop("charges", axis=1)
y = data["charges"]

# Train model
model = LinearRegression()
model.fit(X, y)

# ---------------- INPUT SECTION ----------------
st.subheader("📋 Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )

    sex = st.selectbox(
        "Sex",
        ["female", "male"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

with col2:
    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0
    )

    smoker = st.selectbox(
        "Smoker",
        ["no", "yes"]
    )

    region = st.selectbox(
        "Region",
        ["southwest", "southeast", "northwest", "northeast"]
    )

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Insurance Cost", use_container_width=True):

    sex_value = sex_encoder.transform([sex])[0]
    smoker_value = smoker_encoder.transform([smoker])[0]
    region_value = region_encoder.transform([region])[0]

    input_data = [[
        age,
        sex_value,
        bmi,
        children,
        smoker_value,
        region_value
    ]]

    prediction = model.predict(input_data)[0]

    st.success("Prediction Completed Successfully!")

    st.metric(
        label="💰 Estimated Insurance Cost",
        value=f"${prediction:,.2f}"
    )

st.divider()

st.caption(
    "This application uses Machine Learning with Scikit-learn "
    "to estimate insurance costs."
)