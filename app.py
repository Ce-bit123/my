
import streamlit as st

import pandas as pd
import joblib

# Title
st.header("Risk prediction of early neurological deterioration within 72 hours after thrombolytic therapy in ischemic stroke")

# Input bar 1
time1 = st.number_input("Enter Time from onset to admission")

# Input bar 2
NIHSS_score_before_thrombolysis = st.number_input("Enter NIHSS score before thrombolysis")
NIHSS_score_after_thrombolysis1 = st.number_input("Enter NIHSS score after thrombolysis")
mean_corpuscular_hemoglobin = st.number_input("Enter Mean corpuscular hemoglobin")
Thrombin_time = st.number_input("Enter Thrombin time")
fasting_blood_glucose= st.number_input("Enter Fasting blood glucose")
# Dropdown input
Diabetes_mellitus = st.selectbox("Whether you have diabetes", ("Yes", "No"))

# If button is pressed
if st.button("Submit"):
    # Unpickle classifier
    clf = joblib.load("clf.pkl")

    # Store inputs into dataframe
    X = pd.DataFrame([[time1, NIHSS_score_before_thrombolysis, NIHSS_score_after_thrombolysis1,mean_corpuscular_hemoglobin,
                       Thrombin_time,fasting_blood_glucose,Diabetes_mellitus]],
                     columns=["time1", "NIHSS_score_before_thrombolysis", "NIHSS_score_after_thrombolysis1","mean_corpuscular_hemoglobin",
                       "Thrombin_time","fasting_blood_glucose","Diabetes_mellitus"])
    X = X.replace(["Yes", "No"], [1, 0])

    # Get prediction
    prediction = clf.predict(X)[0]

    # Output prediction
    st.text(f"This patient has a higher probability of {prediction} within 72 hours")

