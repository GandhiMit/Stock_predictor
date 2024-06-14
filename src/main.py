import streamlit as st
import numpy as np
import pandas as pd
from src import data, model, plot



# Default values
DEFAULT_COMPANY = "TCS.NS"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-06-10"
DEFAULT_START_DATE_PREDICTION = "2024-03-07"
DEFAULT_END_DATE_PREDICTION = "2024-06-11"
DEFAULT_FACTOR = 28


def main():
    # User inputs
    asset_type = st.selectbox("Asset Type", ["Stock", "Cryptocurrency"], index=0)
    company = st.text_input("Company/Crypto Symbol", value=DEFAULT_COMPANY)
    start_date = st.date_input("Start Date for Training Data", value=pd.to_datetime(DEFAULT_START_DATE))
    end_date = st.date_input("End Date for Training Data", value=pd.to_datetime(DEFAULT_END_DATE))
    start_date_prediction = st.date_input("Start Date for Prediction Data", value=pd.to_datetime(DEFAULT_START_DATE_PREDICTION))
    end_date_prediction = st.date_input("End Date for Prediction Data", value=pd.to_datetime(DEFAULT_END_DATE_PREDICTION))
    factor = st.number_input("MACD-Days", value=DEFAULT_FACTOR)
    price_type = st.selectbox("Price Type", ["Open", "Close", "High", "Low"], index=1)

    # Add an OK button to start the process
    if st.button("OK"):
        # Display a spinner while the model is training
        with st.spinner('Training the model...'):
            run_model(asset_type, company, start_date, end_date, start_date_prediction, end_date_prediction, factor, price_type)

if __name__ == "__main__":
    st.title("Stock and Cryptocurrency Price Prediction")
    main()
